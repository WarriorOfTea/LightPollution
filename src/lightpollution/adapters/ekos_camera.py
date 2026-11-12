"""Interoperate with Ekos camera-data XML instead of a second sensor database."""

from copy import deepcopy
import hashlib
from xml.etree import ElementTree as ET

import numpy as np

from ..models.camera import camera_config,_number


def import_ekos_camera(xml,base_profile=None,read_mode=0,gain=0):
    if not isinstance(xml,str) or len(xml.encode())>512000 or '<!DOCTYPE' in xml.upper() or '<!ENTITY' in xml.upper():
        raise ValueError('Use camera-data XML up to512KB without DTD/entities.')
    try:root=ET.fromstring(xml)
    except ET.ParseError as exc:raise ValueError('Invalid camera-data XML.') from exc
    if root.tag!='ExposureCalculatorCameraData' or root.findtext('CameraDataClassVersion')!='1':
        raise ValueError('Only Ekos ExposureCalculatorCameraData version1 is supported.')
    def numeric(element,name,lo,hi):
        try:value=float(element.findtext(name))
        except (ValueError,TypeError):raise ValueError('Missing/invalid XML '+name) from None
        return _number(value,name,lo,hi)
    camera_id=root.findtext('CameraId')
    if not camera_id or len(camera_id)>120:raise ValueError('Missing/invalid camera ID.')
    _number(read_mode,'read_mode',0,100,True);_number(gain,'gain',-100,100000)
    modes=[m for m in root.findall('CameraGainReadMode') if numeric(m,'GainReadModeNumber',0,100)==read_mode]
    if len(modes)!=1:raise ValueError('Select an existing unique Ekos read mode.')
    mode=modes[0];values=sorted((numeric(v,'Gain',-100,100000),numeric(v,'ReadNoise',0,100))
        for v in mode.findall('./CameraGainReadNoise/GainReadNoiseValue'))
    if not values or len({g for g,n in values})!=len(values) or not values[0][0]<=gain<=values[-1][0]:
        raise ValueError('Gain must lie inside the read-noise table; no extrapolation.')
    noise=float(np.interp(gain,[g for g,n in values],[n for g,n in values]))
    profile=camera_config({'profile':deepcopy(base_profile or {})})['profile']
    profile.update(profile_id=f'Ekos-{camera_id}-mode{read_mode}-gain{gain:g}-partial',origin='ekos_partial')
    profile['sensor'].update(pixel_pitch_um=numeric(root,'CameraPixelSize',.5,30),read_noise_e=noise)
    profile['provenance']={**profile['provenance'],'ekos_camera_id':camera_id,
        'ekos_read_mode':read_mode,'ekos_read_mode_name':mode.findtext('GainReadModeName') or '',
        'ekos_gain_control_value':gain,'ekos_XML_sha256':hashlib.sha256(xml.encode()).hexdigest(),
        'ekos_scalar_QE_hint':numeric(root,'CameraQuantumEfficiency',0,1),
        'ekos_sensor_type':root.findtext('SensorType'),'imported':['pixel_pitch_um','read_noise_e'],
        'not_provided':['gain_e_per_adu','full_well_e','dark_current_e_per_s','dimensions','QE_wavelength_curve','CFA'],
        'response_preserved_from_explicit_base_profile':True,'hardware_mode_or_capture_queue_changed':False}
    return camera_config({'profile':profile})['profile']
