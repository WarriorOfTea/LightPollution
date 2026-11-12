"""Camera projection, normalized Gaussian pixel integrals and adaptive rate field."""

from math import erf

import numpy as np

from ..skyview import camera_basis
from .eye_field import interpolate_patch,patch_support
from .eye_optics import _normal_cdf,_gaussian_cdf_grid,pixel_solid_angles


def basis(view):
    forward,right,up=camera_basis(view['azimuth_deg'],view['altitude_deg'])
    angle=np.deg2rad(view['rotation_deg']);c,s=np.cos(angle),np.sin(angle)
    return forward,c*right+s*up,-s*right+c*up


def directions(view,x,y):
    x,y=np.broadcast_arrays(x,y);forward,right,up=basis(view)
    edge=np.tan(np.deg2rad(view['horizontal_fov_deg'])/2)
    ray=forward+(x*edge)[...,None]*right+(y*edge*view['height']/view['width'])[...,None]*up
    ray/=np.linalg.norm(ray,axis=-1)[...,None]
    return np.rad2deg(np.arctan2(ray[...,0],ray[...,1]))%360,np.rad2deg(np.arcsin(ray[...,2]))


def pixels(view):
    x=2*(np.arange(view['width'])+.5)/view['width']-1
    y=1-2*(np.arange(view['height'])+.5)/view['height']
    return np.meshgrid(x,y)


def project_star(view,az,alt):
    a,h=np.deg2rad([az,alt]);ray=np.array([np.cos(h)*np.sin(a),np.cos(h)*np.cos(a),np.sin(h)])
    forward,right,up=basis(view);depth=ray@forward
    if depth<=0:return None
    x,y=(ray@right)/depth,(ray@up)/depth
    edge=np.tan(np.deg2rad(view['horizontal_fov_deg'])/2);step=2*edge/view['width']
    px=(x/edge+1)*view['width']/2-.5;py=view['height']/2-y*view['width']/(2*edge)-.5
    scales=step/np.sqrt((1+x*x+y*y)*np.array([1+x*x,1+y*y]))
    correlation=-x*y/np.sqrt((1+x*x)*(1+y*y))
    return px,py,scales,correlation


def deposit_electrons(buffer,position,total_e,fwhm_arcsec):
    px,py,scales,rho=position;sigma=np.deg2rad(fwhm_arcsec/3600)/np.sqrt(8*np.log(2))
    sx,sy=sigma/scales;height,width=buffer.shape
    x0,x1=max(0,int(np.floor(px-6*sx))),min(width,int(np.ceil(px+6*sx))+1)
    y0,y1=max(0,int(np.floor(py-6*sy))),min(height,int(np.ceil(py+6*sy))+1)
    deposited=0.
    if x1>x0 and y1>y0:
        xx=(np.arange(x0,x1+1)-.5-px)/sx;yy=(np.arange(y0,y1+1)-.5-py)/sy
        if abs(rho)<1e-10:mass=np.outer(np.diff(_normal_cdf(yy)),np.diff(_normal_cdf(xx)))
        else:mass=np.diff(np.diff(_gaussian_cdf_grid(xx,yy,rho),axis=0),axis=1)
        mass=np.maximum(0,mass);buffer[y0:y1,x0:x1]+=total_e*mass
        deposited=total_e*float(mass.sum())
    return {'input_e':float(total_e),'deposited_e':float(deposited),
        'outside_frame_or_truncated_e':float(max(0,total_e-deposited)),
        'gaussian_tail_upper_bound_e':float(total_e*2*(1-erf(6/np.sqrt(2))))}


HELDOUT=np.array([[-.83,-.31],[-.27,.79],[.41,-.71],[.89,.29],[-.63,.57],[.19,.91],[.73,-.43],[-.11,-.89]])


def build_camera_patch(view,evaluate,sampling,check=lambda:None):
    cache={};levels=[];floors=np.array([1e-12,1e-12,1e-10])
    def at(x,y):
        key=(float(x),float(y))
        if key not in cache:
            check();az,alt=directions(view,x,y)
            cache[key]=evaluate(float(az),float(alt)) if alt>=10 else [0.,0.,0.,False]
        return cache[key]
    size=3
    while True:
        axis=np.linspace(-1,1,size);nodes=np.asarray([[at(x,y) for x in axis] for y in axis])
        patch={'view':view,'x':axis.tolist(),'y':axis.tolist(),
            'city_response_per_m2_sr_s':nodes[...,0].tolist(),
            'unresolved_response_per_m2_sr_s':nodes[...,1].tolist(),
            'total_V_relative':nodes[...,2].tolist(),'physical_supported':nodes[...,3].astype(bool).tolist()}
        mid=(axis[1:]+axis[:-1])/2;points=np.array([[x,y] for y in mid for x in mid])
        truth=np.asarray([at(*p) for p in points]);predicted=np.stack([
            interpolate_patch(patch,nodes[...,j],*points.T) for j in range(3)],axis=-1)
        valid=truth[:,3].astype(bool)&patch_support(patch,*points.T)
        errors=abs(truth[:,:3]-predicted)/np.maximum(abs(truth[:,:3]),floors)
        maximum=float(errors[valid].max()) if valid.any() else None
        levels.append({'size':size,'probes':len(points),'supported':int(valid.sum()),'max_relative_error':maximum})
        if maximum is None or maximum<=sampling['target_relative_error'] or 2*size-1>sampling['max_grid_size']:break
        size=2*size-1
    patch['refinement']={'levels':levels,'target_relative_error':sampling['target_relative_error'],
        'target_met':maximum is not None and maximum<=sampling['target_relative_error'],'near_zero_floors':floors.tolist(),
        'source_coefficient_refitted':False,'node_units':'detected photons per m2/sr/s before scalar optical throughput'}
    if sampling['validation']=='heldout_8':
        rows=[]
        for x,y in HELDOUT:
            truth=np.asarray(at(x,y));prediction=np.array([interpolate_patch(patch,nodes[...,j],x,y) for j in range(3)])
            valid=bool(truth[3] and patch_support(patch,x,y));az,alt=directions(view,x,y)
            rows.append({'x':x,'y':y,'azimuth_deg':float(az),'altitude_deg':float(alt),'supported':valid,
                'direct':truth[:3].tolist(),'interpolated':prediction.tolist(),
                'relative_error':(abs(truth[:3]-prediction)/np.maximum(abs(truth[:3]),floors)).tolist()})
        errors=[r['relative_error'] for r in rows if r['supported']]
        patch['validation']={'rows':rows,'supported':len(errors),'requested':8,'used_for_refinement':False,
            'max_relative_error':float(np.max(errors)) if errors else None,'physical_accuracy_validated':False}
    return patch


def sample_camera_patch(patch):
    x,y=pixels(patch['view'])
    city=interpolate_patch(patch,patch['city_response_per_m2_sr_s'],x,y)
    residual=interpolate_patch(patch,patch['unresolved_response_per_m2_sr_s'],x,y)
    return city,residual,patch_support(patch,x,y),pixel_solid_angles(patch['view'])
