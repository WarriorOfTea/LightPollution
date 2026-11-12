import importlib.util
import hashlib
from pathlib import Path
import struct

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('native_patch',ROOT/'integrations/kstars/apply_native.py')
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)


def source(tmp_path):
    root=tmp_path/'kstars';(root/'tools').mkdir(parents=True);(root/'skyobjects').mkdir()
    (root/'CMakeLists.txt').write_text('set(tools\n    tools/altvstime.cpp\n)\n')
    (root/'kstarsinit.cpp').write_text('#include "kstars.h"\nvoid init() {\n    setupGUI(StandardWindowOptions(Default), m_KStarsUIResource);\n}\n')
    (root/'skyobjects/starobject.cpp').write_text('void init(const StarData *stardata) {\n    B = V = 99.9;\n}\n')
    return root


def test_patch_is_idempotent_and_does_not_require_modern_Ekos_for_legacy_sources(tmp_path):
    root=source(tmp_path);updates=native.prepare(tmp_path)
    for path,content in updates.items():path.write_text(content)
    assert native.prepare(tmp_path)==updates
    assert 'bv_index / 100.0' in (root/'skyobjects/starobject.cpp').read_text()
    assert (root/'kstarsinit.cpp').read_text().count('new LightPollutionDock')==1


def test_unsupported_Ekos_layout_is_rejected_before_any_source_is_changed(tmp_path):
    root=source(tmp_path);ekos=root/'ekos/capture/exposurecalculator';ekos.mkdir(parents=True)
    (ekos/'optimalsubexposurecalculator.h').write_text('unsupported API')
    (ekos/'optimalsubexposurecalculator.cpp').write_text('unsupported implementation')
    before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match='Unsupported Ekos'):
        native.prepare(tmp_path)
    assert {p:p.read_bytes() for p in root.rglob('*') if p.is_file()}==before


def test_HD_index_offsets_handle_trixel_copies_and_ambiguous_components(tmp_path,monkeypatch):
    module_spec=importlib.util.spec_from_file_location('catalogue_install',ROOT/'integrations/kstars/repair_catalogue.py')
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    tool=tmp_path/'kstars/data/tools/fix-star-catalogs.py';tool.parent.mkdir(parents=True)
    # Stub only the audited upstream reader; exercise our index with raw32-byte records.
    tool.write_text('from pathlib import Path\nclass Catalog:\n def __init__(self,p):\n  self.data=bytearray(Path(p).read_bytes());self.index=[(0,64,(len(self.data)-64)//32)]\n')
    monkeypatch.setattr(module,'TOOL_SHA256',hashlib.sha256(tool.read_bytes()).hexdigest())
    def record(hd,ra=100,mag=950):return struct.pack('<iiiiiihh2sbb',ra,100,0,0,0,hd,mag,50,b'G2',0,0)
    catalogue=tmp_path/'deepstars.dat'
    catalogue.write_bytes(bytes(64)+record(1)+record(1)+record(2)+record(2,ra=101)+record(3))
    path=tmp_path/'Henry-Draper.idx';result=module.hd_index(tmp_path,catalogue,path)
    assert struct.unpack_from('<III',path.read_bytes())==(64,0,64+4*32)
    assert result['indexed_HD']==2 and result['ambiguous_HD_omitted']==1
    assert len(path.read_bytes())==360000*4
