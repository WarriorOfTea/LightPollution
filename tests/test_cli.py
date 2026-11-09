import json

import numpy as np
import pytest

from lightpollution.cli import make_parser,run_simulation


def test_demo_exports_coordinates_raw_numbers_and_synthetic_provenance(tmp_path):
    args=make_parser().parse_args(['demo','--output',str(tmp_path),'--azimuth-step','90',
                                  '--altitudes','5','45','90','--ray-segments','8','--path-quadrature','2'])
    summary=run_simulation(args)
    assert summary['provenance']['input_kind']=='synthetic'
    assert summary['provenance']['actual_nasa_pixels']==0
    with np.load(tmp_path/'sky-grid.npz',allow_pickle=False) as data:
        assert data['radiance'].shape==(3,4)
        np.testing.assert_allclose(data['radiance'][-1],data['radiance'][-1,0])
        assert not np.isclose(data['radiance'].max(),1)
    csv=np.loadtxt(tmp_path/'sky-grid.csv',delimiter=',',skiprows=1)
    assert csv.shape==(12,3)
    np.testing.assert_allclose(csv[:4,:2],[[0,5],[90,5],[180,5],[270,5]])
    assert (tmp_path/'sky.png').stat().st_size>0
    assert json.loads((tmp_path/'summary.json').read_text())['absolute_visible_photometry_calibrated'] is False


def test_simulation_without_real_data_fails_without_synthetic_fallback(tmp_path):
    args=make_parser().parse_args(['simulate','--region',str(tmp_path/'absent.npz'),'--output',str(tmp_path/'result')])
    with pytest.raises(FileNotFoundError): run_simulation(args)
    assert not (tmp_path/'result').exists()


def test_selecting_another_year_does_not_overwrite_the_2025_catalogue(tmp_path,monkeypatch):
    import sys
    from lightpollution import cli
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys,'argv',['lightpollution','catalogue','--year','2024'])
    monkeypatch.setattr(cli,'catalogue',lambda year:{'year':year,'files':[]})
    cli.main()
    assert json.loads((tmp_path/'artifacts/nasa-2024-catalogue.json').read_text())['year']==2024
    assert not (tmp_path/'artifacts/nasa-2025-catalogue.json').exists()
