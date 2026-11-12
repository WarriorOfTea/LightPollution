import numpy as np
import pytest

from lightpollution.models.kstars_horizon import validate_horizon,horizon_visible,_great_circle_level


def horizon(*regions):
    return validate_horizon({'kind':'KStars_piecewise_linear_v1','regions':list(regions)})


def region(points,ceiling=False):
    return {'name':'obstacle','ceiling':ceiling,'points':points}


def test_short_arc_crosses_north_without_closing_open_contour():
    h=horizon(region([[350,40],[10,60]]))
    assert horizon_visible(h,[0,0,180,350,10],[49,51,5,39,59]).tolist()==[False,True,True,False,False]
    assert bool(horizon_visible(h,0,50))  # Native strict above/below comparisons.
    assert bool(horizon_visible(h,20,1))  # No implied closing wall.


def test_nearest_floor_ceiling_rule_preserves_a_window_in_a_wall():
    h=horizon(region([[160,40],[200,40]]),region([[160,60],[200,60]],True),
              region([[160,70],[200,70]]),region([[160,80],[200,80]],True))
    # Lower open window 40..60, upper open window 70..80.
    assert horizon_visible(h,180,[30,45,65,75,85]).tolist()==[False,True,False,True,False]


def test_degenerate_and_half_circle_segments_match_native_short_arc_definition():
    h=horizon(region([[0,40],[180,40]]))
    assert not horizon_visible(h,np.arange(360),30).any()
    h=horizon(region([[20,10],[20,50]]))
    assert horizon_visible(h,[20,20,21],[49,51,1]).tolist()==[False,True,True]


@pytest.mark.parametrize('mutate',[
    lambda h:h['regions'][0].update(ceiling=1),
    lambda h:h['regions'][0].update(points=[[0,40]]),
    lambda h:h['regions'][0]['points'][0].__setitem__(1,float('nan')),
    lambda h:h['regions'][0]['points'][0].__setitem__(0,361),
    lambda h:h['regions'][0].update(extra=True),
])
def test_invalid_horizon_rejected(mutate):
    h=horizon(region([[0,40],[20,50]]));mutate(h)
    with pytest.raises(ValueError):validate_horizon(h)


@pytest.mark.parametrize('a0,h0,a1,h1,az,expected',[
    (290,89,0,89,300,89.0961),
    (290,89,359,89,300,89.0943),
    (0,89,290,89,300,89.0961),
    (0,79,290,79,300,80.0355),
])
def test_great_circle_matches_independent_native_navigation_examples(a0,h0,a1,h1,az,expected):
    # Golden examples from the pinned KStars TestArtificialHorizon::greatCircleTest.
    assert abs(float(_great_circle_level(a0,h0,a1,h1,az))-expected)<.001


def test_spherical_horizon_bows_above_a_linear_equal_height_wall():
    h=horizon(region([[160,45.2],[200,45.2]]))
    assert bool(horizon_visible(h,180,46))
    h['kind']='KStars_great_circle_v1';validate_horizon(h)
    assert not bool(horizon_visible(h,180,46))
    assert bool(horizon_visible(h,180,48))
