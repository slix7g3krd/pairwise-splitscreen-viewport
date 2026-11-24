import pytest
import graphics.splitscreen_viewport as mod

CASES = [
    ("shadow_atlas_partition", (1,)),
    ("reset_viewport", (1,)),
    ("isolate_target", (1,)),
    ("ui_viewport_map", (1,)),
    ("per_player_scale", (1,)),
    ("determinism", (1,)),
    ("camera_order", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("scissor_rect", (1,)),
    ("metrics", (1,)),
    ("split_layout", (1,)),
    ("render_camera", (1,)),
    ("compose", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
