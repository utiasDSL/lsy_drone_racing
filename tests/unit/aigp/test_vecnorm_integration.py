import pytest

from scripts.train_aigp_curriculum import (
    _capture_vecnormalize_state,
    _parse_force_advance_mode,
    _resolve_force_advance,
    _restore_vecnormalize_state,
)


class _FakeVecNorm:
    def __init__(self):
        self.obs_rms = {"mean": [1.0, 2.0], "var": [3.0, 4.0]}
        self.ret_rms = {"mean": 5.0, "var": 6.0}


@pytest.mark.unit
def test_vecnormalize_state_capture_restore_roundtrip():
    src = _FakeVecNorm()
    state = _capture_vecnormalize_state(src)
    assert state is not None

    src.obs_rms["mean"][0] = 99.0
    src.ret_rms["mean"] = -1.0

    dst = _FakeVecNorm()
    dst.obs_rms["mean"][0] = -7.0
    _restore_vecnormalize_state(dst, state)

    assert dst.obs_rms["mean"][0] == 1.0
    assert dst.ret_rms["mean"] == 5.0


@pytest.mark.unit
def test_force_advance_resolution_modes():
    forced, blocked = _resolve_force_advance(
        stage_budget_reached=True,
        stage_idx=1,
        final_stage_idx=9,
        decision_advance=False,
        force_advance_mode="always",
    )
    assert forced
    assert not blocked

    forced, blocked = _resolve_force_advance(
        stage_budget_reached=True,
        stage_idx=1,
        final_stage_idx=9,
        decision_advance=False,
        force_advance_mode="if_passing",
    )
    assert not forced
    assert blocked

    forced, blocked = _resolve_force_advance(
        stage_budget_reached=True,
        stage_idx=1,
        final_stage_idx=9,
        decision_advance=True,
        force_advance_mode="if_passing",
    )
    assert not forced
    assert not blocked


@pytest.mark.unit
def test_parse_force_advance_mode_rejects_invalid():
    assert _parse_force_advance_mode("IF_Passing") == "if_passing"
    with pytest.raises(ValueError):
        _parse_force_advance_mode("sometimes")
