from pathlib import Path

import pytest

from lsy_drone_racing import _compilation_cache_enabled, _maybe_enable_jax_compilation_cache


@pytest.mark.unit
@pytest.mark.parametrize("raw", ["0", "false", "False", "no", "off", " OFF "])
def test_compilation_cache_disabled_values(raw: str):
    assert not _compilation_cache_enabled(raw)


@pytest.mark.unit
def test_maybe_enable_cache_skips_callback_when_disabled():
    calls: list[Path] = []
    enabled, cache_path = _maybe_enable_jax_compilation_cache(
        cache_toggle="0",
        cache_dir="/tmp/should_not_be_used",
        cache_enabler=lambda path: calls.append(path),
        machine="x86_64",
    )
    assert not enabled
    assert cache_path is None
    assert calls == []


@pytest.mark.unit
def test_maybe_enable_cache_uses_default_machine_dir_when_enabled():
    calls: list[Path] = []
    enabled, cache_path = _maybe_enable_jax_compilation_cache(
        cache_toggle=None,
        cache_dir=None,
        cache_enabler=lambda path: calls.append(path),
        machine="x86_64",
    )
    assert enabled
    assert cache_path == Path("/tmp/jax_cache_x86_64")
    assert calls == [Path("/tmp/jax_cache_x86_64")]
