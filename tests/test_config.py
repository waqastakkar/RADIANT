from pathlib import Path

import pytest

from radiant import RadiantConfig, tiny_config


def test_defaults_validate():
    cfg = RadiantConfig()
    assert cfg.kv_groups == cfg.n_query_heads // cfg.n_kv_heads


def test_invalid_loop_counts_raise():
    with pytest.raises(ValueError):
        RadiantConfig(min_loops=4, n_loops_train=2)
    with pytest.raises(ValueError):
        RadiantConfig(n_loops_train=10, max_loops=4)


def test_invalid_head_div_raises():
    with pytest.raises(ValueError):
        RadiantConfig(n_query_heads=4, n_kv_heads=3)


def test_odd_head_dim_rejected():
    with pytest.raises(ValueError):
        RadiantConfig(head_dim=15)


def test_state_gate_bounds():
    with pytest.raises(ValueError):
        RadiantConfig(state_gate_init_scale=0.0)
    with pytest.raises(ValueError):
        RadiantConfig(state_gate_init_scale=1.0)


def test_moe_in_loop_requires_moe():
    with pytest.raises(ValueError):
        RadiantConfig(use_moe=False, moe_in_loop_experimental=True)


def test_dict_round_trip():
    cfg = tiny_config()
    d = cfg.to_dict()
    cfg2 = RadiantConfig.from_dict(d)
    assert cfg == cfg2


def test_json_round_trip(tmp_path: Path):
    cfg = tiny_config()
    path = tmp_path / "cfg.json"
    cfg.to_json(path)
    cfg2 = RadiantConfig.from_json(path)
    assert cfg == cfg2


def test_unknown_field_rejected():
    with pytest.raises(ValueError):
        RadiantConfig.from_dict({"d_model": 64, "not_a_field": 99})


def test_replace_returns_new():
    cfg = tiny_config()
    cfg2 = cfg.replace(d_model=128)
    assert cfg.d_model == 64
    assert cfg2.d_model == 128


def test_iteration_adapter_bottleneck():
    cfg = tiny_config(iteration_adapter_bottleneck_ratio=0.5)
    assert cfg.iteration_adapter_bottleneck >= 8
