"""Pre-baked size variants of :class:`RadiantConfig`."""

from __future__ import annotations

from radiant.config import RadiantConfig


def tiny_config(**overrides) -> RadiantConfig:
    """A few hundred K params; used by the unit-test suite."""
    cfg = RadiantConfig(
        vocab_size=128,
        d_model=64,
        n_query_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        max_seq_len=64,
        n_stem_blocks=1,
        n_refinement_blocks=1,
        n_exit_blocks=1,
        n_loops_train=2,
        min_loops=1,
        max_loops=4,
    )
    return cfg.replace(**overrides) if overrides else cfg


def small_config(**overrides) -> RadiantConfig:
    """A few M params; suitable for the synthetic-task examples."""
    cfg = RadiantConfig(
        vocab_size=512,
        d_model=192,
        n_query_heads=6,
        n_kv_heads=2,
        head_dim=32,
        d_ff=768,
        max_seq_len=256,
        n_stem_blocks=2,
        n_refinement_blocks=2,
        n_exit_blocks=2,
        n_loops_train=4,
        min_loops=1,
        max_loops=12,
    )
    return cfg.replace(**overrides) if overrides else cfg


def base_config(**overrides) -> RadiantConfig:
    """30-50M-param starting point for ChEMBL-scale runs."""
    cfg = RadiantConfig(
        vocab_size=2048,
        d_model=512,
        n_query_heads=8,
        n_kv_heads=4,
        head_dim=64,
        d_ff=2048,
        max_seq_len=512,
        n_stem_blocks=4,
        n_refinement_blocks=4,
        n_exit_blocks=4,
        n_loops_train=6,
        min_loops=2,
        max_loops=24,
    )
    return cfg.replace(**overrides) if overrides else cfg
