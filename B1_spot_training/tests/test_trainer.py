import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trainer import (
    TrainConfig,
    TinyMLP,
    PreemptionFlag,
    train_loop,
    grad_accum_for_world_size,
    prove_no_duplicate_steps,
    save_checkpoint,
    load_checkpoint,
    CheckpointMeta,
)
import numpy as np


def test_grad_accum_keeps_global_batch():
    cfg = TrainConfig(micro_batch=8, target_global_batch=32)
    for ws in [1, 2, 4]:
        accum = grad_accum_for_world_size(cfg, ws)
        assert cfg.micro_batch * ws * accum == cfg.target_global_batch


def test_checkpoint_roundtrip(tmp_path=None):
    cfg = TrainConfig()
    rng = np.random.default_rng(0)
    model = TinyMLP(cfg, rng)
    model.t = 3
    path = Path(__file__).parent.parent / "checkpoints" / "unit_test.ckpt.npz"
    meta = CheckpointMeta(
        global_step=3,
        samples_seen=96,
        world_size=2,
        grad_accum_steps=2,
        rng_state=[],
        loss=1.23,
    )
    save_checkpoint(path, model, meta)
    model2 = TinyMLP(cfg, np.random.default_rng(1))
    loaded = load_checkpoint(path, model2)
    assert loaded.global_step == 3
    assert np.allclose(model.w1, model2.w1)
    assert model2.t == 3


def test_preempt_resume_no_duplicate():
    ckpt_dir = Path(__file__).parent.parent / "checkpoints"
    result = prove_no_duplicate_steps(ckpt_dir)
    assert result["preempted_at"] == 7
    assert result["completed_at"] == 20
    assert result["effective_batch_stable"] is True


if __name__ == "__main__":
    test_grad_accum_keeps_global_batch()
    print("✓ test_grad_accum_keeps_global_batch")
    test_checkpoint_roundtrip()
    print("✓ test_checkpoint_roundtrip")
    test_preempt_resume_no_duplicate()
    print("✓ test_preempt_resume_no_duplicate")
    print("\nAll B1 tests passed!")
