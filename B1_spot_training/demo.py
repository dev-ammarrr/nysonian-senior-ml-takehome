"""B1 demo: preempt → resume with stable global batch."""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from trainer import prove_no_duplicate_steps, grad_accum_for_world_size, TrainConfig


def section(t):
    print(f"\n{'='*72}\n  {t}\n{'='*72}\n")


def main():
    section("B1: Spot / Preemptible Training Demo")

    print("Assumed production model:")
    print("  ~7B params, bf16 weights ~14GB")
    print("  Adam moments + grads + activations ⇒ does not fit on 1×16GB GPU")
    print("  Parallelism: FSDP/ZeRO-3 data-parallel sharding across spot GPUs")
    print("  Mixed precision: ~2× memory cut on activations/weights")
    print("  Grad accumulation: holds global batch fixed under elastic world_size")
    print()

    section("Elastic batch math")
    cfg = TrainConfig()
    for ws in [4, 2, 1]:
        accum = grad_accum_for_world_size(cfg, ws)
        effective = cfg.micro_batch * ws * accum
        print(f"  world_size={ws} → accum={accum} → global_batch={effective}")

    section("Preemption drill (save + resume, world_size 2→1)")
    ckpt_dir = Path(__file__).parent / "checkpoints"
    result = prove_no_duplicate_steps(ckpt_dir)
    print(result)
    assert result["effective_batch_stable"]
    assert result["preempted_at"] == 7
    assert result["completed_at"] == 20
    print("✓ No lost/duplicated steps; global batch stable across elastic shrink")

    section("Cloud fleet mechanism + cost (reasoned)")
    print("Mechanism: AWS EC2 Spot + Auto Scaling Group with Capacity Rebalancing")
    print("  + S3 checkpoints; or GKE/Karpenter Spot + GCS; or SkyPilot.")
    print("  Capacity Rebalancing gives ~2 min notice → drain + checkpoint.")
    print()
    print("Cost sketch (illustrative, 8×A10G 24GB, 24h run):")
    print("  On-demand A10G ~$1.00/hr → 8×24 = $192")
    print("  Spot ~60–70% off → ~$0.30–0.40/hr → 8×24 = $58–$77")
    print("  Checkpoint overhead ~2–5% time + S3 << $5")
    print("  Net: ~2.5–3× cheaper if preemption handled; waste dominates if not.")

    section("Demo Complete")


if __name__ == "__main__":
    main()
