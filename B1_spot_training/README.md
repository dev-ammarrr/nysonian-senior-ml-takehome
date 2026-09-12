# B1: Cheap Multi-GPU Training on Spot / Preemptible Nodes

## Assumed production setup

| Item | Choice |
|------|--------|
| Model | ~7B decoder (Llama-class), ~14GB bf16 weights |
| Why not 1 GPU | Adam moments (~2× params fp32) + grads + activations blow past 16–24GB |
| Parallelism | **FSDP / ZeRO-3** (data parallel + sharded params/opt/grads) |
| Not chosen | Pure tensor parallel (overkill at 7B, harder elasticity); pipeline (bubble + complex) |
| Mixed precision | bf16/fp16 → ~2× less activation/weight memory, faster tensor cores |
| Grad accumulation | Holds **global batch** fixed when spot workers disappear |

## Demo (scaled down)

Tiny MLP + simulated preemption at step 7 → resume with `world_size` 2→1 → finish at step 20.

**Proved:**
- Checkpoint contains weights + Adam state + `global_step` + `samples_seen`
- Resume continues from the same step (no lost / duplicated optimizer steps)
- When workers shrink, `grad_accum` increases so `micro × world × accum` stays constant

## Decisions

### Decision 1: FSDP/ZeRO-3 over tensor/pipeline for this size

**Why:** 7B shards cleanly; elasticity is easier (drop a rank, re-shard) than fixing pipeline stages mid-run.  
**Rejected TP-primary:** Communication pattern + less natural for spot churn.  
**Rejected pipeline-primary:** Pipeline bubbles + stage remapping on node loss is painful.

### Decision 2: Checkpoint frequency vs size

At 7B + Adam, full ckpt ~40–80GB. On 10Gbps → tens of seconds.  
**Chose:** Every N minutes *or* every K steps, plus **immediate checkpoint on rebalance signal**.  
Demo uses every 5 steps + on preemption.  
**Rejected:** Every step — checkpoint IO would dominate cheap spot savings.

### Decision 3: Elastic world_size must not drift batch/LR

**Chose:** Recompute `grad_accum = target_global_batch / (micro_batch × world_size)`.  
**Why:** Silent batch-size change mid-run is a classic “looks like training but diverged” bug.  
**Also track `samples_seen`** so data order can skip already-consumed batches.

### Decision 4: Cloud mechanism

**Chose (AWS sketch):** Spot ASG + **Capacity Rebalancing** + S3 checkpoints (or SkyPilot).  
GCP: managed instance groups with preemptible + GCS.  
**Why:** Rebalancing notice (~2 min) is enough to finish in-flight step + flush ckpt.

### Decision 5: Cost comparison (reasoned)

Illustrative 8×A10G, 24h:
- On-demand ~$1/gpu-hr → **~$192**
- Spot ~60–70% discount → **~$58–77**
- Checkpoint/S3 overhead small (~$1–5) if preemption handled
- Without checkpointing, spot is *more expensive* (rewinding hours of progress)

## Run

```bash
cd B1_spot_training
../A1_tool_routing/venv/bin/python demo.py
../A1_tool_routing/venv/bin/python tests/test_trainer.py
```

## Limits of this demo

- NumPy MLP, not real NCCL/FSDP
- Single-process simulation of world_size (math is real; distributed runtime is not)
- RNG skip-by-count is a teaching stand-in for resumable data loaders (e.g. resumable WebDataset / Mosaic)
