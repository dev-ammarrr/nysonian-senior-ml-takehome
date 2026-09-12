# Senior ML Engineer Take-Home (Nysonian)

Python solutions for four production-style failure modes: agent routing cost, silent tool failures, spot-GPU training, and exactly-once APIs under a slow dependency.

**Judgment over polish.** Full tradeoff write-ups live in each problem’s `README.md` (that’s what the brief asks to grade). This root file is the map and runbook only.

## Repository layout

| # | Folder | Problem | Approach (one line) |
|---|--------|---------|---------------------|
| A1 | [`A1_tool_routing/`](./A1_tool_routing/) | Wrong tool → real money | Confidence-gated local classifier; LLM only on mid-band; disagreement metrics; shadow→canary rollout |
| A2 | [`A2_silent_failures/`](./A2_silent_failures/) | Success status, no real effect | Verified-effect checks + canaries + reported-vs-verified gap; broken mock tool caught |
| B1 | [`B1_spot_training/`](./B1_spot_training/) | Big model on vanishing spot GPUs | Assumed 7B + FSDP; demo MLP with preempt/resume and elastic global batch |
| C1 | [`C1_exactly_once/`](./C1_exactly_once/) | Exactly-once + slow upstream | Atomic idempotency + async timeout/retry + circuit breaker that don’t fight each other |

Assignment PDF/DOCX (reference): [`Docs/`](./Docs/)

## Cross-cutting choices

- **Language:** Python 3.11 everywhere.
- **Agent frameworks:** none (plain modules) — less magic, easier to defend cost/latency paths for A1/A2.
- **LLM:** OpenAI or Groq via `LLM_PROVIDER` in `.env` (A1). Other problems use deterministic mocks where an LLM isn’t the point.
- **Deps:** one venv under `A1_tool_routing/` (A2/B1/C1 can reuse it).
- **Secrets:** never committed — copy `.env.example` → `.env`.

## Setup

```bash
cd A1_tool_routing
./setup.sh
cp .env.example .env   # if needed
# set OPENAI_API_KEY and/or GROQ_API_KEY, LLM_PROVIDER=openai|groq
```

## Run demos & tests

```bash
PY=./A1_tool_routing/venv/bin/python

# A1
$PY A1_tool_routing/demo.py
$PY A1_tool_routing/tests/test_router.py
$PY A1_tool_routing/test_llm_smoke.py          # needs API key

# A2
$PY A2_silent_failures/demo.py
$PY A2_silent_failures/tests/test_detection.py

# B1
$PY B1_spot_training/demo.py
$PY B1_spot_training/tests/test_trainer.py

# C1
$PY C1_exactly_once/demo.py
$PY C1_exactly_once/tests/test_refunds.py
```

## Where to read decisions

| Problem | Decisions / write-up |
|---------|----------------------|
| A1 | [A1_tool_routing/README.md](./A1_tool_routing/README.md) |
| A2 | [A2_silent_failures/README.md](./A2_silent_failures/README.md) |
| B1 | [B1_spot_training/README.md](./B1_spot_training/README.md) |
| C1 | [C1_exactly_once/README.md](./C1_exactly_once/README.md) |

## Honest scope limits

- **B1:** scaled-down NumPy MLP; `world_size` is simulated (batch/elasticity math is real, not NCCL/FSDP).
- **C1:** async service API with concurrency tests (not a full HTTP gateway).
- **A1:** TF-IDF gate is a cold-start stand-in; production would graduate to embeddings/fine-tune once labels exist.

Each README lists “what I’d do next” if this were production.
