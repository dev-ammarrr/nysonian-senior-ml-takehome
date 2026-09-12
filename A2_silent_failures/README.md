# A2: Catching Silent Tool Failures

## Problem

A tool integration can return HTTP/status success, the agent workflow logs the step as completed, and **no error fires** — while the real-world effect never happens (or happens with wrong params). At scale this can run for months until an audit.

## Solution (what we built)

1. **Verified-effect monitoring** — after every reported success, independently check the side-effect store (warehouse DB stand-in) for the expected record + params.
2. **Synthetic canaries** — continuous known transactions through the *real* tool path.
3. **Gap monitoring** — alert when `reported_success_rate - verified_success_rate` exceeds a threshold with enough samples.
4. **Deliberately broken mock tool** that returns success while writing nothing — detection catches it.

```
Tool call → reported success
              ↓
     Workflow logs "completed"   ← blind
              ↓
     VerifiedEffectMonitor reads SideEffectStore independently
              ↓
     missing / mismatch → silent_failure alert
```

## Decisions

### Decision 1: Verified-effect as primary signal (not log sampling)

**Chose:** Independent read of system-of-record after every success (or sampled at high %).  
**Rejected:** Human transcript sampling — too slow, misses months of damage.  
**Rejected:** Only watching error rates / HTTP 5xx — this bug produces *no* errors.  
**Why:** The assignment’s failure mode is “success that lied.” You must verify the *effect*, not the *response*.

### Decision 2: Canaries through the real tool, not a parallel fake

**Chose:** Canary orders with known `order_id`/`sku` through the production tool client.  
**Rejected:** Health-check ping that only hits a `/health` endpoint — that can be green while create_return_label is broken.  
**Cost:** Tiny write volume (e.g. 1 canary / minute / region).  
**Latency:** Detection in seconds–minutes, not weeks.

### Decision 3: Gap metric + paging threshold (false-alarm tradeoff)

**Chose:** Page when `gap >= 2%` AND `n >= 20` in the sliding window (configurable).  
**Also:** Immediate alert on any canary silent failure (canaries should be ~100% verified).

**If tuned too sensitive:** replication lag / eventual consistency causes noise → raise threshold or add short verify-retry (e.g. re-check after 2s).  
**If tuned too loose:** gap stays under threshold while canaries fail → you’d know because canary dashboard stays red while pages stay quiet.  
**How we’d know we tuned wrong:** compare canary failure rate vs page rate weekly; they should move together.

### Decision 4: Also verify *params*, not just existence

**Chose:** Check `order_id`, `sku`, `action_type` match expected.  
**Why:** A write with wrong SKU is as bad as no write — and still “successful” to the caller.

### Decision 5: No LLM for detection

**Chose:** Deterministic store checks.  
**Rejected:** LLM reading transcripts to “notice something’s off.”  
**Why:** Cost, latency, and non-determinism. Silent side-effect bugs are a *systems* problem.

## Run

```bash
cd A2_silent_failures
../A1_tool_routing/venv/bin/python demo.py
../A1_tool_routing/venv/bin/python tests/test_detection.py
```

## What we’d do next in production

- Outbox / change-data-capture as the verify source (not a second write path).
- Per-tool SLOs: max silent-failure rate, max time-to-detect.
- Auto-disable tool + failover when canary streak fails.
- Correlate alerts with deploy versions (catch regressions from a bad release).
