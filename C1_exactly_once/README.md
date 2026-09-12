# C1: Exactly-Once Refund Endpoint + Slow Dependency Resilience

## Framing

Agent calls **issue refund** with an `Idempotency-Key`.  
A slow verification API (2–10s, timeouts/rate limits) must approve before money moves.

Two production failure modes already seen:
1. Check-then-insert race → duplicate refunds
2. Slow upstream → cascading timeouts on unrelated requests

## Architecture

```
Request + Idempotency-Key
        ↓
 SQLite UNIQUE claim (BEGIN-equivalent lock)
   ├─ first → status=in_progress → call verifier (timeout/retry/breaker)
   ├─ twin  → 409 in_progress
   └─ replay of success → 200 duplicate + cached body
        ↓
 Circuit breaker around verifier
   ├─ open → 503 circuit_open, status=failed_retryable (NOT terminal)
   └─ closed/half-open → verify → issue refund once → status=succeeded
```

Async service (no thread blocked on upstream). Health path stays responsive.

## Decisions

### Decision 1: Atomic claim via UNIQUE row, not check-then-act

**Chose:** `INSERT` idempotency key as `in_progress` (SQLite PRIMARY KEY). Conflict ⇒ inspect status.  
**Rejected:** `SELECT` then `INSERT` — race window is exactly the bug in the prompt.  
**Lease:** `in_progress` rows expire after `lease_sec` so a crashed worker cannot wedgie the key forever.

### Decision 2: Hold in_progress for the whole upstream call

**Chose:** Yes — guard stays held while verification runs.  
**Concurrent twin sees:** `409 in_progress` (or later `200 duplicate` if first finished).  
**Why not release early:** releasing before money moves re-opens the double-refund race.  
**Why not return 200 success to the twin early:** that would lie about completion.

### Decision 3: Circuit-open must NOT permanently consume the key

**Chose:** `failed_retryable` status; later attempt after recovery `UPDATE`s back to `in_progress`.  
**Rejected:** marking circuit-open as success/duplicate — that would block a legitimate refund forever once the dependency heals.  
This is the coexistence bug the prompt asks about.

### Decision 4: Async + circuit breaker + bounded retries

**Chose:** `asyncio.wait_for` timeout, exponential backoff retries, breaker (closed/open/half-open).  
**Rejected:** sync blocking workers — slow verifier exhausts the pool → healthy endpoints die.  
**Proved in tests:** `/health` still returns 200 while a refund waits on a 500ms verifier.

### Decision 5: Caller-visible outcomes

| Outcome | HTTP | Body | Why |
|---------|------|------|-----|
| Genuine success | 200 | `outcome=success`, `refund_id` | Money moved once |
| Duplicate | 200 | `outcome=duplicate`, cached refund | Safe client retry after timeout |
| In progress | 409 | `outcome=in_progress` | Twin arrived mid-flight; wait/retry |
| Circuit open | 503 | `outcome=circuit_open` | Back off; key reclaimable later |
| Upstream error | 502 | `outcome=upstream_error` | Retryable failure recorded |

## Run

```bash
cd C1_exactly_once
../A1_tool_routing/venv/bin/python demo.py
../A1_tool_routing/venv/bin/python tests/test_refunds.py
```

## Production next steps

- Redis `SET key NX EX` or DynamoDB conditional writes if multi-node
- Outbox for refund ledger write + async settlement
- Separate thread/process pools so verifier latency never shares with health/admin
- Breaker metrics → auto rollback of bad verifier deploys
