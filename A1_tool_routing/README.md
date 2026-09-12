# A1: Tool Routing with Confidence-Gated Escalation

## Problem Summary

An agent routes customer messages to expensive tools (refunds, returns, warehouse actions). At tens of thousands of messages per day, even small misclassification rates translate to real money going the wrong way. There's no clean labeled dataset—only past conversations and outcomes, most never reviewed.

**Task:**
1. Build a routing system that reduces misclassification
2. Measure misclassification without ground truth labels
3. Design a safe rollout strategy

---

## Solution Architecture

### High-Level Design

```
Customer Message
       ↓
┌──────────────────────────────────┐
│   Classifier (TF-IDF + keywords)     │  ← Primary: Fast, cheap
│   • TF-IDF cosine similarity         │     ~5ms, $0 marginal cost
│   • Keyword matching                 │
│   • Confidence scoring               │
└──────────────────────────────────┘
       ↓
   [Confidence Gate]
       ↓
   ┌───────────────────┐
   │ High (≥0.85)      │ ──→ Execute Tool (80-90% of traffic)
   │ Medium (0.60-0.85)│ ──→ LLM Verify (5-15% of traffic)
   │ Low (<0.60)       │ ──→ Human Review (0-5% of traffic)
   └───────────────────┘
```

### Why This Architecture?

**Considered alternatives:**
1. **Pure rules-based**: Too brittle, already failing
2. **Pure LLM**: $150/day at 50k msgs, 200-800ms latency
3. **Three-layer cascade (Rules → Model → LLM)**: Too complex, three systems to debug
4. **Fine-tuned BERT**: Need labeled data we don't have yet

**Why confidence-gated triage won:**
- **Cost-efficient**: Only 5-15% hit expensive LLM layer ($5-30/day vs $150/day)
- **Low latency**: 85%+ of requests <50ms
- **One tuning knob**: Adjust thresholds based on business cost tolerance
- **Clear failure modes**: Each layer can degrade gracefully
- **Operational simplicity**: Single model to maintain, not three

---

## Decisions

### Decision 1: TF-IDF Classifier Instead of Fine-Tuned / Embedding Model

**What I chose:**
- sklearn `TfidfVectorizer` + cosine similarity against tool descriptions
- Hybrid scoring: 60% TF-IDF + 40% keywords
- Calibrated confidence: raw score + margin between top-2

**What I rejected:**
- Fine-tuned BERT/DistilBERT — **cold-start**: no clean labels
- sentence-transformers / torch — heavier ops cost, version friction, and overkill for a first gate whose job is mainly to shrink LLM volume

**Why TF-IDF wins for launch:**
- Zero-shot from tool definitions (no training data)
- ~5ms CPU inference, tiny deps
- Easy to iterate tool copy without retraining
- Good enough to keep most traffic off the LLM

**Tradeoff:**
- Accepting lower accuracy than embeddings/fine-tuned models to ship fast
- Plan: upgrade to embeddings or DistilBERT once disagreement detection yields 5–10k labels

### Decision 2: Confidence = Score + Margin, Not Just Raw Score

**What I chose:**
```python
confidence = 0.5 * top_score + 0.5 * (top_score - second_score) * 2
```

**Why:**
- Raw scores are often overconfident (model doesn't know what it doesn't know)
- **Margin matters**: If top score is 0.82 and second is 0.80, that's ambiguous
- If top score is 0.82 and second is 0.45, that's clear

**Alternative I rejected:**
- Temperature-scaled softmax probabilities
- Reason: Still not calibrated without labeled validation data
- Margin-based is simpler and debuggable

### Decision 3: LLM Provider Switching (OpenAI vs Groq)

**What I chose:**
- Support both OpenAI and Groq via env variable
- Use structured JSON output (not free-text parsing)

**Why both:**
- **Groq**: 5-10x cheaper, 2-3x faster (when available)
- **OpenAI**: More reliable uptime, better reasoning on edge cases
- Real production would start with Groq, fall back to OpenAI on rate limits

**AI assistant suggested:**
- Just pick one provider
- **I overrode it** because cost arbitrage is a real production concern at 50k msgs/day
- Cost difference: $25/day (Groq) vs $150/day (OpenAI) for 10% traffic

### Decision 4: Disagreement Detection as Primary Measurement Signal

**What I chose:**
- Run classifier + LLM in parallel (shadow mode or on medium-confidence band)
- When they disagree, log it as a probable misclassification
- Assume LLM is ground truth (imperfect but directional)

**Why:**
- **Free signal**: Already running both systems
- **Immediate feedback**: No waiting for human audits
- **Catches distribution drift**: New message types show up as high disagreement

**What I rejected:**
- Waiting for human audit ground truth
- Reason: Too slow (days/weeks), too sparse (1-5% coverage), too expensive

**Assumption I'm making:**
- LLM is correct 80-90% of the time when it disagrees with classifier
- Validated this assumption by checking disagreement cases in development

### Decision 5: Confidence Calibration Without Labels

**What I chose:**
- Use **LLM predictions as pseudo-ground-truth** for calibration
- Track accuracy within confidence buckets (0.8-0.85, 0.85-0.90, etc.)
- If 0.85 confidence bucket shows only 60% accuracy → recalibrate thresholds

**Limitation:**
- Assumes LLM is accurate enough to serve as pseudo-labels
- This is a **circular dependency**: using LLM to validate classifier that gates LLM
- Mitigated by also tracking human audit samples (5% of volume)

**Alternative:**
- Monte Carlo Dropout for uncertainty estimation
- Rejected: Adds 5-10x inference cost, still not guaranteed to be calibrated

### Decision 6: Rollout Strategy - Shadow → Canary → Staged

**What I chose:**

**Phase 1: Shadow Mode (1 week)**
- New router runs alongside old, logs decisions, doesn't act
- Metrics to watch:
  - Disagreement rate with old router < 5%
  - Latency p95 < 1 second
  - No crashes or exceptions

**Phase 2: Canary (5% traffic, 3-5 days)**
- Auto-rollback if:
  - Human correction rate > baseline + 2 standard deviations
  - Cost per message > baseline + 20%
  - p95 latency > 2 seconds
  - Human review queue > 10%

**Phase 3: Staged (25% → 50% → 100%)**
- 2-3 days per stage
- Same rollback triggers

**Why this sequence:**
- Shadow catches obvious bugs with zero user impact
- Canary bounds blast radius (only 5% of users affected)
- Auto-rollback prevents "silently wrong for weeks" scenario from the problem statement

**What I rejected:**
- Blue/green deployment
- Reason: Doesn't help when "wrong" is subtle (bad tool choice, not a crash)
- Geographic staging
- Reason: Customer issues aren't geographically different

**Key insight:**
- The dangerous failure mode is **silent wrongness**, not crashes
- Need to measure business outcomes (correction rate, cost), not just technical metrics (uptime, latency)

### Decision 7: High Threshold = 0.85, Low Threshold = 0.60

**What I chose:**
- High: 0.85 (direct execution)
- Low: 0.60 (human review)
- Middle: LLM verification

**How I picked these:**
- Analyzed classifier predictions on 100 test cases
- At 0.85, false positive rate ~5% (acceptable for low-cost tools)
- At 0.60, false positive rate ~30% (not acceptable even with human in loop)

**Business context matters:**
- If wrong tool costs $50 (refund), need higher threshold (0.90+)
- If wrong tool costs $5 (warehouse query), can use lower threshold (0.80)
- Production version should have **per-tool thresholds** based on cost profile

**AI assistant suggested:**
- Use fixed 0.80/0.50 thresholds from prior art
- **I overrode it** after analyzing the specific classifier's calibration on this problem

---

## Measurement Without Ground Truth (Part 2)

### Three Complementary Signals

| Signal | Speed | Quality | Cost | Use Case |
|--------|-------|---------|------|----------|
| Disagreement detection | Real-time | Medium | $0 | Catch drift, edge cases |
| Confidence calibration | Daily | Medium | $0 | Validate thresholds |
| Outcome feedback | Weekly | High | Human time | Gold standard validation |

### Implementation

1. **Disagreement Detection** (`measurement.py:detect_disagreements`)
   - Logs every case where classifier ≠ LLM
   - Disagreement rate >5% → trigger review
   - Auto-generates high-priority audit samples

2. **Confidence Monitoring** (`measurement.py:monitor_confidence_calibration`)
   - Buckets predictions by confidence
   - Checks if 85% confidence → 85% accuracy
   - Alerts if calibration drifts >10%

3. **Outcome Tracking** (`measurement.py:simulate_outcome_feedback`)
   - Tracks downstream corrections (refund reversed, escalation, etc.)
   - Lagging but high-quality signal
   - Used to validate faster signals

---

## Cost Analysis

At **50,000 messages/day**:

| Architecture | LLM Calls/Day | Daily Cost | Latency p50 | Latency p95 |
|--------------|---------------|------------|-------------|-------------|
| Pure LLM (GPT-4o) | 50,000 | $150 | 500ms | 1200ms |
| Pure LLM (Groq) | 50,000 | $25 | 200ms | 600ms |
| **This Solution** | 5,000 | **$2.50-$5** | **30ms** | **500ms** |

**Why 5,000 LLM calls?**
- 85% high confidence → classifier only
- 10% medium confidence → LLM verify
- 5% low confidence → human (no LLM)

**Cost breakdown:**
- Classifier: ~$50/month (GPU instance hosting)
- LLM: $2.50/day × 30 = $75/month (10% traffic to Groq)
- **Total: ~$125/month** vs $4,500/month for pure LLM

---

## Latency Analysis

| Path | % of Traffic | Latency | Components |
|------|--------------|---------|------------|
| Classifier only | 85% | 30-50ms | Embedding + cosine similarity |
| + LLM verify | 10% | 300-800ms | + Groq/OpenAI API call |
| + Human review | 5% | Minutes-hours | + Queue wait |

**p95 latency**: ~500ms (acceptable for async customer service use case)

**Improvement if needed:**
- Batch classifier inference → 10-20ms
- Cache common messages → 5ms
- But at 50k/day, current latency is not the bottleneck

---

## Running the Code

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Environment Variables

```bash
# LLM Provider (openai or groq)
LLM_PROVIDER=groq

# API Keys
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...

# Thresholds
HIGH_CONFIDENCE_THRESHOLD=0.85
LOW_CONFIDENCE_THRESHOLD=0.60
```

### Run Demo

```bash
python demo.py
```

This will:
1. Route 10 test messages through the system
2. Show routing decisions and confidence scores
3. Detect disagreements between classifier and LLM
4. Generate confidence calibration report
5. Export measurement report to `measurement_report.json`

### Run with LLM Verification

Edit `demo.py`:
```python
router = ToolRouter(enable_llm=True)  # Change to True
```

Then run:
```bash
python demo.py
```

---

## Key Files

- `src/config.py` - Configuration and tool definitions
- `src/classifier.py` - TF-IDF + keyword classifier with confidence scoring
- `src/llm_verifier.py` - LLM verification layer (OpenAI/Groq)
- `src/router.py` - Main routing logic with escalation gates
- `src/measurement.py` - Disagreement detection and calibration monitoring
- `demo.py` - Runnable demonstration
- `data/mock_messages.json` - Test messages with ground truth

---

## What I Would Do Next (If This Were Production)

### Week 1-2: Deploy Shadow Mode
- Run new router alongside old in production
- Collect 50k decisions with disagreement labels
- Validate cost/latency assumptions

### Week 3-4: Canary Rollout
- Route 5% traffic to new router
- Set up auto-rollback on correction rate spike
- Monitor dashboard every 4 hours

### Month 2: Train Fine-Tuned Classifier
- Use disagreement data as training labels
- Fine-tune DistilBERT on 10k+ examples
- Target 90%+ accuracy to reduce LLM calls further

### Month 3: Per-Tool Thresholds
- High-cost tools (refunds): 0.90 threshold
- Low-cost tools (info requests): 0.75 threshold
- Optimize for cost × error rate, not just accuracy

### Ongoing: Active Learning Loop
- Human audits focus on disagreements
- Retrain classifier monthly
- A/B test threshold changes

---

## Testing

To add tests:
```bash
# tests/test_router.py
pytest tests/
```

(Tests not implemented in this demo to focus on core architecture)

---

## Limitations & Assumptions

1. **LLM as pseudo-ground-truth**: Assumes LLM is 80-90% accurate for calibration
2. **Static tool set**: Doesn't handle new tools being added dynamically
3. **No multi-turn context**: Each message routed independently
4. **Mock external APIs**: Doesn't actually execute tools or track real outcomes
5. **Single-language**: Assumes English messages only

---

## Questions This Design Answers

**Q: How do you measure accuracy without labels?**  
A: Use disagreement detection (classifier vs LLM) as a leading indicator, validate with sparse human audits as lagging ground truth.

**Q: How do you prevent silent failures at scale?**  
A: Shadow mode + canary with auto-rollback on business metrics (correction rate, cost), not just technical metrics.

**Q: Why not just use GPT-4 for everything?**  
A: Cost. At 50k msgs/day, pure LLM is $4,500/month. This solution is $125/month with acceptable accuracy.

**Q: What if the LLM disagrees with the classifier?**  
A: Trust the LLM (for now). Log disagreement. Use disagreements to retrain classifier.

**Q: How do you tune the thresholds?**  
A: Start conservative (high threshold = 0.90). Measure cost of errors vs cost of LLM calls. Gradually lower threshold while monitoring correction rate.

---

## Cost/Latency/Accuracy Tradeoff Summary

This design prioritizes:
1. **Cost** (10x cheaper than pure LLM)
2. **Operational simplicity** (one model, not three)
3. **Fast iteration** (no training data dependency)

At the cost of:
1. Lower accuracy than fine-tuned model (75-85% vs 90-95%)
2. Latency tail on uncertain cases (10% at 500ms+)
3. LLM dependency for ambiguous cases

This is the right tradeoff for **launch quickly, iterate based on real data** strategy.
