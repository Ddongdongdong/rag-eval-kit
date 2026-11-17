# rag-eval-kit — Demo Report

Generated offline (TF-IDF retrieval, extractive answers, no LLM API). Strategy: `heading`, top_k=5, docs=6, chunks=22, questions=12, total time=6 ms.

## Summary

| Metric | Value | What it means |
|---|---|---|
| Recall@5 | **1.000** | Share of gold sources found in the top-5 results |
| Citation accuracy | **1.000** | Share of gold docs present in the citations |
| Answer coverage | **0.363** | Token-overlap F1 vs the gold answer |
| Hallucination proxy | **0.009** | Share of answer tokens absent from retrieved context (lower is better) |

## Per-question results

| # | Question | Recall | Citation | Coverage | Halluc. |
|---|---|---|---|---|---|
| 1 | How quickly should we reply to a new sales lead? | 1.00 | 1.00 | 0.34 | 0.00 |
| 2 | What are the three discovery questions in the sales process? | 1.00 | 1.00 | 0.40 | 0.01 |
| 3 | How long is the return window and what condition must the pr | 1.00 | 1.00 | 0.38 | 0.00 |
| 4 | How long does a refund take after the warehouse receives the | 1.00 | 1.00 | 0.41 | 0.01 |
| 5 | What is the customer follow-up cadence after first contact? | 1.00 | 1.00 | 0.39 | 0.01 |
| 6 | How long should a follow-up message be? | 1.00 | 1.00 | 0.17 | 0.02 |
| 7 | What discount can a sales lead approve without a manager? | 1.00 | 1.00 | 0.33 | 0.00 |
| 8 | How much discount does annual billing give compared with mon | 1.00 | 1.00 | 0.43 | 0.01 |
| 9 | When does onboarding start and when should the account be cr | 1.00 | 1.00 | 0.32 | 0.01 |
| 10 | When is onboarding considered complete? | 1.00 | 1.00 | 0.47 | 0.00 |
| 11 | How long does standard shipping take? | 1.00 | 1.00 | 0.30 | 0.02 |
| 12 | What should a customer do if a parcel arrives damaged? | 1.00 | 1.00 | 0.39 | 0.00 |

## Worked example

**Q:** How long does a refund take after the warehouse receives the return?

**Extractive answer (top-3 chunks):** Refunds go back to the original payment method. Once the warehouse receives the returned item, the refund is processed within 5 working days. The bank may need 3 to 7 extra days to show the refund. Send a written proposal within 2 working days after the demo. The proposal lists the plan, the price, and the start date. If the customer does not reply in 48 hours, send one follow-up message and one phone call. Do not discount without manager approval. The customer contacts support with the order nu

**Retrieved chunks:**

- `doc-returns#c2` score=0.3915 — Refunds go back to the original payment method. Once the warehouse receives the returned item, the refund is processed w
- `doc-sales-sop#c3` score=0.2415 — Send a written proposal within 2 working days after the demo. The proposal lists the plan, the price, and the start date
- `doc-returns#c1` score=0.1982 — The customer contacts support with the order number. Support checks the order, confirms the item is inside the 30-day wi

## How to reproduce

```bash
pip install -r requirements.txt
python scripts/run_demo.py
uvicorn app.main:app --reload
```

Then `POST /evaluate` returns the same numbers via the API.
