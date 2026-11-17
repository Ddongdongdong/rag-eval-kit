# rag-eval-kit — Demo Report (v0.2)

Generated offline (TF-IDF retrieval, extractive answers, no LLM API). Strategy: `heading`, top_k=5, mode=extractive, docs=6, chunks=22, total time=10 ms.

## easy set — 12 questions

| Metric | Value | What it means |
|---|---|---|
| Recall@5 | **1.000** | Share of gold sources found in the top-5 results |
| Citation accuracy | **1.000** | Share of gold docs present in the citations |
| Answer coverage | **0.363** | Token-overlap F1 vs the gold answer |
| Hallucination proxy | **0.009** | Share of answer tokens absent from retrieved context (lower is better) |

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

## hard (paraphrased + cross-doc) set — 12 questions

| Metric | Value | What it means |
|---|---|---|
| Recall@5 | **1.000** | Share of gold sources found in the top-5 results |
| Citation accuracy | **1.000** | Share of gold docs present in the citations |
| Answer coverage | **0.410** | Token-overlap F1 vs the gold answer |
| Hallucination proxy | **0.007** | Share of answer tokens absent from retrieved context (lower is better) |

| # | Question | Recall | Citation | Coverage | Halluc. |
|---|---|---|---|---|---|
| 1 | How long after delivery can a customer send a product back,  | 1.00 | 1.00 | 0.27 | 0.01 |
| 2 | If a customer pays for the whole year in one go, how much ch | 1.00 | 1.00 | 0.41 | 0.02 |
| 3 | Once we've made first contact, how often do we check in, and | 1.00 | 1.00 | 0.40 | 0.00 |
| 4 | What shows that onboarding is complete for a new customer? | 1.00 | 1.00 | 0.46 | 0.00 |
| 5 | If a parcel goes missing, how long before support files a cl | 1.00 | 1.00 | 0.38 | 0.00 |
| 6 | How soon after the demo should the team send the written pro | 1.00 | 1.00 | 0.53 | 0.01 |
| 7 | A parcel arrived damaged — how soon must the customer report | 1.00 | 1.00 | 0.38 | 0.02 |
| 8 | A customer has not replied two days after the demo — what sh | 1.00 | 1.00 | 0.42 | 0.01 |
| 9 | The team wants to offer a 15 percent discount — who must app | 1.00 | 1.00 | 0.44 | 0.01 |
| 10 | The contract was signed yesterday — by when must the account | 1.00 | 1.00 | 0.37 | 0.00 |
| 11 | A customer on annual billing returns a product — how long is | 1.00 | 1.00 | 0.30 | 0.00 |
| 12 | A new customer has not logged in by day 7 — what should the  | 1.00 | 1.00 | 0.56 | 0.00 |

## Easy vs hard — reading the gap honestly

Recall@5 is 1.000 on the easy set and 1.000 on the hard set. On this small, cleanly separated 6-document corpus the hard set does not break Recall@5 — the retriever still finds every gold document inside the top-5. Where the hard set bites is at smaller k: see `reports/ablation_report.md` for the Recall@1/3 curves, where paraphrased and cross-document questions drop Recall@1 well below the easy set. That is the honest signal here — top-5 on 6 documents is too forgiving to discriminate, and a bigger, messier corpus is the v0.3 job, not a new metric.

## Worked example

**Q:** How long does a refund take after the warehouse receives the return?

**Answer (extractive mode, top-3 chunks):** Refunds go back to the original payment method. Once the warehouse receives the returned item, the refund is processed within 5 working days. The bank may need 3 to 7 extra days to show the refund. Send a written proposal within 2 working days after the demo. The proposal lists the plan, the price, and the start date. If the customer does not reply in 48 hours, send one follow-up message and one phone call. Do not discount without manager approval. The customer contacts support with the order nu

**Retrieved chunks:**

- `doc-returns#c2` score=0.3915 — Refunds go back to the original payment method. Once the warehouse receives the returned item, the refund is processed w
- `doc-sales-sop#c3` score=0.2415 — Send a written proposal within 2 working days after the demo. The proposal lists the plan, the price, and the start date
- `doc-returns#c1` score=0.1982 — The customer contacts support with the order number. Support checks the order, confirms the item is inside the 30-day wi

## How to reproduce

```bash
pip install -r requirements.txt
python scripts/run_demo.py                  # both QA sets, extractive
python scripts/run_demo.py --qa hard        # hard set only
python scripts/ablation.py                # fixed vs heading
# LLM answers (needs env vars, see rag/llm.py):
RAG_EVAL_LLM=1 RAG_EVAL_LLM_BASE_URL=<url> \
  RAG_EVAL_LLM_API_KEY=<key> RAG_EVAL_LLM_MODEL=<model> \
  python scripts/run_demo.py --mode llm
uvicorn app.main:app --reload
```

Then `POST /evaluate` returns the same numbers via the API.
