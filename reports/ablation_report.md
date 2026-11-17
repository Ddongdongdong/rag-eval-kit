# rag-eval-kit — Chunking Ablation Report (v0.2)

Fixed-length vs heading/paragraph chunking, same retriever (TF-IDF + overlap rerank), same QA sets, extractive answers, no LLM API. Generated offline by `python scripts/ablation.py`.

Corpus: 6 documents. Chunks — fixed: 12, heading: 22. Each QA set has 12 questions.

## Recall@k curves

| k | easy / fixed | easy / heading | hard / fixed | hard / heading |
|---|---|---|---|---|
| 1 | 1.000 | 0.917 | 0.667 | 0.667 |
| 3 | 1.000 | 0.917 | 0.917 | 1.000 |
| 5 | 1.000 | 1.000 | 1.000 | 1.000 |
| 10 | 1.000 | 1.000 | 1.000 | 1.000 |

## Full metrics at k=5

| QA set | Strategy | Recall@5 | Citation acc. | Coverage | Halluc. proxy |
|---|---|---|---|---|---|
| easy | fixed | 1.000 | 1.000 | 0.357 | 0.011 |
| easy | heading | 1.000 | 1.000 | 0.363 | 0.009 |
| hard | fixed | 1.000 | 1.000 | 0.421 | 0.009 |
| hard | heading | 1.000 | 1.000 | 0.410 | 0.007 |

## Where Recall@1 misses

Questions whose gold document was not the top-1 hit. These are the cases where paraphrasing or cross-document reasoning actually costs the retriever.

### easy / fixed — 0 miss(es) at k=1

- (none — every gold document was the top-1 hit)

### hard / fixed — 7 miss(es) at k=1

- Once we've made first contact, how often do we check in, and when do we stop?
- A parcel arrived damaged — how soon must the customer report it, and what do they receive?
- A customer has not replied two days after the demo — what should the team do next, and which follow-up day is that?
- The team wants to offer a 15 percent discount — who must approve it, and what does the sales process say about discounting without approval?
- The contract was signed yesterday — by when must the account exist, and how soon after first contact should a demo be booked?
- A customer on annual billing returns a product — how long is the return window, and does the billing plan change it?
- A new customer has not logged in by day 7 — what should the team send, and how does the day 1, 3, 7, 14 rhythm fit in?

### easy / heading — 1 miss(es) at k=1

- How long should a follow-up message be?

### hard / heading — 7 miss(es) at k=1

- If a customer pays for the whole year in one go, how much cheaper is it than paying monthly?
- A parcel arrived damaged — how soon must the customer report it, and what do they receive?
- A customer has not replied two days after the demo — what should the team do next, and which follow-up day is that?
- The team wants to offer a 15 percent discount — who must approve it, and what does the sales process say about discounting without approval?
- The contract was signed yesterday — by when must the account exist, and how soon after first contact should a demo be booked?
- A customer on annual billing returns a product — how long is the return window, and does the billing plan change it?
- A new customer has not logged in by day 7 — what should the team send, and how does the day 1, 3, 7, 14 rhythm fit in?

## Conclusion

- **easy set:** heading − fixed is -0.083 at Recall@1 and +0.000 at Recall@5 — at k=1 the fixed strategy is ahead, at k=5 the two are tied.
- **hard set:** heading and fixed tie at Recall@1 and differ by +0.000 at Recall@5.

Reading this honestly: on this 6-document, 12-question corpus the chunking strategy barely matters once k reaches 3 — both strategies find every gold document by k=5 on both sets. The hard set only bites at k=1, where paraphrased and cross-document questions drop Recall@1 from 0.917–1.000 (easy) to 0.667–0.667 (hard). That is the ablation working as intended — it shows *where* the difference lives — and it also shows the corpus is too small for k>=3 to discriminate. The next step (v0.3) is a bigger, messier corpus, not a third chunking strategy.

## How to reproduce

```bash
pip install -r requirements.txt
python scripts/ablation.py
python scripts/ablation.py --ks 1 3 5 10
```
