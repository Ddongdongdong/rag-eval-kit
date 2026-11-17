# rag-eval-kit

An offline RAG evaluation toolkit: ingest documents, retrieve with TF-IDF,
rerank, and score the result with reproducible metrics — no LLM API, no
API key, no network required by default. An optional OpenAI-compatible
LLM mode exists but stays off unless you opt in with environment variables.

Built for one question that every production RAG / Agent team has to answer:
**did the retriever actually return the right evidence, and does the answer
stay inside that evidence?** This kit gives you a number for that, instead
of a vibe.

## What problem does it solve?

Most RAG demos only show that *an* answer came back. They do not show:

- whether the gold document was in the top-k at all (Recall@k),
- whether the citations point at the right documents,
- how much of the gold answer the response actually covers,
- whether the answer asserts things the retrieved context never said.

`rag-eval-kit` measures all four, deterministically, on your own data.
Swap in your documents and QA set and you get the same report shape —
which is exactly what a hiring reviewer or an ExpertHub portfolio
reviewer can verify in 3 minutes.

## Architecture

```text
data/sample_docs/*.md ──┐
data/sample_qa.jsonl    │  (offline, no LLM by default)
data/sample_qa_hard.jsonl│  (paraphrased + cross-doc)
                        ▼
              rag/chunker.py ── fixed-length / heading strategies
                        │
                        ▼
            rag/retriever.py ── TF-IDF + overlap rerank
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
   rag/evaluator.py        rag/trace.py
   Recall@k, citation,    trace_id, latency,
   coverage, halluc.       retrieved ids → JSONL
            │                       │
            └───────────┬───────────┘
                        ▼
              app/main.py (FastAPI)
   GET /health · POST /ingest · POST /query
   POST /evaluate · GET /traces
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
   scripts/run_demo.py     scripts/ablation.py
   → reports/demo_report.md → reports/ablation_report.md
   demo/index.html (static frontend, calls the local API)
```

Key design choice: the evaluation core never calls an LLM by default.
Retrieval is TF-IDF (standard library only), answers are extractive.
`rag/llm.py` is the single, **disabled-by-default** seam where a real
model can be plugged in — set `RAG_EVAL_LLM=1` plus
`RAG_EVAL_LLM_BASE_URL` / `RAG_EVAL_LLM_API_KEY` /
`RAG_EVAL_LLM_MODEL` and run with `--mode llm`. Without them the
llm run fails loudly instead of silently falling back to extractive
answers. `scripts/ablation.py` runs fixed vs heading chunking over
both QA sets and prints the Recall@k curves.

## Quick start

Requires Python 3.11+.

```bash
git clone <your-fork> rag-eval-kit
cd rag-eval-kit
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. One-click demo — ingest + evaluate both QA sets, writes reports/demo_report.md
python scripts/run_demo.py

# 1b. Chunking ablation — fixed vs heading, writes reports/ablation_report.md
python scripts/ablation.py

# 1c. LLM answers (optional; fails loudly without env vars, see rag/llm.py)
RAG_EVAL_LLM=1 RAG_EVAL_LLM_BASE_URL=<url> RAG_EVAL_LLM_API_KEY=<key> \
  RAG_EVAL_LLM_MODEL=<model> python scripts/run_demo.py --mode llm

# 2. API server (sample docs are auto-loaded on startup)
uvicorn app.main:app --reload
# → http://127.0.0.1:8000/docs

# 3. Try it
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "How long does a refund take?", "top_k": 3}'
curl -X POST http://127.0.0.1:8000/evaluate \
  -H 'Content-Type: application/json' -d '{"top_k": 5}'
curl -X POST http://127.0.0.1:8000/evaluate \
  -H 'Content-Type: application/json' -d '{"top_k": 5, "qa_set": "hard"}'

# 4. Tests
pytest -q
```

Docker:

```bash
docker build -t rag-eval-kit .
docker run -p 8000:8000 rag-eval-kit
```

Open `demo/index.html` in a browser while the API is running to query
and inspect traces from a minimal UI (it calls `http://localhost:8000`).

## Metrics

| Metric | Definition | Range |
|---|---|---|
| Recall@k | Share of gold chunk/doc ids found in the top-k retrieved results | 0–1, higher is better |
| Citation accuracy | Share of gold doc ids present in the predicted citations | 0–1, higher is better |
| Answer coverage | Token-overlap F1 between the extractive answer and the gold answer | 0–1, higher is better |
| Hallucination proxy | Share of answer tokens absent from the retrieved context | 0–1, **lower is better** |

Notes on reading the numbers honestly:

- **Recall@k / citation accuracy** are retrieval metrics — they tell you
  whether the right evidence was found. They are the most trustworthy
  numbers in this kit.
- **Answer coverage** is deliberately conservative: the extractive
  answer concatenates up to 600 chars of retrieved text, so extra
  context dilutes the F1 even when the gold facts are present. Compare
  it across chunking / rerank settings on the *same* QA set, not
  against an absolute bar.
- **Hallucination proxy** is a proxy, not a judgement: it counts tokens
  the answer uses that never appear in the retrieved context. An
  extractive answer should score near 0; a generative answer that
  invents numbers will score high.

## Example results (v0.2)

Bundled demo sets: 6 generic business documents, 22 chunks
(heading strategy), 12 QA pairs per set, `top_k=5`, rerank on.
Reproduce with `python scripts/run_demo.py` — full tables in
[`reports/demo_report.md`](reports/demo_report.md); chunking
comparison in [`reports/ablation_report.md`](reports/ablation_report.md).

| QA set | Recall@5 | Citation accuracy | Answer coverage | Hallucination proxy |
|---|---|---|---|---|
| easy | **1.000** | **1.000** | 0.363 | 0.009 |
| hard (paraphrased + cross-doc) | **1.000** | **1.000** | 0.410 | 0.007 |

Interpretation: on this small, cleanly separated sample set the
retriever finds the right document every time, and the extractive
answer stays inside the retrieved context (hallucination proxy ≤0.009).
Coverage reflects the conservative extractive answer, not a retrieval
failure — see the metric notes above.

Read the hard set honestly: it does **not** break Recall@5. Both sets
sit at 1.000 at k=5, because top-5 over a 6-document corpus is too
forgiving to discriminate. Where the hard set bites is at smaller k:

| Recall@k | easy / fixed | easy / heading | hard / fixed | hard / heading |
|---|---|---|---|---|
| @1 | 1.000 | 0.917 | 0.667 | 0.667 |
| @3 | 1.000 | 0.917 | 0.917 | 1.000 |
| @5 | 1.000 | 1.000 | 1.000 | 1.000 |

At k=1 the paraphrased and cross-document questions drop recall from
0.917–1.000 (easy) to 0.667 (hard). The chunking ablation itself is
nearly a tie at k≥3 — fixed is ahead at Recall@1 on the easy set
(1.000 vs 0.917) and the two tie at k=5 everywhere. On a corpus this
small that is the real finding: the next discrimination has to come
from a bigger, messier corpus (v0.3), not from a third chunking strategy.

## Project layout

```text
app/main.py            FastAPI: /health, /ingest, /query, /evaluate, /traces
rag/chunker.py         fixed-length + heading/paragraph chunking
rag/retriever.py       TF-IDF retriever + overlap rerank (stdlib only)
rag/evaluator.py       Recall@k, citation accuracy, coverage, hallucination proxy
rag/trace.py           JSONL trace log (trace_id, latency, retrieved ids)
rag/llm.py             optional OpenAI-compatible LLM client, disabled by default
data/sample_docs/      6 generic business docs (written for this repo)
data/sample_qa.jsonl   12 QA pairs with gold answers + gold doc ids
data/sample_qa_hard.jsonl  12 harder QA pairs: 6 paraphrased, 6 cross-doc
scripts/run_demo.py    one-click demo → reports/demo_report.md (--mode extractive|llm)
scripts/ablation.py    fixed vs heading ablation → reports/ablation_report.md
demo/index.html        minimal static frontend
tests/                 20 pytest cases (chunker, retriever, evaluator, trace, v0.2)
Dockerfile, pyproject.toml, requirements.txt, LICENSE (MIT)
```

## Roadmap

- [x] **v0.2 — Real LLM answers, side by side.** `rag/llm.py:LLMClient.generate()`
  is implemented for OpenAI-compatible endpoints, key read from the
  environment only, with `--mode extractive|llm` scoring both modes on
  the same QA set. Without env vars the llm run fails loudly; it never
  silently falls back to extractive answers.
- [x] **v0.2 — Recall@k curves.** `scripts/ablation.py` runs fixed vs
  heading chunking over both QA sets and prints Recall@1/3/5/10.
- [x] **v0.2 — Harder eval set.** `data/sample_qa_hard.jsonl`: 6 paraphrased
  + 6 cross-document questions. It discriminates at Recall@1 (0.667 vs
  0.917–1.000 easy), not at Recall@5 — see the honest note above.
- [ ] **v0.3 — Better retrieval.** Embedding retriever behind the same
  `search()` interface, hybrid TF-IDF + embedding, and the same
  Recall@k curves to show whether embeddings actually help.
- [ ] **v0.3 — Bigger, messier eval set.** More documents with overlapping
  vocabulary, paraphrased + multi-doc questions, so Recall@5 stops
  saturating at 1.0 and the ablation can discriminate at k>=3.

## Contributing

PRs welcome — especially new QA sets, a new retriever behind the
`search()` interface, or a new metric in `rag/evaluator.py` with tests.
Please keep the offline default: `pytest -q` and
`python scripts/run_demo.py` must pass with no network and no key.

---

# rag-eval-kit（中文）

一个**离线** RAG 评测工具箱：导入文档，用 TF-IDF 召回 + Rerank，
再用可复现的指标给结果打分。不需要调用任何 LLM API，不需要
API Key，不需要联网。

它要回答的是每个做 RAG / Agent 落地的团队都必须回答的问题：
**检索器到底有没有找回正确的证据，答案有没有超出这些证据？**
这个工具给的是数字，不是感觉。

## 解决什么问题

大多数 RAG Demo 只证明了“返回了一个答案”，没有证明：

- 标准答案所在的文档有没有进 top-k（Recall@k）
- 引用有没有指到正确的文档
- 答案覆盖了标准答案的多少内容
- 答案有没有说出检索结果里根本没有的东西

这个工具把四项都测了，而且是确定性的。换成你自己的文档
和问答集，就能得到同一份报告——面试官或作品集审核者
3 分钟就能验证。

## 架构

见上面英文部分的架构图。核心设计：评测核心默认不调 LLM，
检索是纯标准库 TF-IDF，答案是抽取式的。`rag/llm.py` 是唯一
的、**默认关闭**的 LLM 接入点，v0.2 已实现 OpenAI 兼容调用，
Key 只从环境变量读；无配置时明确报错，不会静默回退。

## 快速开始

Python 3.11+：

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_demo.py        # 一键 Demo，两套题都跑，生成 reports/demo_report.md
python scripts/ablation.py        # 切分消融，生成 reports/ablation_report.md
uvicorn app.main:app --reload     # API，启动时自动加载示例文档
pytest -q                         # 测试
# LLM 模式（可选，无环境变量会明确报错，不会静默回退）：
RAG_EVAL_LLM=1 RAG_EVAL_LLM_BASE_URL=<url> RAG_EVAL_LLM_API_KEY=<key> \
  RAG_EVAL_LLM_MODEL=<model> python scripts/run_demo.py --mode llm
```

接口：`GET /health`、`POST /ingest`、`POST /query`、
`POST /evaluate`（可传 `{"qa_set": "hard"}` 跑难题集）、`GET /traces`，交互文档在 `/docs`。

## 指标定义

| 指标 | 定义 | 方向 |
|---|---|---|
| Recall@k | top-k 检索结果里找到的标准来源占比 | 越高越好 |
| 引用准确率 | 预测引用里包含的标准文档占比 | 越高越好 |
| 答案覆盖率 | 生成答案与标准答案的 token 重叠 F1 | 越高越好 |
| 幻觉代理指标 | 答案中未出现在检索上下文里的 token 占比 | **越低越好** |

诚实说明：答案覆盖率是保守的——抽取式答案会拼最多 600 字
检索文本，多余上下文会拉低 F1，即使关键事实都在里面。
所以它适合在**同一套问答集上对比不同切分 / Rerank 设置**，
不适合拿绝对值去跟别的系统比。幻觉代理指标只是代理指标，
不是对单条答案的定性判断。

## 示例结果（v0.2）

自带 Demo 集：6 篇通用业务文档、22 个 chunk（按标题切分）、
每套 12 条问答，`top_k=5`，开 Rerank。跑 `python scripts/run_demo.py`
可复现，完整表格见 `reports/demo_report.md`，切分对比见
`reports/ablation_report.md`。

| 问答集 | Recall@5 | 引用准确率 | 答案覆盖率 | 幻觉代理指标 |
|---|---|---|---|---|
| 简单集 | **1.000** | **1.000** | 0.363 | 0.009 |
| 难题集（改写 + 跨文档） | **1.000** | **1.000** | 0.410 | 0.007 |

诚实说明：难题集**没有**把 Recall@5 打下来，两套题在 k=5 时
都是 1.000——6 篇文档的语料上 top-5 太宽容了，区分不出来。
难题集真正起作用的地方在更小的 k：

| Recall@k | 简单集 / 固定 | 简单集 / 标题 | 难题集 / 固定 | 难题集 / 标题 |
|---|---|---|---|---|
| @1 | 1.000 | 0.917 | 0.667 | 0.667 |
| @3 | 1.000 | 0.917 | 0.917 | 1.000 |
| @5 | 1.000 | 1.000 | 1.000 | 1.000 |

改写和跨文档问题把 Recall@1 从 0.917–1.000 拉到 0.667。
切分消融本身在 k≥3 时基本打平——固定切分在简单集 Recall@1
上领先（1.000 对 0.917），k=5 时处处打平。语料这么小，这个
结论本身就是发现：下一步的区分度要靠更大、更杂的语料（v0.3），
而不是第三种切分方法。

## Roadmap

- **v0.2 已完成：接真实 LLM，并排对比。** `rag/llm.py` 已实现
  OpenAI 兼容接口，Key 只从环境变量读，加 `--mode extractive|llm`，
  同一套题同时测两种答案。无环境变量时明确报错，绝不静默回退。
- **v0.2 已完成：切分消融实验。** `scripts/ablation.py` 一条命令
  同时跑固定长度切分和按标题切分，输出 Recall@1/3/5/10 曲线。
- **v0.2 已完成：更难的评测集。** `data/sample_qa_hard.jsonl`：
  6 条改写 + 6 条跨文档问题，在 Recall@1 上能区分（0.667 对
  0.917–1.000），在 Recall@5 上还区分不出来——见上面的诚实说明。
- **v0.3：更好的检索。** 在同一个 `search()` 接口后面接向量检索，
  做 TF-IDF + 向量的混合检索，用同一套 Recall@k 曲线验证向量
  到底有没有帮助。
- **v0.3：更大、更杂的评测集。** 更多文档、词汇重叠更大，改写
  和跨文档问题更多，让 Recall@5 不再轻易到 1.0。

## License

MIT — 见 [LICENSE](LICENSE)。
