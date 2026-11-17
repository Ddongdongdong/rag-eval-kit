# rag-eval-kit

An offline RAG evaluation toolkit: ingest documents, retrieve with TF-IDF,
rerank, and score the result with reproducible metrics — no LLM API, no
API key, no network required.

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
data/sample_qa.jsonl    │  (offline, no LLM)
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
                        ▼
            scripts/run_demo.py → reports/demo_report.md
            demo/index.html (static frontend, calls the local API)
```

Key design choice: the evaluation core never calls an LLM. Retrieval is
TF-IDF (standard library only), answers are extractive. `rag/llm.py`
is the single, **disabled-by-default** seam where a real model can be
plugged in later — see Roadmap.

## Quick start

Requires Python 3.11+.

```bash
git clone <your-fork> rag-eval-kit
cd rag-eval-kit
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. One-click demo — ingest + evaluate, writes reports/demo_report.md
python scripts/run_demo.py

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

## Example results

Bundled demo set: 6 generic business documents, 22 chunks
(heading strategy), 12 QA pairs, `top_k=5`, rerank on.
Reproduce with `python scripts/run_demo.py` — full table in
[`reports/demo_report.md`](reports/demo_report.md).

| Metric | Value |
|---|---|
| Recall@5 | **1.000** |
| Citation accuracy | **1.000** |
| Answer coverage | **0.363** |
| Hallucination proxy | **0.009** |

Interpretation: on this small, cleanly separated sample set the
retriever finds the right document every time, and the extractive
answer stays inside the retrieved context (hallucination proxy 0.009).
Coverage 0.363 reflects the conservative extractive answer, not a
retrieval failure — see the metric notes above. On a larger, messier
corpus these numbers will drop; that is the point of having the harness.

## Project layout

```text
app/main.py            FastAPI: /health, /ingest, /query, /evaluate, /traces
rag/chunker.py         fixed-length + heading/paragraph chunking
rag/retriever.py       TF-IDF retriever + overlap rerank (stdlib only)
rag/evaluator.py       Recall@k, citation accuracy, coverage, hallucination proxy
rag/trace.py           JSONL trace log (trace_id, latency, retrieved ids)
rag/llm.py             optional LLM seam, disabled by default
data/sample_docs/      6 generic business docs (written for this repo)
data/sample_qa.jsonl   12 QA pairs with gold answers + gold doc ids
scripts/run_demo.py    one-click demo → reports/demo_report.md
demo/index.html        minimal static frontend
tests/                 13 pytest cases (chunker, retriever, evaluator, trace)
Dockerfile, pyproject.toml, requirements.txt, LICENSE (MIT)
```

## Roadmap

- [ ] **v0.2 — Real LLM answers, side by side.** Implement
  `rag/llm.py:LLMClient.generate()` for an OpenAI-compatible endpoint
  (e.g. DeepSeek), key read from the environment only, and add a
  `--mode extractive|llm` flag so the same QA set scores both modes.
  Expected effect: coverage up, hallucination proxy becomes the
  interesting metric.
- [ ] **v0.2 — Better retrieval.** Embedding retriever behind the same
  `search()` interface, hybrid TF-IDF + embedding, and Recall@k
  curves for k = 1 / 3 / 5 / 10.
- [ ] **v0.3 — Chunking ablation.** One command that runs
  fixed vs heading chunking over the QA set and prints the metric
  delta — the experiment this kit exists to run.
- [ ] **v0.3 — Eval datasets.** A second, harder QA set with
  paraphrased questions and multi-doc questions, so Recall@5 stops
  saturating at 1.0.

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

见上面英文部分的架构图。核心设计：评测核心永远不调 LLM，
检索是纯标准库 TF-IDF，答案是抽取式的。`rag/llm.py` 是唯一
的、**默认关闭**的 LLM 接入点，以后接真实模型只改这一个文件。

## 快速开始

Python 3.11+：

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_demo.py        # 一键 Demo，生成 reports/demo_report.md
uvicorn app.main:app --reload     # API，启动时自动加载示例文档
pytest -q                         # 测试
```

接口：`GET /health`、`POST /ingest`、`POST /query`、
`POST /evaluate`、`GET /traces`，交互文档在 `/docs`。

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

## 示例结果

自带 Demo 集：6 篇通用业务文档、22 个 chunk（按标题切分）、
12 条问答，`top_k=5`，开 Rerank。跑 `python scripts/run_demo.py`
可复现，完整表格见 `reports/demo_report.md`。

| 指标 | 数值 |
|---|---|
| Recall@5 | **1.000** |
| 引用准确率 | **1.000** |
| 答案覆盖率 | **0.363** |
| 幻觉代理指标 | **0.009** |

解读：在这套小而干净的示例集上，检索每次都找对了文档，
抽取式答案也基本没有超出检索上下文（幻觉代理 0.009）。
覆盖率 0.363 反映的是抽取式答案的保守性，不是检索失败。
换成更大、更杂的语料，这些数字会下降——这正是要这个
评测框架的原因。

## Roadmap

- **v0.2：接真实 LLM，并排对比。** 在 `rag/llm.py` 里实现
  OpenAI 兼容接口（如 DeepSeek），Key 只从环境变量读，
  加 `--mode extractive|llm`，同一套题同时测两种答案。
  预期：覆盖率上升，幻觉代理指标变成真正值得看的指标。
- **v0.2：更好的检索。** 在同一个 `search()` 接口后面接
  向量检索，做 TF-IDF + 向量的混合检索，出 k=1/3/5/10
  的 Recall 曲线。
- **v0.3：切分消融实验。** 一条命令同时跑固定长度切分
  和按标题切分，输出指标差值——这正是这个工具存在的意义。
- **v0.3：更难的评测集。** 加一套改写过的问题和跨文档
  问题，让 Recall@5 不再轻易到 1.0。

## License

MIT — 见 [LICENSE](LICENSE)。
