# Adaptive Token-Budget CRAG (AT-CRAG)

> **Research-grade implementation** of Adaptive Token-Budget Corrective Retrieval Augmented Generation — a novel extension of the CRAG framework (Yan et al., 2024, arXiv:2401.15884).

---

## 🎯 Core Idea

The original CRAG treats **every query identically**: it always retrieves the same top-K documents and runs the same corrective loop regardless of query difficulty. This wastes compute on easy queries and under-retrieves on hard ones.

**AT-CRAG** inserts a lightweight **Query Complexity Controller** *before* the CRAG retrieval loop. This controller:

1. **Classifies** each query as SIMPLE / MEDIUM / COMPLEX using handcrafted features + a trained scikit-learn classifier.
2. **Assigns** a dynamic retrieval budget:

| Complexity | K (documents) | N (iterations) |
|------------|:---:|:---:|
| SIMPLE     |  3  |  1  |
| MEDIUM     |  5  |  2  |
| COMPLEX    | 10  |  3  |

3. **Runs the identical CRAG loop** — only K and N differ.

**Hypothesis:** Adaptive retrieval depth improves the accuracy-latency Pareto front over fixed-depth CRAG.

---

## 📂 Project Structure

```
at_crag/
├── retriever/               # Dense retrieval + FAISS
│   ├── corpus.py            # Wikipedia chunk loader
│   ├── embedder.py          # all-MiniLM-L6-v2 encoder
│   └── faiss_index.py       # FAISS flat-L2 index
├── evaluator/
│   └── relevance_scorer.py  # Cross-encoder CRAG scorer
├── complexity/              # ★ Novel component
│   ├── feature_extractor.py # 15-dim handcrafted features
│   └── classifier.py        # Complexity classifier (0/1/2)
├── controller/
│   └── budget_controller.py # Complexity → (K, N) mapping
├── generator/
│   └── reader.py            # flan-t5-base answer generator
├── pipeline/
│   ├── baseline_crag.py     # Fixed K=5, N=2 (control)
│   └── at_crag.py           # Dynamic K, N (treatment)
├── eval/
│   ├── metrics.py           # EM, token F1, latency
│   └── run_eval.py          # Full evaluation harness
├── experiments/results/     # JSON logs per run
├── build_index.py           # One-time FAISS index builder
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the FAISS index (one-time, ~10-20 min on CPU)

```bash
# Full build (50K articles)
python -m at_crag.build_index

# Quick dev build (1K articles)
python -m at_crag.build_index --limit 1000
```

### 3. Train the complexity classifier

```bash
python -m at_crag.complexity.classifier
```

### 4. Run evaluation

```bash
# Quick dev run (50 questions)
python -m at_crag.eval.run_eval --num 50

# Full evaluation (500 questions) with ablation
python -m at_crag.eval.run_eval --num 500 --ablation
```

---

## 🧪 Evaluation

The evaluation harness (`eval/run_eval.py`) runs both pipelines on TriviaQA validation questions and outputs:

| Method        | EM   | F1   | Avg Latency (s) |
|---------------|------|------|-----------------|
| Baseline CRAG | 0.XX | 0.XX | X.XX            |
| AT-CRAG       | 0.XX | 0.XX | X.XX            |

The ablation table breaks AT-CRAG metrics down by complexity tier, showing that:
- **SIMPLE** queries are faster (shallower retrieval)
- **COMPLEX** queries are more accurate (deeper retrieval)

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| Retrieval encoder | `all-MiniLM-L6-v2` (384-dim, CPU) |
| Relevance scorer | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generator | `google/flan-t5-base` (~250 MB) |
| Vector search | `faiss-cpu` (flat L2) |
| Complexity classifier | scikit-learn LogisticRegression |
| Evaluation dataset | TriviaQA (rc.nocontext) |

**All operations run on CPU.** No GPU required.

---

## 📖 Citation

```bibtex
@article{yan2024crag,
  title={Corrective Retrieval Augmented Generation},
  author={Yan, Shi-Qi and Gu, Jia-Chen and Zhu, Yun and Ling, Zhen-Hua},
  journal={arXiv preprint arXiv:2401.15884},
  year={2024}
}
```

---

## 📄 License

This project is for research and educational purposes.
