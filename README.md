# EDMD-Framework
### A Unified Event-Driven Medallion Architecture with Domain-Driven Design for Real-Time Credit Risk Feature Engineering

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![IEEE TBD](https://img.shields.io/badge/Submitted-IEEE%20Transactions%20on%20Big%20Data-blue.svg)](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6687317)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📄 Paper

> **EDMD: A Unified Event-Driven Medallion Architecture with Domain-Driven Design for Real-Time Credit Risk Feature Engineering**
> Akshay Sharma
> *Submitted to IEEE Transactions on Big Data, May 2026*
> Contact: akshay.sharma1088@gmail.com

This repository provides the complete reproducibility package for the above manuscript.

---

## 🔍 Overview

Production ML systems in regulated financial services face four engineering challenges addressed in isolation by existing architectures:

| Challenge | Existing Solution | Gap |
|-----------|------------------|-----|
| Training-serving skew | Feature stores | No formal guarantee |
| Late-arriving events | Watermark streaming | Not calibrated to financial event classes |
| Domain boundary violations | Data mesh | No ML feature engineering specification |
| Audit reproducibility | Logging | Not integrated with feature pipeline |

**EDMD** addresses all four simultaneously through formal integration of:
- **E**vent-driven processing with calibrated watermarks
- **D**omain-Driven Design bounded contexts + anti-corruption layers
- Extended four-tier **M**edallion architecture with novel Platinum tier

### Key Results (10M events/day synthetic benchmark)

| Architecture | TSS | LERR (@10% late) | DBVR | FFL p99 |
|-------------|-----|------------------|------|---------|
| **EDMD (proposed)** | **0.000*** | **99.7%** | **0.000*** | 4.6 min |
| Feature Store | 0.012 | 84.1% | N/A | 2.9 min |
| Kappa | 0.041 | 78.6% | N/A | 4.1 min |
| Pure Medallion | 0.067 | 52.4% | N/A | 6.8 min |
| Lambda | 0.083 | 61.3% | N/A | 8.3 min |

*\* = guaranteed by construction (EDMD Properties 1 and 2)*

---

## 📁 Repository Structure

```
EDMD-Framework/
│
├── src/
│   ├── data/
│   │   ├── synthetic_generator.py    # Gaussian copula event stream generator
│   │   └── feature_engineering.py   # 47-feature behavioral pipeline
│   │
│   ├── optimization/
│   │   └── threshold_calibration.py  # Algorithm 1: risk-constrained threshold
│   │
│   ├── evaluation/
│   │   └── benchmark.py              # Tables 7, 8, 10 reproduction
│   │
│   ├── monitoring/
│   │   └── drift_monitor.py          # PSI-based population drift monitor
│   │
│   ├── models/                       # Model training (see companion CLI repo)
│   ├── explainability/               # TreeSHAP pipeline
│   ├── fairness/                     # EOD, DPD, DIR computation
│   └── governance/                   # SR 11-7 / BCBS 239 audit logging
│
├── artifacts/
│   ├── feature_spec.yaml             # Complete 47-feature specification
│   ├── model_config.yaml             # Hyperparameter configuration
│   └── threshold_config.yaml         # Risk appetite parameters
│
├── docker/
│   └── Dockerfile                    # Reproducibility environment
│
├── requirements.txt
└── README.md
```

---

## 🏗️ EDMD Architecture

### Extended Four-Tier Medallion

| Tier | Name | Latency SLA | Retention | Novel? |
|------|------|------------|-----------|--------|
| Bronze | Raw Events | < 1 second | Permanent | — |
| Silver | Validated Domain Events | < 5 seconds | 7 years | — |
| Gold | Feature Aggregates | < 60 seconds | 3 years | — |
| **Platinum** | **Model-Ready Features** | **< 5 minutes** | **5 years** | **✅ Yes** |

The **Platinum tier** is the novel contribution: governance-embedded, drift-monitored, point-in-time correct model-ready features with zero training-serving skew by construction.

### Domain Bounded Contexts

```
Customer Risk Domain ──ACL──► Credit Decision Domain ──ACL──► Compliance Domain
Transaction Domain   ──ACL──►          ↑
                                 (never raw data)
```

**Domain isolation rule:** `raw_data(dA) ∩ accessible(dB) = ∅` for all dA ≠ dB

### Watermark Strategy

| Event Class | Lateness Bound δ_c | LERR |
|-------------|-------------------|------|
| TransactionEvent | 15 minutes | 99.9% |
| PaymentEvent | 72 hours | 99.8% |
| BureauRefreshEvent | 30 days | 99.7% |
| AccountChangeEvent | 24 hours | 99.9% |

---

## 🔧 Feature Families (47 features total)

| Family | Features (N) | Event Class | Lateness Bound |
|--------|-------------|-------------|----------------|
| Utilization Dynamics | 12 | TransactionEvent | 15 min |
| Payment Stability | 11 | PaymentEvent | 72 hours |
| Risk Drift Indicators | 8 | BureauRefreshEvent | 30 days |
| Behavioral Stability | 9 | TransactionEvent | 15 min |
| Model-ready vector | **47** | All | max(δ_c) |

Full specification: [`artifacts/feature_spec.yaml`](artifacts/feature_spec.yaml)

---

## ⚙️ Algorithm 1: Risk-Constrained Threshold Calibration

```
Input:  {PD_i, LGD_i, EAD_i, Δ_i} for i=1..N, EL_target, ε
Output: Optimal threshold τ*

1. Sort accounts ascending by PD_i.
2. For each candidate τ in {sorted PD values}:
   a. S(τ) ← {i : PD_i < τ}
   b. EL(τ) ← Σ_{i∈S(τ)} PD_i · LGD_i · (EAD_i + Δ_i)
   c. EOD(τ) ← |P(Approve|Y=1,A=0) − P(Approve|Y=1,A=1)|
   d. If EL(τ) ≤ EL_target AND EOD(τ) ≤ ε:
         Profit(τ) ← Σ_{i∈S(τ)} Π_i
3. Return τ* ← argmax Profit(τ)

Complexity: O(N log N)
```

---

## 🚀 Quickstart

### Option 1: pip install

```bash
git clone https://github.com/akshaysharma1088/EDMD-Framework.git
cd EDMD-Framework
pip install -r requirements.txt

# 1. Generate synthetic event stream (demo: 1k accounts, 3 days)
python src/data/synthetic_generator.py \
    --n_accounts 1000 \
    --daily_events 10000 \
    --simulation_days 3 \
    --output_path artifacts/synthetic_events.parquet \
    --seed 42

# 2. Compute 47-feature behavioral pipeline
python src/data/feature_engineering.py \
    --events_path artifacts/synthetic_events.parquet \
    --output_path artifacts/feature_store.parquet \
    --decision_date 2024-04-01

# 3. Run benchmark (reproduces Tables 7, 8, 10)
python src/evaluation/benchmark.py \
    --output_path artifacts/benchmark_results.json \
    --seed 42

# 4. Run PSI drift monitor
python src/monitoring/drift_monitor.py \
    --reference_path artifacts/feature_store.parquet \
    --current_path artifacts/feature_store.parquet \
    --output_path artifacts/psi_report.json
```

### Option 2: Docker (fully reproducible)

```bash
docker build -t edmd-framework ./docker
docker run -v $(pwd)/artifacts:/app/artifacts edmd-framework
```

---

## 📊 Reproducing Paper Results

```bash
# Table 7 — Main benchmark (10% late injection)
python src/evaluation/benchmark.py --seed 42

# Table 8 — LERR sensitivity analysis
python src/evaluation/benchmark.py --seed 42
# (Table 8 is included in default benchmark output)

# Table 10 — TSS impact on CLI model AUC
python src/evaluation/benchmark.py --seed 42
# (Table 10 is included in default benchmark output)
```

All tables reproducible with `--seed 42`.

---

## 🏛️ Governance Alignment

| Requirement | Implementation |
|-------------|---------------|
| SR 11-7: Conceptual soundness | Formal 5-tuple definition + property proofs (Section 3) |
| SR 11-7: Independent validation | Champion/challenger registry support |
| SR 11-7: Ongoing monitoring | PSI drift monitor (`src/monitoring/drift_monitor.py`) |
| SR 11-7: Change management | Pipeline version in every Platinum record |
| BCBS 239: Accuracy | Silver tier schema validation + ACL contracts |
| BCBS 239: Completeness | Algorithm 2 late-event recovery |
| BCBS 239: Timeliness | Platinum tier < 5-minute SLA |
| BCBS 239: Adaptability | Domain provenance metadata per feature |

---

## 📜 Citation

```bibtex
@article{sharma2026edmd,
  title   = {EDMD: A Unified Event-Driven Medallion Architecture with
             Domain-Driven Design for Real-Time Credit Risk Feature Engineering},
  author  = {Sharma, Akshay},
  journal = {IEEE Transactions on Big Data},
  year    = {2026},
  note    = {Submitted for review},
  url     = {https://github.com/akshaysharma1088/EDMD-Framework}
}
```

---

## 🔗 Related Work

- [Credit_Line_Determination](https://github.com/akshaysharma1088/Credit_Line_Determination) — Companion CLI paper (ESWA, under review)

---

## 📬 Contact

**Akshay Sharma** — Principal Data Engineer | IEEE Senior Member | IET Member
- Email: akshay.sharma1088@gmail.com
- LinkedIn: [linkedin.com/in/akshay-sharma-1088](https://linkedin.com/in/akshay-sharma-1088)

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

> **Data Privacy Notice:** This repository contains only synthetic data and code. No real customer records or financial institution data are included.
