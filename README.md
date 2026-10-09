# 🛰️ Space Debris Crisis - Autonomous Collision Avoidance System

> **ESA Kelvins Collision Avoidance & Space Debris Mitigation Challenge**
> Hackfest 1C | End-to-End Autonomous Pipeline & Decision Support System

---

## 📌 Architecture & 5-Lane Pipeline Overview

The repository is structured into five isolated, modular lanes. Each teammate owns one lane end-to-end and codes against fixed shared interface contracts established at project kickoff.

```
                  ┌─────────────────────────────────────────┐
                  │ P1: Data & Evaluation (ingest & eval)   │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │ P2: SGP4 Propagation & Screener         │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │ P3: XGBoost/LSTM Risk & Calibration     │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │ P4: Maneuver Optimizer (2D Foster/Chan) │ ◄── (Owned by Anirudh)
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │ P5: XAI Telemetry & Operator Dashboard  │
                  └─────────────────────────────────────────┘
```

---

## 👥 Role Assignments & Function Contracts

| Role | Domain | Deliverables & Contract Signatures |
| :--- | :--- | :--- |
| **P1** | **Data & Evaluation** | Ingestion (`ESA Kelvins CDM`, `CelesTrak TLE`, `F10.7/Kp`). Event schema generator & `data/eval.py` (`evaluate_risk_model(y_true, y_prob, lead_times)`). |
| **P2** | **Propagation & Screening** | SGP4 baseline + along-track ML error correction + KD-Tree all-vs-all candidate filter. `screen(catalog, horizon="7d")` -> candidate list under a few minutes. |
| **P3** | **Risk Model & Calibration** | XGBoost + Sequence stacked model. Isotonic calibration & conformal confidence bands. `predict_risk(event)` -> `{poc, lo, hi, priority}`. |
| **P4** | **Maneuver Optimization** *(Anirudh)* | 2D Foster/Chan analytical PoC, along-track $\Delta v$ grid search ($0.05 - 2.0$ m/s), Pareto optimization & P2 secondary conjunction filter. `recommend_maneuver(event)` -> ranked `{dv, dir, t_burn, residual_poc}`. |
| **P5** | **XAI, Dashboard & Pitch** | SHAP feature attribution turned into persona-based plain-English telemetry readouts (`xai/explainer.py`) + Streamlit Operator Dashboard (`app/main.py`). |

---

## 🔒 Locked-in Data Schema & Conventions

- **High-Risk Label Definition**: $\text{final } \log_{10}(\text{PoC}) \ge -6.0$ (i.e. $\text{PoC} \ge 10^{-6}$).
- **Data Splitting Rule**: Strict split **by `event_id`**, NOT by CDM row, preventing data leakage across sequence steps.
- **Lead Time Claim**: Model trained/evaluated on CDMs at least **2–3 days before TCA**.

---

## 🌿 Git Branching Strategy & Workflow

All teammates work on dedicated feature branches cut from `main`:

```bash
# Main development branch
main

# Feature branches for teammates:
feature/p1-data-eval
feature/p2-propagation-screening
feature/p3-risk-calibration
feature/p4-maneuver-optimization    # (Anirudh's branch)
feature/p5-xai-dashboard
```

### Git Workflow:
1. Checkout your feature branch: `git checkout -b feature/p4-maneuver-optimization`
2. Commit implementation updates: `git commit -am "Implement 2D Foster PoC grid search"`
3. Push to remote: `git push origin feature/p4-maneuver-optimization`
4. Submit Pull Request and merge to `main`.

---

## ⏱️ Hackathon Checkpoints

- **Tonight**: P1 ships `v1 events.parquet`. P2-P5 work off locked stubs and mock datasets.
- **Day 2 Midday**: Modules populated with real ML/physics logic; first end-to-end integration test.
- **Day 2 Night**: End-to-end pipeline frozen. P1 executes full KPI evaluation. P5 finalizes dashboard flow.
- **Day 3 Morning**: Polish visual style, record backup demo video, rehearse pitch presentation.

---

## ⚠️ Judge Evaluation Checklist & Traps Avoided

- ✅ **Beyond Accuracy**: Evaluated using **Brier Score**, **Recall at fixed precision**, and **Reliability diagrams**.
- ✅ **No Data Leakage**: Events split strictly by `event_id`.
- ✅ **Honest Lead Times**: Evaluated at 3-7 day TCA lead times.
- ✅ **Screening Throughput**: Full-catalog candidate screening time benchmarked and recorded for presentation.

---

## 🚀 Quickstart & Verification

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Verification Tests
```bash
pytest tests/test_stubs.py
```

### 3. Launch Operator Dashboard (P5 Streamlit UI)
```bash
streamlit run app/main.py
```
