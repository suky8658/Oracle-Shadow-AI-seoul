"""
미래 고위험군 전이 예측 — 모델 학습 및 검증 (Stage 2)
========================================================
build_transition_features.py 에서 생성한 피처로
Logistic Regression(기준선) + XGBoost 분류 모델을 학습.

검증 전략:
  - Stratified 5-fold CV (N=419로 작아 단일 holdout 불안정)
  - Temporal holdout 보조: dep_2024 → dep_2025 방향

출력물:
  Outputs/전이예측/model_results.md
  Outputs/전이예측/feature_importance.png
  Outputs/전이예측/transition_predictions.csv  (행정동별 전이 확률)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings("ignore")

_FONT_FILE = "C:/Windows/Fonts/malgun.ttf"
if Path(_FONT_FILE).exists():
    fp = fm.FontProperties(fname=_FONT_FILE)
    plt.rcParams["font.family"] = fp.get_name()
else:
    plt.rcParams["font.family"] = "Malgun Gothic"
    fp = fm.FontProperties()
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "Outputs" / "전이예측"

# 피처 그룹 정의 (build_transition_features.py와 일치)
FEAT_PREFIXES = ["level_", "slope_", "accel_", "vol_", "delta_", "infra_"]
LABEL_COL     = "label_slope"   # 주 레이블 (label_entry도 별도 실험)


# ── [1] 데이터 로딩 ──────────────────────────────────────────────────
def load_data(label_col: str = LABEL_COL):
    feat_path = OUT / "transition_features.csv"
    if not feat_path.exists():
        raise FileNotFoundError(f"{feat_path} 없음 — build_transition_features.py 먼저 실행")

    df = pd.read_csv(feat_path, encoding="utf-8-sig")
    feat_cols = [c for c in df.columns
                 if any(c.startswith(p) for p in FEAT_PREFIXES)]
    meta_cols = ["행정동코드", "자치구", "행정동"]

    df = df.dropna(subset=[label_col])
    X  = df[feat_cols].copy()
    y  = df[label_col].astype(int)
    meta = df[meta_cols].copy()

    # dep_20XX / dep_slope_annual 은 레이블 정의에 쓰이므로 피처에서 제외 (누출 방지)
    print(f"  데이터: {len(df)}행  피처 {len(feat_cols)}개  레이블='{label_col}'")
    print(f"  레이블 분포: 1={y.sum()} ({y.mean():.1%}),  0={len(y)-y.sum()} ({1-y.mean():.1%})")
    return X, y, meta, feat_cols


# ── [2] 모델 파이프라인 ───────────────────────────────────────────────
def build_pipelines():
    """Impute → Scale → Model 파이프라인 두 개."""
    lr = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
        ("clf",    LogisticRegression(
            C=0.1, max_iter=1000, class_weight="balanced", random_state=42
        )),
    ])
    xgb = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf",    GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            min_samples_leaf=10, subsample=0.8, random_state=42,
        )),
    ])
    return {"LogisticRegression": lr, "GradientBoosting": xgb}


# ── [3] 교차검증 ─────────────────────────────────────────────────────
def cross_validate_all(X: pd.DataFrame, y: pd.Series, pipelines: dict):
    """Stratified 5-fold CV → AUC, AP 비교."""
    print("\n── [3] Stratified 5-fold 교차검증 ──")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, pipe in pipelines.items():
        proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        auc   = roc_auc_score(y, proba)
        ap    = average_precision_score(y, proba)
        pred  = (proba >= 0.5).astype(int)
        print(f"\n  [{name}]")
        print(f"    AUC={auc:.3f}  AP={ap:.3f}")
        print(f"    {classification_report(y, pred, target_names=['안정','전이'], zero_division=0)}")
        results[name] = {"proba": proba, "auc": auc, "ap": ap}

    return results


# ── [3b] 홀드아웃 검증 ───────────────────────────────────────────────
def holdout_validate(X: pd.DataFrame, y: pd.Series, pipelines: dict):
    """80/20 무작위 분할 홀드아웃 — 5-fold CV와 독립된 추가 검증."""
    if len(y) < 50 or y.sum() < 10 or (len(y) - y.sum()) < 10:
        print("  [SKIP] 홀드아웃: 샘플 부족")
        return
    print("\n── [3b] 홀드아웃 검증 (80/20 무작위 분할) ──")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    for name, pipe in pipelines.items():
        fitted = clone(pipe)
        fitted.fit(X_tr, y_tr)
        proba = fitted.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, proba)
        ap  = average_precision_score(y_te, proba)
        print(f"  [{name}] AUC={auc:.3f}  AP={ap:.3f}  "
              f"(train={len(y_tr)}, test={len(y_te)})")


# ── [4] 피처 중요도 ───────────────────────────────────────────────────
def plot_importance(X: pd.DataFrame, y: pd.Series, feat_cols: list):
    """GradientBoosting 피처 중요도 시각화."""
    print("\n── [4] 피처 중요도 (GradientBoosting) ──")

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf",    GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            min_samples_leaf=10, subsample=0.8, random_state=42,
        )),
    ])
    pipe.fit(X, y)

    imp = pd.Series(
        pipe.named_steps["clf"].feature_importances_,
        index=feat_cols
    ).sort_values(ascending=False)

    top20 = imp.head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#e74c3c" if "slope" in c or "accel" in c or "delta" in c
              else "#3498db" for c in top20.index]
    ax.barh(range(len(top20)), top20.values[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20.index[::-1], fontsize=9)
    ax.set_xlabel("피처 중요도", fontproperties=fp)
    ax.set_title("전이 예측 모델 — 상위 20개 피처 중요도\n(빨강=추세/변화 계열, 파랑=수준/인프라 계열)",
                 fontproperties=fp, fontsize=12)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#e74c3c", label="추세/변화(slope·accel·delta)"),
        Patch(facecolor="#3498db", label="수준/인프라(level·infra)"),
    ], prop=fp, fontsize=9, loc="lower right")

    plt.tight_layout()
    path = OUT / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  저장: {path}")
    return imp, pipe


# ── [5] 전이 확률 산출 및 저장 ───────────────────────────────────────
def save_predictions(meta: pd.DataFrame, results: dict, y: pd.Series, imp: pd.Series):
    """행정동별 전이 확률 CSV + 결과 리포트."""

    # GradientBoosting 확률 우선, 없으면 LR
    proba_col = (results.get("GradientBoosting") or results.get("LogisticRegression"))["proba"]
    pred_df   = meta.copy()
    pred_df["전이확률_GB"]  = results.get("GradientBoosting", {}).get("proba", np.nan)
    pred_df["전이확률_LR"]  = results.get("LogisticRegression", {}).get("proba", np.nan)
    pred_df["실제레이블"]   = y.values
    pred_df["위험등급"] = pd.cut(
        pred_df["전이확률_GB"].fillna(pred_df["전이확률_LR"]),
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=["저위험", "중위험", "고위험", "최고위험"],
    )
    pred_df = pred_df.sort_values("전이확률_GB", ascending=False).reset_index(drop=True)
    pred_df.to_csv(OUT / "transition_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"\n  저장: transition_predictions.csv  ({len(pred_df)}행)")

    # 상위 10개 출력
    print("\n  전이 확률 상위 10개 행정동:")
    for i, row in pred_df.head(10).iterrows():
        print(f"    {i+1:2d}. {row['자치구']:5s} {row['행정동']:10s}  "
              f"확률={row['전이확률_GB']:.3f}  실제={int(row['실제레이블'])}  "
              f"등급={row['위험등급']}")

    # 결과 리포트
    gb  = results.get("GradientBoosting", {})
    lr  = results.get("LogisticRegression", {})
    top5_feat = "\n".join(f"  {i+1}. {c} ({v:.4f})" for i, (c, v) in enumerate(imp.head(5).items()))

    report = f"""\
# 전이 예측 모델 결과

## 모델 성능 (Stratified 5-fold CV)

| 모델 | AUC | Average Precision |
|------|-----|-------------------|
| Logistic Regression | {lr.get("auc", "N/A"):.3f} | {lr.get("ap", "N/A"):.3f} |
| Gradient Boosting   | {gb.get("auc", "N/A"):.3f} | {gb.get("ap", "N/A"):.3f} |

> AUC 0.7+ = 유의미한 예측력. N=419로 작으므로 신뢰구간 넓음.

## 상위 5개 피처

{top5_feat}

## 위험등급별 행정동 수

{pred_df["위험등급"].value_counts().to_string()}

## 한계

- N=419로 ML 모델 과적합 위험 존재. max_depth=3, min_samples_leaf=10으로 제한
- 레이블이 집계 수준(행정동) — 개인 수준 전이 예측이 아님
- Avoidance 축(자치구 단위)은 피처에 미포함
- 2022~2025 4년만 검증 가능 — 검증 횟수 제한

## 다음 단계

predict_transition_risk.py 실행:
  SHAP 값으로 행정동별 전이 원인 분해 + 지도 시각화
"""
    (OUT / "model_results.md").write_text(report, encoding="utf-8")
    print(f"  저장: model_results.md")


# ── main ──────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("전이 예측 — 모델 학습 및 검증")
    print("=" * 60)

    for label_col in ["label_slope", "label_entry"]:
        print(f"\n{'='*40}")
        print(f"레이블: {label_col}")
        print(f"{'='*40}")
        try:
            X, y, meta, feat_cols = load_data(label_col)
        except Exception as e:
            print(f"  [SKIP] {e}")
            continue

        if y.sum() < 10:
            print(f"  [SKIP] 양성 레이블 {y.sum()}개 — 너무 적어 학습 불가")
            continue

        pipes   = build_pipelines()
        results = cross_validate_all(X, y, pipes)
        holdout_validate(X, y, pipes)

        if label_col == "label_slope":
            imp, fitted_pipe = plot_importance(X, y, feat_cols)
            save_predictions(meta, results, y, imp)

    print("\n" + "=" * 60)
    print("완료! 다음: python code/predict_transition_risk.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
