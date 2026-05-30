"""
SHAP 원인 분해 — analyze_shap.py
=================================
저장된 GradientBoosting 모델로 행정동별 전이 확률의 원인을 분해.
각 행정동에 대해 "어떤 피처가 얼마나 영향을 줬는가"를 정량화.

실행:
  python code/analyze_shap.py

의존성:
  pip install shap

산출물:
  Outputs/전이예측/shap_values.csv       — 행정동 × 피처 SHAP 값
  Outputs/전이예측/shap_top3.csv         — 행정동별 주요 원인 상위 3개
  Outputs/전이예측/shap_bar.png          — 전체 평균 피처 중요도 (|SHAP|)
  Outputs/전이예측/shap_beeswarm.png     — 피처별 SHAP 값 분포
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_FONT_FILE = "C:/Windows/Fonts/malgun.ttf"
if Path(_FONT_FILE).exists():
    fp = fm.FontProperties(fname=_FONT_FILE)
    plt.rcParams["font.family"] = fp.get_name()
else:
    plt.rcParams["font.family"] = "Malgun Gothic"
    fp = fm.FontProperties()
plt.rcParams["axes.unicode_minus"] = False

try:
    import shap
except ImportError:
    print("[ERROR] shap 패키지 없음 — pip install shap 후 재실행")
    sys.exit(1)

ROOT      = Path(__file__).resolve().parent.parent
OUT       = ROOT / "Outputs" / "전이예측"
MODEL_DIR = OUT / "model"

# ── 피처 이름 한글 약칭 매핑 ──────────────────────────────────────────
FEAT_LABELS = {
    "level_외출커뮤적은": "고립수준(외출↓)",
    "level_이동횟수":    "이동수준",
    "level_배달식재료":  "배달의존수준",
    "level_독거비율":    "독거비율",
    "level_인프라":      "인프라밀도",
    "level_인구":        "인구규모",
    "delta_외출커뮤적은": "고립변화(22→23)",
    "delta_이동횟수":    "이동변화(22→23)",
    "delta_배달식재료":  "배달변화(22→23)",
    "delta_독거비율":    "독거변화(22→23)",
    "infra_slope":       "인프라추세(slope)",
    "infra_delta":       "인프라변화(delta)",
    "infra_recent":      "최근인프라",
}


def load_model_and_data():
    gb_path   = MODEL_DIR / "gb_model.pkl"
    meta_path = MODEL_DIR / "model_meta.json"
    feat_path = OUT / "transition_features.csv"

    if not gb_path.exists():
        raise FileNotFoundError(f"{gb_path} 없음 — run_all.py 먼저 실행")
    if not feat_path.exists():
        raise FileNotFoundError(f"{feat_path} 없음 — run_all.py 먼저 실행")

    with open(gb_path, "rb") as f:
        gb_pipe = pickle.load(f)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    df = pd.read_csv(feat_path, encoding="utf-8-sig")
    feat_cols = meta["feat_cols"]
    meta_cols = [c for c in meta["meta_cols"] if c in df.columns]

    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        print(f"  [WARN] 누락 피처 {len(missing)}개: {missing}")
        for c in missing:
            df[c] = np.nan

    label_col = meta["label_col"]
    df_valid  = df.dropna(subset=[label_col]).copy() if label_col in df.columns else df.copy()

    X    = df_valid[feat_cols].copy()
    meta_df = df_valid[meta_cols].copy()
    y    = df_valid[label_col].astype(int) if label_col in df_valid.columns else None

    print(f"  모델 학습일: {meta.get('trained_at','N/A')}")
    print(f"  피처 {len(feat_cols)}개, 행정동 {len(df_valid)}개")
    if y is not None:
        print(f"  레이블: {label_col}  양성={y.sum()} ({y.mean():.1%})")

    return gb_pipe, feat_cols, X, meta_df, y


def compute_shap(gb_pipe, feat_cols, X):
    """Pipeline에서 Imputer 적용 후 SHAP 계산."""
    X_imp = gb_pipe.named_steps["imp"].transform(X)
    clf   = gb_pipe.named_steps["clf"]

    explainer  = shap.TreeExplainer(clf)
    shap_vals  = explainer.shap_values(X_imp)
    base_value = float(explainer.expected_value)

    print(f"  SHAP 계산 완료 (기대값={base_value:.4f})")
    return shap_vals, X_imp, base_value


def save_shap_csv(shap_vals, feat_cols, meta_df, pred_proba):
    """행정동별 SHAP 값 CSV 저장."""
    shap_df = pd.DataFrame(shap_vals, columns=feat_cols, index=meta_df.index)
    for col in meta_df.columns:
        shap_df.insert(0, col, meta_df[col].values)
    shap_df["전이확률"] = pred_proba.round(4)
    shap_df = shap_df.sort_values("전이확률", ascending=False).reset_index(drop=True)
    shap_df.to_csv(OUT / "shap_values.csv", index=False, encoding="utf-8-sig")
    print(f"  shap_values.csv 저장 ({len(shap_df)}행)")
    return shap_df


def save_top3_csv(shap_df, feat_cols, meta_df):
    """행정동별 영향력 상위 3개 피처 저장."""
    rows = []
    shap_only = shap_df[feat_cols].values
    for i, (_, row) in enumerate(meta_df.iterrows()):
        vals = shap_only[i]
        top_idx = np.argsort(np.abs(vals))[::-1][:3]
        top3 = [(feat_cols[j], vals[j]) for j in top_idx]
        rows.append({
            "자치구":  row.get("자치구", ""),
            "행정동":  row.get("행정동", ""),
            "전이확률": shap_df["전이확률"].iloc[i],
            "1위원인": FEAT_LABELS.get(top3[0][0], top3[0][0]),
            "1위SHAP": round(top3[0][1], 4),
            "2위원인": FEAT_LABELS.get(top3[1][0], top3[1][0]),
            "2위SHAP": round(top3[1][1], 4),
            "3위원인": FEAT_LABELS.get(top3[2][0], top3[2][0]),
            "3위SHAP": round(top3[2][1], 4),
        })
    top3_df = pd.DataFrame(rows).sort_values("전이확률", ascending=False).reset_index(drop=True)
    top3_df.to_csv(OUT / "shap_top3.csv", index=False, encoding="utf-8-sig")
    print(f"  shap_top3.csv 저장 ({len(top3_df)}행)")

    print("\n  고위험 행정동 상위 10개 주요 원인:")
    for _, r in top3_df.head(10).iterrows():
        print(f"    {r['자치구']:5s} {r['행정동']:10s}  확률={r['전이확률']:.3f}"
              f"  →  {r['1위원인']}({r['1위SHAP']:+.3f})  "
              f"{r['2위원인']}({r['2위SHAP']:+.3f})  "
              f"{r['3위원인']}({r['3위SHAP']:+.3f})")
    return top3_df


def plot_shap_bar(shap_vals, feat_cols):
    """평균 |SHAP| 막대 그래프 — 전체 피처 중요도."""
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order    = np.argsort(mean_abs)
    labels   = [FEAT_LABELS.get(feat_cols[i], feat_cols[i]) for i in order]

    fig, ax = plt.subplots(figsize=(9, max(5, len(feat_cols) * 0.45)))
    colors = ["#e74c3c" if feat_cols[i].startswith("delta_") else
              "#f39c12" if feat_cols[i].startswith("infra_") else
              "#3498db" for i in order]
    bars = ax.barh(range(len(order)), mean_abs[order], color=colors, edgecolor="white", height=0.65)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontproperties=fp, fontsize=10)
    ax.set_xlabel("평균 |SHAP 값| (예측 확률에 대한 영향력)", fontproperties=fp, fontsize=11)
    ax.set_title("피처별 평균 영향력 (SHAP)\n파랑=수준, 주황=인프라추세, 빨강=변화속도",
                 fontproperties=fp, fontsize=12, fontweight="bold")
    for i, (val, bar) in enumerate(zip(mean_abs[order], bars)):
        ax.text(val + mean_abs.max() * 0.01, i, f"{val:.4f}", va="center", fontsize=9)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#3498db", label="수준 피처(level_)"),
        Patch(facecolor="#e74c3c", label="변화속도 피처(delta_)"),
        Patch(facecolor="#f39c12", label="인프라 피처(infra_)"),
    ], prop=fp, fontsize=9, loc="lower right")

    plt.tight_layout()
    fig.savefig(OUT / "shap_bar.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  shap_bar.png 저장")


def plot_shap_beeswarm(shap_vals, X_imp, feat_cols):
    """SHAP Beeswarm — 피처 값과 SHAP 방향의 관계."""
    # 피처 이름을 한글 약칭으로 교체한 후 summary_plot 호출
    shap_exp = shap.Explanation(
        values=shap_vals,
        base_values=np.zeros(len(shap_vals)),
        data=X_imp,
        feature_names=[FEAT_LABELS.get(c, c) for c in feat_cols],
    )
    fig, ax = plt.subplots(figsize=(10, max(5, len(feat_cols) * 0.5)))
    shap.plots.beeswarm(shap_exp, show=False, max_display=len(feat_cols), ax=ax)
    ax.set_title("SHAP Beeswarm — 피처 값(색깔)이 전이 확률에 미치는 방향",
                 fontproperties=fp, fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT / "shap_beeswarm.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  shap_beeswarm.png 저장")


def main():
    print("=" * 60)
    print("SHAP 원인 분해 — analyze_shap.py")
    print("=" * 60)

    print("\n  모델 및 데이터 로딩...")
    gb_pipe, feat_cols, X, meta_df, y = load_model_and_data()

    print("\n  SHAP 값 계산...")
    shap_vals, X_imp, base_val = compute_shap(gb_pipe, feat_cols, X)

    pred_proba = gb_pipe.predict_proba(X)[:, 1]

    print("\n  결과 저장...")
    shap_df  = save_shap_csv(shap_vals, feat_cols, meta_df, pred_proba)
    top3_df  = save_top3_csv(shap_df, feat_cols, meta_df)
    plot_shap_bar(shap_vals, feat_cols)

    try:
        plot_shap_beeswarm(shap_vals, X_imp, feat_cols)
    except Exception as e:
        print(f"  [WARN] beeswarm 스킵: {e}")

    print(f"\n  완료! 산출물:")
    for f in ["shap_values.csv", "shap_top3.csv", "shap_bar.png", "shap_beeswarm.png"]:
        p = OUT / f
        if p.exists():
            print(f"    ✓ {f}  ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
