"""
전이 위험 예측 스크립트 — predict_risk.py
==========================================
저장된 시제품 모델로 행정동별 Q1 전이 확률을 예측.
단독 실행 가능 (run_transition_pipeline.py + save_model.py 먼저 실행 필요).

사용법:
  python code/predict_risk.py                          # 기존 피처 파일로 예측
  python code/predict_risk.py --input new_features.csv # 새 피처 파일로 예측
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches

_FONT_FILE = "C:/Windows/Fonts/malgun.ttf"
if Path(_FONT_FILE).exists():
    fp = fm.FontProperties(fname=_FONT_FILE)
    plt.rcParams["font.family"] = fp.get_name()
else:
    plt.rcParams["font.family"] = "Malgun Gothic"
    fp = fm.FontProperties()
plt.rcParams["axes.unicode_minus"] = False

ROOT      = Path(__file__).resolve().parent.parent
OUT       = ROOT / "Outputs" / "전이예측"
MODEL_DIR = OUT / "model"


def load_model():
    """저장된 모델 + 메타 로딩."""
    gb_path   = MODEL_DIR / "gb_model.pkl"
    meta_path = MODEL_DIR / "model_meta.json"
    if not gb_path.exists():
        raise FileNotFoundError(f"{gb_path} 없음 — save_model.py 먼저 실행하세요.")
    with open(gb_path, "rb") as f:
        gb = pickle.load(f)
    with open(MODEL_DIR / "lr_model.pkl", "rb") as f:
        lr = pickle.load(f)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return gb, lr, meta


def predict_dong(input_csv: str | Path | None = None, threshold: float = 0.5):
    """
    행정동별 전이 확률 예측.

    Parameters
    ----------
    input_csv : 피처 CSV 경로. None이면 Outputs/전이예측/transition_features.csv 사용.
    threshold : 이진 분류 임계값 (기본 0.5).

    Returns
    -------
    pred_df : 행정동별 예측 결과 DataFrame
    """
    gb, lr, meta = load_model()
    feat_cols = meta["feat_cols"]
    meta_cols = meta["meta_cols"]

    # 데이터 로딩
    csv_path = Path(input_csv) if input_csv else OUT / "transition_features.csv"
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # 피처 컬럼 확인
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        print(f"  [WARN] 누락 피처 {len(missing)}개: {missing[:5]}{'...' if len(missing)>5 else ''}")
        for c in missing:
            df[c] = np.nan

    X = df[feat_cols].copy()
    avail_meta = [c for c in meta_cols if c in df.columns]

    # 예측
    gb_prob = gb.predict_proba(X)[:, 1]
    lr_prob = lr.predict_proba(X)[:, 1]

    result = df[avail_meta].copy() if avail_meta else pd.DataFrame()
    result["전이확률_GB"] = gb_prob.round(4)
    result["전이확률_LR"] = lr_prob.round(4)
    result["전이예측_GB"] = (gb_prob >= threshold).astype(int)
    result["위험등급"] = pd.cut(
        result["전이확률_GB"],
        bins=[-0.001, 0.30, 0.50, 0.70, 1.001],
        labels=["저위험", "중위험", "고위험", "최고위험"],
    )

    # dep 컬럼 추가 (있으면)
    dep_cols = [c for c in df.columns if c.startswith("dep_")]
    if dep_cols:
        result = pd.concat([result, df[dep_cols]], axis=1)

    result = result.sort_values("전이확률_GB", ascending=False).reset_index(drop=True)
    return result


def make_risk_map(pred_df: pd.DataFrame, save_path: Path):
    """위험등급별 행정동 분포 시각화."""
    grade_order = ["최고위험", "고위험", "중위험", "저위험"]
    grade_colors = {"최고위험": "#e74c3c", "고위험": "#e67e22",
                    "중위험": "#f1c40f", "저위험": "#27ae60"}

    # 상위 30개 행정동 시각화
    top30 = pred_df.head(30).copy()
    top30 = top30.iloc[::-1].reset_index(drop=True)  # 역순(위에서 아래로 내림차순)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    # 왼쪽: 상위 30개 행정동 확률 바 차트
    ax = axes[0]
    colors = [grade_colors.get(str(g), "#636e72") for g in top30["위험등급"]]
    dong_labels = top30.apply(
        lambda r: f"{r.get('자치구','')} {r.get('행정동',r.get('행정동코드',''))}", axis=1)
    bars = ax.barh(range(len(top30)), top30["전이확률_GB"], color=colors,
                   edgecolor="white", height=0.7)
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(dong_labels, fontproperties=fp, fontsize=8)
    ax.set_xlabel("Q1 전이 확률 (GradientBoosting)", fontproperties=fp, fontsize=11)
    ax.set_title("Q1 전이 위험 상위 30개 행정동", fontproperties=fp,
                 fontsize=13, fontweight="bold")
    ax.axvline(0.5, color="#636e72", ls="--", alpha=0.5, label="임계값 0.5")
    ax.axvline(0.7, color="#e74c3c", ls=":", alpha=0.5, label="최고위험 0.7")
    ax.legend(prop=fp, fontsize=9)
    for i, (val, _) in enumerate(zip(top30["전이확률_GB"], bars)):
        ax.text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=7.5)

    # 오른쪽: 위험등급 파이차트 + 분포
    ax2 = axes[1]
    grade_counts = pred_df["위험등급"].value_counts()
    # 순서 맞추기
    grade_counts = grade_counts.reindex([g for g in grade_order if g in grade_counts.index])
    colors_pie = [grade_colors[g] for g in grade_counts.index]
    wedges, texts, autotexts = ax2.pie(
        grade_counts.values, labels=None,
        colors=colors_pie, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
    ax2.legend(
        handles=[mpatches.Patch(facecolor=grade_colors[g], label=f"{g} ({grade_counts.get(g,0)}개)")
                 for g in grade_order if g in grade_counts.index],
        prop=fp, fontsize=10, loc="lower center",
        bbox_to_anchor=(0.5, -0.08), ncol=2,
    )
    ax2.set_title(f"전체 {len(pred_df)}개 행정동 위험등급 분포",
                  fontproperties=fp, fontsize=13, fontweight="bold")

    fig.suptitle("서울 5060 남성 1인가구 Q1 전이 위험 예측 결과",
                 fontproperties=fp, fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  지도 저장: {save_path}")


def print_summary(pred_df: pd.DataFrame, meta: dict):
    """예측 결과 요약 출력."""
    grade_order = ["최고위험", "고위험", "중위험", "저위험"]
    total = len(pred_df)

    print("\n" + "=" * 55)
    print("  Q1 전이 위험 예측 결과 요약")
    print("=" * 55)
    print(f"  분석 행정동 수: {total}개")
    print(f"  모델 학습일: {meta.get('trained_at','N/A')}")
    print(f"  GradientBoosting CV AUC: {meta['cv_metrics']['GradientBoosting']['auc']:.3f}")
    print()

    print("  위험등급 분포:")
    grade_counts = pred_df["위험등급"].value_counts()
    for g in grade_order:
        cnt = grade_counts.get(g, 0)
        pct = cnt / total * 100
        bar = "█" * int(pct / 2)
        print(f"    {g:6s}: {cnt:3d}개 ({pct:5.1f}%) {bar}")

    print()
    print("  최고위험 행정동 (상위 10개):")
    top10 = pred_df[pred_df["위험등급"] == "최고위험"].head(10)
    if len(top10) == 0:
        top10 = pred_df.head(10)
        print("  (최고위험 없음 — 전체 상위 10개)")
    for i, (_, r) in enumerate(top10.iterrows()):
        gu   = r.get("자치구", "")
        dong = r.get("행정동", str(r.get("행정동코드","")))
        dep_s = r.get("dep_slope_annual", float("nan"))
        dep_s_str = f"slope={dep_s:.3f}" if pd.notna(dep_s) else ""
        print(f"    {i+1:2d}. {gu:5s} {dong:10s}  확률={r['전이확률_GB']:.3f}  {dep_s_str}")

    print()
    print("  피처 중요도 상위 5개:")
    imp = meta.get("feat_importance", {})
    for i, (k, v) in enumerate(list(imp.items())[:5]):
        print(f"    {i+1}. {k}: {v:.4f}")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="Q1 전이 위험 예측")
    parser.add_argument("--input", default=None, help="입력 피처 CSV 경로")
    parser.add_argument("--threshold", type=float, default=0.5, help="분류 임계값 (기본 0.5)")
    parser.add_argument("--output", default=None, help="출력 CSV 경로")
    args = parser.parse_args()

    print("=" * 60)
    print("Q1 전이 위험 예측 — predict_risk.py")
    print("=" * 60)

    # 모델 로딩
    print("\n  모델 로딩...")
    gb, lr, meta = load_model()
    print(f"  학습일: {meta.get('trained_at','N/A')}")
    print(f"  피처: {len(meta['feat_cols'])}개")

    # 예측
    print("\n  예측 실행...")
    pred_df = predict_dong(args.input, args.threshold)

    # 출력 저장
    out_path = Path(args.output) if args.output else OUT / "risk_predictions_final.csv"
    pred_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  예측 결과 저장: {out_path} ({len(pred_df)}행)")

    # 시각화
    map_path = OUT / "risk_map_final.png"
    make_risk_map(pred_df, map_path)

    # 요약 출력
    print_summary(pred_df, meta)

    print(f"\n  완료! 결과 파일:")
    print(f"    {out_path}")
    print(f"    {map_path}")


if __name__ == "__main__":
    main()
