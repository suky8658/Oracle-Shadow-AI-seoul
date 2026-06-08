"""
섀도우 AI — 처방 점수 시각화
============================
입력: Outputs/shadow_ai/shadow_prescriptions.csv
출력: Outputs/shadow_ai/shadow_prescriptions_chart.png

패널 구성:
  [좌] 처방등급 분포 막대
  [중] 상위 30 행정동 처방 점수 가로 막대 (등급별 색상)
  [우] 전이확률 vs Shadow Score 산점도 (Avoidance 색상)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "Outputs" / "shadow_ai"

# 폰트
_FONT_PATHS = [
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/gulim.ttc",
]
for fp_path in _FONT_PATHS:
    if Path(fp_path).exists():
        fp = fm.FontProperties(fname=fp_path)
        plt.rcParams["font.family"] = fp.get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

GRADE_COLOR = {"S": "#c0392b", "A": "#e67e22", "B": "#2980b9", "C": "#7f8c8d"}
GRADE_LABEL = {"S": "S — 즉시 개입", "A": "A — 고위험", "B": "B — 예방", "C": "C — 정기 안내"}


def load_data() -> pd.DataFrame:
    path = OUT_DIR / "shadow_prescriptions.csv"
    if not path.exists():
        raise FileNotFoundError(f"먼저 run_shadow_ai.py를 실행하세요: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def plot_all(df: pd.DataFrame):
    fig = plt.figure(figsize=(22, 11), facecolor="#fafafa")
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[0.85, 1.5, 1.1],
        left=0.05, right=0.97, top=0.88, bottom=0.08,
        wspace=0.12,
    )

    fig.text(
        0.05, 0.95,
        "섀도우 AI — 행정동별 처방 우선순위",
        fontsize=22, fontweight="bold", ha="left", va="center", color="#111111",
    )
    fig.text(
        0.05, 0.915,
        "전이 확률(가중 75%) + 복지 회피 인덱스(가중 25%)를 결합한 섀도우 처방 점수",
        fontsize=12, ha="left", va="center", color="#636e72",
    )

    _panel_grade_dist(fig.add_subplot(gs[0, 0]), df)
    _panel_top30(fig.add_subplot(gs[0, 1]), df)
    _panel_scatter(fig.add_subplot(gs[0, 2]), df)

    fig.text(
        0.5, 0.025,
        "Shadow Score = 전이확률_정규화 × 0.75 + Avoidance × 0.25  |  "
        "전이확률: GradientBoosting, 2022-2025 행정동 피처  |  "
        "Avoidance: 서울서베이 2023-2025, 자치구 단위",
        ha="center", fontsize=9, color="#636e72", style="italic",
    )

    out_path = OUT_DIR / "shadow_prescriptions_chart.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  저장: {out_path}")


def _panel_grade_dist(ax, df: pd.DataFrame):
    """처방등급별 행정동 수 막대그래프."""
    grades = ["S", "A", "B", "C"]
    counts = [int((df["처방등급"] == g).sum()) for g in grades]
    colors = [GRADE_COLOR[g] for g in grades]

    bars = ax.barh(grades, counts, color=colors, edgecolor="white", linewidth=1.2, height=0.55)

    for bar, cnt in zip(bars, counts):
        ax.text(
            bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
            str(cnt), va="center", ha="left", fontsize=13, fontweight="bold", color="#2d3436",
        )

    ax.set_xlabel("행정동 수", fontsize=11)
    ax.set_title("처방등급 분포", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(0, max(counts) * 1.25)
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=13, length=0)
    ax.set_facecolor("#fafafa")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    legend_patches = [
        mpatches.Patch(color=GRADE_COLOR[g], label=GRADE_LABEL[g]) for g in grades
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right",
              framealpha=0.85, edgecolor="#dddddd")


def _panel_top30(ax, df: pd.DataFrame):
    """상위 30 행정동 Shadow Score 가로 막대."""
    top = df.head(30).iloc[::-1].reset_index(drop=True)  # 위에서 아래로 1위

    labels = top["자치구"] + " " + top["행정동"]
    scores = top["Shadow_Score"]
    colors = [GRADE_COLOR[g] for g in top["처방등급"]]

    y = np.arange(len(top))
    bars = ax.barh(y, scores, color=colors, edgecolor="white", linewidth=0.8, height=0.72)

    for bar, score, grade in zip(bars, scores, top["처방등급"]):
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}",
            va="center", ha="left", fontsize=8, color="#2d3436",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Shadow Score", fontsize=11)
    ax.set_title("상위 30 행정동 처방 점수", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(0, scores.max() * 1.12)
    ax.set_facecolor("#fafafa")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.axvline(80, color=GRADE_COLOR["S"], linestyle="--", linewidth=1.0, alpha=0.5)
    ax.axvline(65, color=GRADE_COLOR["A"], linestyle="--", linewidth=1.0, alpha=0.5)
    ax.text(80.3, 1, "S기준", fontsize=7.5, color=GRADE_COLOR["S"], va="bottom")
    ax.text(65.3, 1, "A기준", fontsize=7.5, color=GRADE_COLOR["A"], va="bottom")


def _panel_scatter(ax, df: pd.DataFrame):
    """전이확률 vs Shadow Score, Avoidance 색상맵 산점도."""
    cmap = cm.get_cmap("YlOrRd")
    norm = mcolors.Normalize(vmin=df["Avoidance"].min(), vmax=df["Avoidance"].max())

    sc = ax.scatter(
        df["전이확률_정규화"], df["Shadow_Score"],
        c=df["Avoidance"], cmap=cmap, norm=norm,
        s=55, edgecolors="#555555", linewidths=0.4, alpha=0.85, zorder=3,
    )

    # S등급 라벨
    s_grade = df[df["처방등급"] == "S"]
    for _, row in s_grade.iterrows():
        ax.annotate(
            row["행정동"],
            (row["전이확률_정규화"], row["Shadow_Score"]),
            xytext=(4, 2), textcoords="offset points",
            fontsize=7, color=GRADE_COLOR["S"], fontweight="bold", zorder=5,
        )

    # 등급 구분선
    ax.axhline(80, color=GRADE_COLOR["S"], linestyle="--", linewidth=1.0, alpha=0.5)
    ax.axhline(65, color=GRADE_COLOR["A"], linestyle="--", linewidth=1.0, alpha=0.4)
    ax.text(101, 80.5, "S", fontsize=9, color=GRADE_COLOR["S"], fontweight="bold", va="bottom")
    ax.text(101, 65.5, "A", fontsize=9, color=GRADE_COLOR["A"], fontweight="bold", va="bottom")

    cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Avoidance (복지 회피도)", fontsize=9)

    ax.set_xlabel("전이확률 정규화 (0~100)", fontsize=11)
    ax.set_ylabel("Shadow Score", fontsize=11)
    ax.set_title("전이확률 × Avoidance → Shadow Score", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(-2, 105)
    ax.set_ylim(0, 102)
    ax.set_facecolor("#fafafa")
    ax.grid(color="#eeeeee", linewidth=0.8, alpha=0.9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def main():
    print("=" * 60)
    print("섀도우 AI — 처방 점수 시각화")
    print("=" * 60)

    df = load_data()
    print(f"  행정동 수: {len(df)}")
    plot_all(df)
    print("\n완료!")


if __name__ == "__main__":
    main()
