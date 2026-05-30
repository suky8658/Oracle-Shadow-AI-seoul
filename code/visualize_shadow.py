"""
SHADOW Map — 종합 시각화 스크립트
=================================
1) 발표용 종합판: 06_shadow_story.png
   - 두 역설 근거 + 4사분면 산점도 + 처방 카드
2) 분석 검증팩: 06_shadow_validation.png
   - 축 독립성 + 기준선 민감도 + Q1 안정성

입력:
  - Outputs/shadow_index.csv
  - Outputs/편의의 역설/dependency_index.csv
  - Outputs/복지의 역설/avoidance_index.csv
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import minmax_scale

fp = fm.FontProperties(fname="C:/Windows/Fonts/malgun.ttf")
plt.rcParams["font.family"] = fp.get_name()
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Outputs"

# 색상
C = {
    "Q1": "#e74c3c", "Q2": "#e67e22", "Q3": "#27ae60", "Q4": "#3498db",
    "gray": "#636e72", "light": "#dfe6e9", "bg": "#fafafa",
}

# 사분면 라벨 (표현 수정 반영)
Q_INFO = {
    "Q1": {"name": "의존 + 회피", "sub": "에코-씰 우선배포", "risk": "최위험"},
    "Q2": {"name": "의존 + 복지연계 가능군", "sub": "50플러스/복지관 연계", "risk": ""},
    "Q3": {"name": "상대 안정", "sub": "분기별 모니터링", "risk": ""},
    "Q4": {"name": "회피 우세", "sub": "지역접점 간접접근", "risk": ""},
}


def load_data():
    """dependency_index.csv + avoidance_index.csv에서 shadow_index.csv 재생성."""
    dep = pd.read_csv(OUT / "편의의 역설" / "dependency_index.csv")
    avo = pd.read_csv(OUT / "복지의 역설" / "avoidance_index.csv")

    # Dependency: 행정동 → 자치구 롤업 (평균/표준편차/행정동수)
    dep_gu = dep.groupby("자치구").agg(
        Dependency=("Dependency", "mean"),
        Dependency_std=("Dependency", "std"),
        n_행정동=("Dependency", "count"),
    ).reset_index()
    dep_gu["Dependency"] = minmax_scale(dep_gu["Dependency"]) * 100

    # Avoidance: 자치구 단위 그대로
    avo_sub = avo[["자치구", "Avoidance", "n_가구주", "n_지역사회"]].copy()

    # 통합
    shadow = dep_gu.merge(avo_sub, on="자치구", how="inner")

    # 사분면 분류
    dep_med = shadow["Dependency"].median()
    avo_med = shadow["Avoidance"].median()
    def assign_q(row):
        if row["Dependency"] >= dep_med and row["Avoidance"] >= avo_med:
            return "Q1"
        elif row["Dependency"] >= dep_med and row["Avoidance"] < avo_med:
            return "Q2"
        elif row["Dependency"] < dep_med and row["Avoidance"] < avo_med:
            return "Q3"
        else:
            return "Q4"
    shadow["Quadrant"] = shadow.apply(assign_q, axis=1)
    shadow = shadow.sort_values("자치구").reset_index(drop=True)

    # 저장
    shadow.to_csv(OUT / "shadow_index.csv", index=False, encoding="utf-8-sig")
    print(f"  shadow_index.csv 재생성: {len(shadow)}개 자치구")

    return shadow, dep_med, avo_med


# ══════════════════════════════════════════════════════════════════
# [1] 발표용 종합판
# ══════════════════════════════════════════════════════════════════
def plot_story(shadow, dep_med, avo_med):
    """16:9 발표 슬라이드형. Q1 스포트라이트 중심의 4사분면 맵."""
    print("\n-- [1] 발표용 종합판 --")

    fig = plt.figure(figsize=(22, 12), facecolor="#fbfbfa")
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[0.16, 0.84],
        width_ratios=[2.55, 1.0],
        left=0.045, right=0.965, top=0.90, bottom=0.08,
        hspace=0.08, wspace=0.06,
    )

    # ── 상단 제목 ──
    fig.text(
        0.045, 0.955,
        "편의에 잠기고, 복지를 피하는 지역이 보인다",
        fontsize=25, fontweight="bold", ha="left", va="center", color="#111111",
    )
    fig.text(
        0.045, 0.918,
        "SHADOW Map은 편의 의존도와 복지 회피도를 분리해, 같은 고립 위험에도 다른 처방이 필요한 지역을 찾는다.",
        fontsize=12.5, ha="left", va="center", color=C["gray"],
    )

    # ── 상단 KPI 3개 ──
    kpi_ax = fig.add_subplot(gs[0, :])
    kpi_ax.axis("off")
    kpis = [
        ("편의 인프라 ↔ 고립", "r = +0.643", "p < .001", C["Q1"]),
        ("5060 남성 부정 갭", "+0.323", "8그룹 중 최대", C["Q4"]),
        ("두 축 관계", "rho = 0.095", "p = .653", "#2d3436"),
    ]
    x0s = [0.00, 0.34, 0.68]
    for x0, (label, value, sub, color) in zip(x0s, kpis):
        card = mpatches.FancyBboxPatch(
            (x0, 0.05), 0.30, 0.80,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            facecolor="white", edgecolor="#ececec", linewidth=1.4,
            transform=kpi_ax.transAxes,
        )
        kpi_ax.add_patch(card)
        kpi_ax.text(x0 + 0.035, 0.67, label, transform=kpi_ax.transAxes,
                    fontsize=11, color=C["gray"], va="center")
        kpi_ax.text(x0 + 0.035, 0.39, value, transform=kpi_ax.transAxes,
                    fontsize=24, fontweight="bold", color=color, va="center")
        kpi_ax.text(x0 + 0.205, 0.39, sub, transform=kpi_ax.transAxes,
                    fontsize=10.5, color=C["gray"], va="center")

    # ────────────────────────────────────────────────
    # 중앙: SHADOW 산점도
    # ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])

    # 배경: Q1만 스포트라이트
    ax.set_facecolor("#fbfbfb")
    ax.fill_between([dep_med, 105], avo_med, 105, alpha=0.18, color=C["Q1"], zorder=0)
    ax.fill_between([dep_med, 105], -5, avo_med, alpha=0.035, color=C["Q2"], zorder=0)
    ax.fill_between([-5, dep_med], -5, avo_med, alpha=0.035, color=C["Q3"], zorder=0)
    ax.fill_between([-5, dep_med], avo_med, 105, alpha=0.035, color=C["Q4"], zorder=0)
    for pad, alpha in [(0, 0.22), (-2.5, 0.10)]:
        rect = mpatches.Rectangle(
            (dep_med + pad, avo_med + pad),
            105 - dep_med - pad, 105 - avo_med - pad,
            facecolor="none", edgecolor=C["Q1"], linewidth=2.2,
            alpha=alpha, zorder=1,
        )
        ax.add_patch(rect)

    # 기준선
    ax.axvline(dep_med, color=C["gray"], linestyle="--", linewidth=1.1, alpha=0.45)
    ax.axhline(avo_med, color=C["gray"], linestyle="--", linewidth=1.1, alpha=0.45)

    # 기준선 값 표시
    ax.text(dep_med, -3, f"{dep_med:.0f}", ha="center", fontsize=8, color=C["gray"])
    ax.text(-3, avo_med, f"{avo_med:.0f}", ha="right", va="center", fontsize=8,
            color=C["gray"])

    # 사분면 코너 라벨
    ax.text(101, 101, "Q1\n의존↑ 회피↑", fontsize=16, ha="right", va="top",
            fontweight="bold", color=C["Q1"], alpha=0.95)
    ax.text(101, 2, "Q2  의존↑ / 회피↓", fontsize=10.5, ha="right", va="bottom",
            fontweight="bold", color=C["Q2"], alpha=0.45)
    ax.text(2, 2, "Q3  상대 안정", fontsize=10.5, ha="left", va="bottom",
            fontweight="bold", color=C["Q3"], alpha=0.45)
    ax.text(2, 101, "Q4  회피 우세", fontsize=10.5, ha="left", va="top",
            fontweight="bold", color=C["Q4"], alpha=0.45)

    # 점 그리기
    non_q1 = shadow[shadow["Quadrant"] != "Q1"]
    ax.scatter(non_q1["Dependency"], non_q1["Avoidance"], c="#b9c0c5",
               s=95, edgecolors="white", linewidths=0.6, alpha=0.55, zorder=4)

    for _, row in shadow.iterrows():
        q = row["Quadrant"]
        is_q1 = q == "Q1"
        if is_q1:
            ax.scatter(row["Dependency"], row["Avoidance"], c=C["Q1"], s=520,
                       alpha=0.12, edgecolors="none", zorder=7)
            ax.scatter(row["Dependency"], row["Avoidance"], c=C["Q1"], s=250,
                       edgecolors="#111111", linewidths=1.8, zorder=10)

        # 라벨
        offset_y = 7 if row["Avoidance"] < 92 else -7
        va = "bottom" if offset_y > 0 else "top"
        ax.annotate(row["자치구"], (row["Dependency"], row["Avoidance"]),
                    xytext=(0, offset_y), textcoords="offset points",
                    fontsize=10.5 if is_q1 else 7.4, ha="center", va=va,
                    fontweight="bold" if is_q1 else "normal",
                    color=C["Q1"] if is_q1 else "#8a949b", zorder=11 if is_q1 else 5)

    ax.set_xlabel("Dependency Index (편의 의존도)", fontsize=12, labelpad=10)
    ax.set_ylabel("Avoidance Index (복지 회피도)", fontsize=12, labelpad=10)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_title("두 역설이 겹치는 Q1 지역을 우선 개입한다", fontsize=16, fontweight="bold", pad=12)
    ax.grid(color="#eeeeee", linewidth=0.8, alpha=0.8)

    # ────────────────────────────────────────────────
    # 오른쪽: Q1 우선 처방 패널
    # ────────────────────────────────────────────────
    ax_right = fig.add_subplot(gs[1, 1])
    ax_right.axis("off")
    ax_right.set_xlim(0, 1)
    ax_right.set_ylim(0, 1)

    ax_right.text(0.02, 0.98, "Q1 최우선 처방", fontsize=18, fontweight="bold",
                  color=C["Q1"], ha="left", va="top")
    ax_right.text(0.02, 0.93, "의존도와 회피도가 동시에 높은 지역", fontsize=11,
                  color=C["gray"], ha="left", va="top")

    q1 = shadow[shadow["Quadrant"] == "Q1"].sort_values("Dependency", ascending=False)
    q1_names = "금천구 · 강북구 · 광진구 · 영등포구\n은평구 · 강서구 · 강남구"
    rect = mpatches.FancyBboxPatch(
        (0.02, 0.62), 0.94, 0.25,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor="#fff0ee", edgecolor=C["Q1"], linewidth=2.0,
        transform=ax_right.transAxes,
    )
    ax_right.add_patch(rect)
    ax_right.text(0.06, 0.81, "대상 지역", fontsize=10, color=C["gray"],
                  transform=ax_right.transAxes, ha="left", va="center")
    ax_right.text(0.06, 0.735, q1_names, fontsize=12.5, color="#111111",
                  transform=ax_right.transAxes, ha="left", va="center",
                  fontweight="bold", linespacing=1.35)
    ax_right.text(0.06, 0.665, "에코-씰 우선 배포 + opt-out 우편 + 미배출 추적",
                  fontsize=11, color=C["Q1"], transform=ax_right.transAxes,
                  ha="left", va="center", fontweight="bold")

    ax_right.text(0.02, 0.54, "왜 Q1인가", fontsize=13, fontweight="bold",
                  color="#111111", ha="left", va="top")
    bullets = [
        "편의 인프라가 외출·대면 접촉을 대체",
        "복지 도움 요청은 낮고, 회피 신호는 높음",
        "단일 위험점수로는 Q2·Q4와 구분되지 않음",
    ]
    y = 0.49
    for b in bullets:
        ax_right.text(0.05, y, f"• {b}", fontsize=10.5, color="#2d3436",
                      transform=ax_right.transAxes, ha="left", va="top")
        y -= 0.065

    ax_right.text(0.02, 0.25, "나머지 사분면은 처방이 달라진다", fontsize=12,
                  fontweight="bold", color="#111111", ha="left", va="top")
    mini = [
        ("Q2", "의존↑ / 회피↓", "50플러스·복지관 연계"),
        ("Q4", "의존↓ / 회피↑", "지역 접점 기반 간접 접근"),
        ("Q3", "상대 안정", "분기별 모니터링"),
    ]
    y = 0.19
    for q, name, action in mini:
        ax_right.scatter(0.045, y, s=130, color=C[q], alpha=0.75, transform=ax_right.transAxes)
        ax_right.text(0.08, y + 0.018, f"{q} {name}", fontsize=10, fontweight="bold",
                      color=C[q], transform=ax_right.transAxes, va="center")
        ax_right.text(0.08, y - 0.020, action, fontsize=9, color=C["gray"],
                      transform=ax_right.transAxes, va="center")
        y -= 0.07

    # ── 하단 주석 ──
    fig.text(0.5, 0.02,
             "Dependency: 통신정보+상권 2022-2025, 행정동->자치구 롤업  |  "
             "Avoidance: 서울서베이 가구주 2023-2025 + 지역사회 2022·2024, 자치구 단위  |  "
             "두 축은 거의 같은 지표가 아님: rho=0.095, p=.653  |  기준선: 중앙값",
             ha="center", fontsize=9, color=C["gray"], style="italic")

    path = OUT / "06_shadow_story.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  저장: {path}")


# ══════════════════════════════════════════════════════════════════
# [2] 분석 검증팩
# ══════════════════════════════════════════════════════════════════
def plot_validation(shadow, dep_med, avo_med):
    """부록용: 축 관계 + 기준선 민감도 히트맵 + Q1 점수 배지."""
    print("\n-- [2] 분석 검증팩 --")

    fig = plt.figure(figsize=(22, 8.5), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.1, 1.0],
                          left=0.045, right=0.965, top=0.86, bottom=0.10,
                          wspace=0.16)
    fig.suptitle("SHADOW Map 검증: 두 축은 분리되고, Q1 핵심지역은 유지된다",
                 fontsize=19, fontweight="bold", y=0.96)

    # ── Panel 1: 축 독립성 ──
    ax = fig.add_subplot(gs[0, 0])
    from scipy.stats import spearmanr
    rho, p = spearmanr(shadow["Dependency"], shadow["Avoidance"])

    ax.scatter(shadow["Dependency"], shadow["Avoidance"],
               c=["#bbbbbb" if q != "Q1" else C["Q1"] for q in shadow["Quadrant"]],
               s=[70 if q != "Q1" else 145 for q in shadow["Quadrant"]],
               edgecolors="white", linewidths=0.8, alpha=0.9)
    for _, row in shadow.iterrows():
        ax.annotate(row["자치구"], (row["Dependency"], row["Avoidance"]),
                    fontsize=7.2 if row["Quadrant"] == "Q1" else 6.2,
                    fontweight="bold" if row["Quadrant"] == "Q1" else "normal",
                    color=C["Q1"] if row["Quadrant"] == "Q1" else C["gray"],
                    ha="center", va="bottom", xytext=(0, 4), textcoords="offset points")

    # 회귀선
    z = np.polyfit(shadow["Dependency"], shadow["Avoidance"], 1)
    xline = np.linspace(0, 100, 100)
    ax.plot(xline, np.polyval(z, xline), "k--", alpha=0.3, linewidth=1)

    ax.set_xlabel("Dependency Index", fontsize=11)
    ax.set_ylabel("Avoidance Index", fontsize=11)
    ax.set_title("두 축은 거의 같은 지표가 아님", fontsize=13, fontweight="bold")
    ax.text(0.05, 0.95, f"Spearman rho = {rho:.3f}\np = {p:.3f}",
            transform=ax.transAxes, fontsize=11, va="top",
            bbox=dict(boxstyle="round", facecolor="#f8f9fa", edgecolor=C["gray"]))
    ax.text(0.05, 0.78,
            "강한 단조 관계가 확인되지 않음\n-> 단일 점수보다 4사분면이 유용",
            transform=ax.transAxes, fontsize=9, va="top", color=C["gray"],
            style="italic")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.grid(color="#eeeeee", linewidth=0.8, alpha=0.9)

    # ── Panel 2: 기준선 민감도 히트맵 ──
    ax = fig.add_subplot(gs[0, 1])

    dep_mean = shadow["Dependency"].mean()
    avo_mean = shadow["Avoidance"].mean()
    dep_p60 = shadow["Dependency"].quantile(0.6)
    avo_p60 = shadow["Avoidance"].quantile(0.6)

    criteria = {
        "중앙값 (현재)": (dep_med, avo_med),
        "평균": (dep_mean, avo_mean),
        "상위 40%": (dep_p60, avo_p60),
    }

    def get_q1(dep_cut, avo_cut):
        return set(shadow[(shadow["Dependency"] >= dep_cut) &
                          (shadow["Avoidance"] >= avo_cut)]["자치구"])

    q1_sets = {name: get_q1(dc, ac) for name, (dc, ac) in criteria.items()}
    all_gus = sorted(set().union(*q1_sets.values()))

    # 히트맵식 테이블
    y_pos = np.arange(len(all_gus))
    for xi, (_, q1_set) in enumerate(q1_sets.items()):
        for yi, gu in enumerate(all_gus):
            in_q1 = gu in q1_set
            rect = mpatches.FancyBboxPatch(
                (xi + 0.05, yi - 0.38), 0.90, 0.76,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                facecolor=C["Q1"] if in_q1 else "#edf0f2",
                edgecolor="white", linewidth=1.0,
                alpha=0.92 if in_q1 else 0.85,
            )
            ax.add_patch(rect)
            ax.text(xi + 0.50, yi, "Q1" if in_q1 else "",
                    fontsize=8.5, fontweight="bold", color="white",
                    ha="center", va="center")

    # 축 라벨
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_gus, fontsize=9)
    ax.set_xticks(np.arange(len(criteria)) + 0.5)
    crit_labels = []
    for name, (dc, ac) in criteria.items():
        crit_labels.append(f"{name}\n(D>{dc:.0f}, A>{ac:.0f})")
    ax.set_xticklabels(crit_labels, fontsize=8)
    ax.set_xlim(0, len(criteria))
    ax.set_ylim(-0.5, len(all_gus) - 0.5)
    ax.invert_yaxis()
    ax.set_title("기준선 민감도: Q1 유지 매트릭스", fontsize=13, fontweight="bold")
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 안정/불안정 표시
    stable = set.intersection(*q1_sets.values())
    if stable:
        ax.text(0.98, -0.08,
                f"3기준 모두 Q1: {', '.join(sorted(stable))}",
                transform=ax.transAxes, fontsize=9, ha="right", va="top",
                fontweight="bold", color=C["Q1"])

    # ── Panel 3: Q1 좌표 미니맵 + 점수 배지 ──
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    q1_data = shadow[shadow["Quadrant"] == "Q1"].sort_values("Dependency", ascending=False)

    if len(q1_data) > 0:
        ax.text(0.02, 0.98, "Q1 지역 점수 배지", fontsize=14,
                fontweight="bold", color=C["Q1"], ha="left", va="top")
        ax.text(0.02, 0.935, "D/A = Dependency / Avoidance", fontsize=9.5,
                color=C["gray"], ha="left", va="top")

        # 미니 좌표맵
        map_left, map_bottom, map_w, map_h = 0.07, 0.56, 0.86, 0.30
        ax.add_patch(mpatches.FancyBboxPatch(
            (map_left, map_bottom), map_w, map_h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor="#fff7f5", edgecolor=C["Q1"], linewidth=1.3,
            transform=ax.transAxes,
        ))
        ax.text(map_left + map_w - 0.02, map_bottom + map_h - 0.025,
                "Q1 확대", fontsize=10, fontweight="bold", color=C["Q1"],
                transform=ax.transAxes, ha="right", va="top")
        for _, row in q1_data.iterrows():
            x = map_left + map_w * (row["Dependency"] - dep_med) / max(1, 105 - dep_med)
            y = map_bottom + map_h * (row["Avoidance"] - avo_med) / max(1, 105 - avo_med)
            x = float(np.clip(x, map_left + 0.03, map_left + map_w - 0.03))
            y = float(np.clip(y, map_bottom + 0.04, map_bottom + map_h - 0.04))
            ax.scatter(x, y, s=95, color=C["Q1"], edgecolors="#111111",
                       linewidths=0.8, transform=ax.transAxes, zorder=5)
        ax.text(map_left + 0.02, map_bottom + 0.03,
                "각 점은 Q1 자치구의 상대 위치", fontsize=8.5, color=C["gray"],
                transform=ax.transAxes, ha="left", va="bottom")

        # 배지 그리드
        positions = [(0.04, 0.42), (0.52, 0.42),
                     (0.04, 0.30), (0.52, 0.30),
                     (0.04, 0.18), (0.52, 0.18),
                     (0.04, 0.06)]
        for (x, y), (_, row) in zip(positions, q1_data.iterrows()):
            card = mpatches.FancyBboxPatch(
                (x, y), 0.42, 0.085,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                facecolor="white", edgecolor="#f0c4bd", linewidth=1.0,
                transform=ax.transAxes,
            )
            ax.add_patch(card)
            ax.text(x + 0.03, y + 0.055, row["자치구"], fontsize=10,
                    fontweight="bold", color="#111111", ha="left", va="center",
                    transform=ax.transAxes)
            ax.text(x + 0.29, y + 0.055,
                    f"D {row['Dependency']:.0f}", fontsize=8.5,
                    color=C["Q1"], ha="left", va="center",
                    fontweight="bold", transform=ax.transAxes)
            ax.text(x + 0.29, y + 0.027,
                    f"A {row['Avoidance']:.0f}", fontsize=8.5,
                    color=C["Q4"], ha="left", va="center",
                    fontweight="bold", transform=ax.transAxes)

    plt.tight_layout()
    path = OUT / "06_shadow_validation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  저장: {path}")


# ── main ──
def main():
    print("=" * 60)
    print("SHADOW Map 종합 시각화")
    print("=" * 60)

    shadow, dep_med, avo_med = load_data()
    print(f"  자치구: {len(shadow)}개, 기준선: Dep={dep_med:.1f}, Avo={avo_med:.1f}")

    plot_story(shadow, dep_med, avo_med)
    plot_validation(shadow, dep_med, avo_med)

    print("\n완료!")


if __name__ == "__main__":
    main()
