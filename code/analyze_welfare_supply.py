"""
복지시설 공급 레이어 검증
========================
복지시설 공급량이 Avoidance Index를 낮추는 보호 효과가 있는지 검증.

분석 구조:
  1. 메인 검증: 시설군별 밀도 vs Avoidance 상관
  2. 보조 검증: 50플러스시설 유무 vs Avoidance
  3. 사분면 검증: Q1이 Q3보다 시설 부족한가?
  4. 파생지표: Disconnect Score

산출물:
  - Outputs/복지의 역설/복지공급검증/welfare_supply_analysis.md
  - Outputs/복지의 역설/복지공급검증/welfare_supply_scatter.png
  - Outputs/복지의 역설/복지공급검증/welfare_supply_quadrant.png
  - Outputs/복지의 역설/복지공급검증/welfare_supply_disconnect_map.png
  - Outputs/복지의 역설/복지공급검증/welfare_disconnect.png
"""

import pandas as pd
import numpy as np
import glob
from pathlib import Path
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.preprocessing import minmax_scale
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sys

sys.stdout.reconfigure(encoding="utf-8")

fp = fm.FontProperties(fname="C:/Windows/Fonts/malgun.ttf")
plt.rcParams["font.family"] = fp.get_name()
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data"
OUT = ROOT / "Outputs"

C_RED = "#e74c3c"
C_BLUE = "#3498db"
C_GREEN = "#27ae60"
C_ORANGE = "#e67e22"
C_GRAY = "#636e72"
C_PURPLE = "#8e44ad"
Q_COLORS = {"Q1": C_RED, "Q2": C_ORANGE, "Q3": C_GREEN, "Q4": C_BLUE}


# ── [1] 데이터 로딩 ──────────────────────────────────────────────
def load_facilities():
    """자치구별 시설군 분류."""
    frames = []
    for f in glob.glob(str(DATA / "서울시 사회복지시설 목록" / "서울시_*구_사회복지시설_목록.csv")):
        try:
            df = pd.read_csv(f, encoding="cp949")
        except UnicodeDecodeError:
            df = pd.read_csv(f, encoding="utf-8-sig")
        frames.append(df)
    all_fac = pd.concat(frames, ignore_index=True)

    type_col = "시설종류명(시설유형)"

    # 시설군 분류
    def classify(row):
        t = str(row[type_col])
        if "사회복지관" in t:
            return "사회복지관"
        if "노인복지관" in t:
            return "노인복지관"
        if "자활" in t:
            return "자활센터"
        if "정신" in t or "상담" in t:
            return "정신건강/상담"
        if "노인" in t:
            return "노인복지시설"
        return "기타"

    all_fac["시설군"] = all_fac.apply(classify, axis=1)

    # 자치구별 시설군 수 피벗
    gu_fac = all_fac.groupby(["시군구명", "시설군"]).size().unstack(fill_value=0)
    gu_fac["전체"] = all_fac.groupby("시군구명").size()
    gu_fac["복지관합산"] = gu_fac.get("사회복지관", 0) + gu_fac.get("노인복지관", 0)
    gu_fac = gu_fac.reset_index().rename(columns={"시군구명": "자치구"})

    # 50플러스
    plus50 = pd.read_csv(DATA / "서울시 사회복지시설 목록" / "서울시_50플러스시설_목록.csv",
                         encoding="utf-8-sig")
    gu_50 = plus50.groupby("자치구").size().reset_index(name="50플러스")
    gu_fac = gu_fac.merge(gu_50, on="자치구", how="left")
    gu_fac["50플러스"] = gu_fac["50플러스"].fillna(0).astype(int)

    print(f"  복지시설: {len(all_fac)}개, 자치구: {len(gu_fac)}개")
    return gu_fac


def load_indices():
    """Avoidance + Shadow 인덱스 + 5060남성 실제 인구."""
    shadow = pd.read_csv(OUT / "shadow_index.csv")

    # 통신정보 패널에서 5060남성 실제 인구/1인가구 산출
    panel_cache = OUT / "편의의 역설" / "_full_panel_cache.pkl"
    if panel_cache.exists():
        panel = pd.read_pickle(panel_cache)
        m5060 = panel[panel["그룹"] == "남 50-60대"]
        gu_pop = m5060.groupby("자치구").agg(
            pop_raw=("pop", "sum"),
            solo_raw=("solo", "sum"),
        ).reset_index()
        # 4년(48개월) 합산 -> 월평균
        gu_pop["pop_5060"] = gu_pop["pop_raw"] / 48
        gu_pop["solo_5060"] = gu_pop["solo_raw"] / 48
        shadow = shadow.merge(gu_pop[["자치구", "pop_5060", "solo_5060"]], on="자치구", how="left")
        print(f"  5060남성 인구: {shadow['pop_5060'].sum():,.0f}명, 1인가구: {shadow['solo_5060'].sum():,.0f}명")
    else:
        print("  [경고] 패널 캐시 없음 — n_가구주로 대체")

    return shadow


# ── [2] 메인 검증: 시설 밀도 vs Avoidance ────────────────────────
def analyze_main(fac, idx):
    """시설군별 밀도 vs Avoidance Spearman 상관."""
    print("\n== [1] 메인 검증: 시설 밀도 vs Avoidance ==")

    merged = idx.merge(fac, on="자치구", how="inner")

    # 밀도 = 시설수 / 5060 남성 1인가구 수 (통신정보 기반 실제 인구)
    denom_col = "solo_5060" if "solo_5060" in merged.columns else "n_가구주"
    denom_label = "5060남성 1인가구" if denom_col == "solo_5060" else "서베이 표본"
    print(f"  밀도 분모: {denom_label} ({denom_col})")

    facility_groups = {
        "전체 복지시설": "전체",
        "노인복지시설": "노인복지시설",
        "복지관 합산 (사회+노인)": "복지관합산",
        "자활센터": "자활센터",
        "정신건강/상담": "정신건강/상담",
    }

    results = []
    for label, col in facility_groups.items():
        if col not in merged.columns:
            continue
        merged[f"{col}_밀도"] = merged[col] / merged[denom_col] * 10000  # 1만명당
        rho, p = spearmanr(merged[f"{col}_밀도"], merged["Avoidance"])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        results.append({"시설군": label, "rho": rho, "p": p, "sig": sig})
        print(f"  {label:25s}  rho={rho:+.3f} ({sig})")

    return pd.DataFrame(results), merged


# ── [3] 보조 검증: 50플러스 유무 vs Avoidance ────────────────────
def analyze_50plus(merged):
    """50플러스 시설 유무에 따른 Avoidance 비교."""
    print("\n== [2] 보조 검증: 50플러스 유무 vs Avoidance ==")

    has = merged[merged["50플러스"] > 0]["Avoidance"]
    no = merged[merged["50플러스"] == 0]["Avoidance"]

    print(f"  50플러스 있음: {len(has)}개 구, Avoidance 평균={has.mean():.1f}")
    print(f"  50플러스 없음: {len(no)}개 구, Avoidance 평균={no.mean():.1f}")

    if len(has) >= 3 and len(no) >= 3:
        u, p = mannwhitneyu(has, no, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  Mann-Whitney p={p:.3f} ({sig})")
        return {"has_mean": has.mean(), "no_mean": no.mean(), "p": p, "sig": sig,
                "n_has": len(has), "n_no": len(no)}
    return None


# ── [4] 사분면 검증: Q1 vs Q3 시설 공급 ──────────────────────────
def analyze_quadrant(merged):
    """Q1과 Q3의 시설 밀도 비교."""
    print("\n== [3] 사분면 검증: Q1 vs Q3 시설 공급 ==")

    q1 = merged[merged["Quadrant"] == "Q1"]
    q3 = merged[merged["Quadrant"] == "Q3"]

    denom_col = "solo_5060" if "solo_5060" in merged.columns else "n_가구주"
    cols = ["전체", "노인복지시설", "복지관합산", "자활센터"]
    results = []
    for col in cols:
        if col not in merged.columns:
            continue
        q1_density = (q1[col] / q1[denom_col] * 10000).mean()  # 1만명당
        q3_density = (q3[col] / q3[denom_col] * 10000).mean()
        ratio = q1_density / q3_density if q3_density > 0 else float("inf")
        results.append({"시설군": col, "Q1_밀도": q1_density, "Q3_밀도": q3_density, "Q1/Q3": ratio})
        direction = "Q1이 더 많음" if ratio > 1 else "Q1이 부족"
        print(f"  {col:15s}  Q1={q1_density:.2f}  Q3={q3_density:.2f}  비율={ratio:.2f} ({direction})")

    return pd.DataFrame(results)


# ── [5] Disconnect Score ─────────────────────────────────────────
def calculate_disconnect(merged):
    """Disconnect Score = 공급 밀도(정규화) x Avoidance(정규화)."""
    print("\n== [4] Disconnect Score ==")

    denom_col = "solo_5060" if "solo_5060" in merged.columns else "n_가구주"
    merged["공급밀도_norm"] = minmax_scale(merged["전체"] / merged[denom_col])
    merged["Avoidance_norm"] = minmax_scale(merged["Avoidance"])
    merged["Disconnect"] = merged["공급밀도_norm"] * merged["Avoidance_norm"]
    merged["Disconnect_100"] = (minmax_scale(merged["Disconnect"]) * 100).round(1)

    top5 = merged.nlargest(5, "Disconnect_100")
    print("  연결 실패 의심 상위 5개 자치구:")
    for _, row in top5.iterrows():
        print(f"    {row['자치구']:5s}  Disconnect={row['Disconnect_100']:.0f}  "
              f"시설={row['전체']}개  Avoidance={row['Avoidance']:.1f}  Q={row['Quadrant']}")

    return merged


# ── [6] 시각화 ───────────────────────────────────────────────────
def plot_scatter(merged, corr_df):
    """시설 밀도 vs Avoidance 산점도."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    plots = [
        ("전체_밀도", "전체 복지시설 밀도", C_BLUE),
        ("복지관합산_밀도", "복지관 (사회+노인) 밀도", C_GREEN),
        ("노인복지시설_밀도", "노인복지시설 밀도", C_ORANGE),
    ]

    for ax, (col, title, color) in zip(axes, plots):
        if col not in merged.columns:
            continue
        for _, row in merged.iterrows():
            q = row["Quadrant"]
            c = C_RED if q == "Q1" else color
            size = 150 if q == "Q1" else 80
            ax.scatter(row[col], row["Avoidance"], c=c, s=size,
                       edgecolors="white", linewidths=0.5, zorder=5 if q == "Q1" else 3)
            ax.annotate(row["자치구"], (row[col], row["Avoidance"]),
                        fontsize=7, ha="center", va="bottom",
                        xytext=(0, 4), textcoords="offset points",
                        fontweight="bold" if q == "Q1" else "normal",
                        color=C_RED if q == "Q1" else C_GRAY)

        rho_row = corr_df[corr_df["시설군"].str.contains(title.split("(")[0].strip().replace("밀도", "").strip())]
        if len(rho_row) > 0:
            rho_val = rho_row.iloc[0]["rho"]
            sig_val = rho_row.iloc[0]["sig"]
        else:
            rho_val, sig_val = spearmanr(merged[col], merged["Avoidance"])
            sig_val = "ns"

        ax.text(0.05, 0.95, f"Spearman rho = {rho_val:+.3f} ({sig_val})",
                transform=ax.transAxes, fontsize=11, va="top",
                bbox=dict(boxstyle="round", facecolor="#f8f9fa", edgecolor=C_GRAY))

        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel("Avoidance Index", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")

    fig.suptitle("복지시설 공급량 vs 복지 회피도: 보호 효과가 있는가?",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = OUT / "복지의 역설" / "복지공급검증" / "welfare_supply_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n  저장: {path}")


def plot_quadrant_comparison(merged, quad_df):
    """Q1 vs Q3 시설 공급량 비교. 막대 대신 덤벨/산점도로 표현."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 왼쪽: Q1 vs Q3 밀도 덤벨 플롯
    ax = axes[0]
    cols = quad_df["시설군"].values
    q1_vals = quad_df["Q1_밀도"].values
    q3_vals = quad_df["Q3_밀도"].values
    y = np.arange(len(cols))
    for i, (name, v1, v3) in enumerate(zip(cols, q1_vals, q3_vals)):
        ax.plot([v1, v3], [i, i], color="#cfd6dc", linewidth=4, alpha=0.9, zorder=1)
        ax.scatter(v1, i, s=170, color=C_RED, edgecolors="white", linewidths=1.5,
                   label="Q1 (의존+회피)" if i == 0 else "", zorder=3)
        ax.scatter(v3, i, s=170, color=C_GREEN, edgecolors="white", linewidths=1.5,
                   label="Q3 (상대 안정)" if i == 0 else "", zorder=3)
        ax.text(v1 - max(q3_vals) * 0.02, i, f"{v1:.1f}", ha="right", va="center",
                fontsize=9, color=C_RED, fontweight="bold")
        ax.text(v3 + max(q3_vals) * 0.02, i, f"{v3:.1f}", ha="left", va="center",
                fontsize=9, color=C_GREEN, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(cols, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("시설 밀도 (5060남성 1인가구 1만명당)", fontsize=11)
    ax.set_title("Q1에도 공급은 존재하지만, Q3보다 높지는 않다", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="x", color="#eeeeee", linewidth=0.8)
    ax.set_xlim(-max(q3_vals) * 0.06, max(q3_vals) * 1.14)

    # 오른쪽: 자치구별 시설수 vs Avoidance (사분면 색)
    ax = axes[1]
    for _, row in merged.iterrows():
        c = Q_COLORS.get(row["Quadrant"], C_GRAY)
        ax.scatter(row["전체"], row["Avoidance"], c=c, s=120,
                   edgecolors="white", linewidths=0.5)
        ax.annotate(row["자치구"], (row["전체"], row["Avoidance"]),
                    fontsize=7.5, ha="center", va="bottom",
                    xytext=(0, 4), textcoords="offset points")

    ax.set_xlabel("전체 복지시설 수", fontsize=11)
    ax.set_ylabel("Avoidance Index", fontsize=11)
    ax.set_title("시설 수(절대량) vs Avoidance", fontsize=13, fontweight="bold")

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                              markersize=10, label=f"{q}") for q, c in Q_COLORS.items()]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    plt.tight_layout()
    path = OUT / "복지의 역설" / "복지공급검증" / "welfare_supply_quadrant.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  저장: {path}")


def plot_disconnect_bubble(merged, corr_df):
    """복지시설 공급 밀도와 회피도의 연결 실패를 버블 차트로 표현."""
    fig, ax = plt.subplots(figsize=(15, 9), facecolor="white")

    x = merged["전체_밀도"]
    y = merged["Avoidance"]
    x_cut = x.median()
    y_cut = y.median()
    x_min, x_max = x.min() * 0.95, x.max() * 1.05

    # 오른쪽 위 연결 실패 의심 구역
    ax.axvspan(x_cut, x_max, ymin=(y_cut + 5) / 110, ymax=1,
               color=C_RED, alpha=0.07, zorder=0)
    ax.axvline(x_cut, color=C_GRAY, linestyle="--", linewidth=1, alpha=0.35)
    ax.axhline(y_cut, color=C_GRAY, linestyle="--", linewidth=1, alpha=0.35)
    ax.text(x_max * 0.985, 102, "연결 실패 의심 구역",
            ha="right", va="top", fontsize=14, fontweight="bold", color=C_RED)
    ax.text(x_max * 0.985, 97, "시설 공급↑ + 회피도↑",
            ha="right", va="top", fontsize=10, color=C_GRAY)

    # 기대 방향 화살표
    ax.annotate("공급이 보호 효과를 낸다면\n점들은 아래로 내려가야 함",
                xy=(x_cut + (x_max - x_cut) * 0.55, y_cut - 6),
                xytext=(x_cut + (x_max - x_cut) * 0.55, y_cut + 28),
                arrowprops=dict(arrowstyle="->", color=C_GRAY, linewidth=1.8),
                fontsize=10, color=C_GRAY, ha="center", va="center")

    dep_size = 120 + merged["Dependency"] * 5.0
    top_disconnect = set(merged.nlargest(5, "Disconnect_100")["자치구"])
    for i, row in merged.iterrows():
        q = row["Quadrant"]
        is_top = row["자치구"] in top_disconnect
        ax.scatter(row["전체_밀도"], row["Avoidance"],
                   s=dep_size.loc[i], color=Q_COLORS.get(q, C_GRAY),
                   alpha=0.82 if is_top else 0.55,
                   edgecolors="#111111" if is_top else "white",
                   linewidths=1.7 if is_top else 0.8,
                   zorder=5 if is_top else 3)
        if is_top or q == "Q1":
            ax.annotate(row["자치구"], (row["전체_밀도"], row["Avoidance"]),
                        xytext=(0, 8), textcoords="offset points",
                        fontsize=10 if is_top else 8.5,
                        ha="center", va="bottom",
                        fontweight="bold" if is_top else "normal",
                        color=Q_COLORS.get(q, C_GRAY))

    total_row = corr_df[corr_df["시설군"] == "전체 복지시설"].iloc[0]
    ax.text(0.035, 0.95,
            f"시설 밀도 ↔ Avoidance\nSpearman rho = {total_row['rho']:+.3f} ({total_row['sig']})",
            transform=ax.transAxes, fontsize=12, va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=C_GRAY))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-5, 108)
    ax.set_xlabel("전체 복지시설 밀도 (5060 남성 1인가구 1만 명당)", fontsize=12)
    ax.set_ylabel("Avoidance Index (복지 회피도)", fontsize=12)
    ax.set_title("시설이 많아도 회피가 낮아지지 않는다",
                 fontsize=18, fontweight="bold", pad=14)
    ax.grid(color="#eeeeee", linewidth=0.8)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
               markeredgecolor="white", markersize=11, label=q)
        for q, c in Q_COLORS.items()
    ]
    size_legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#aaaaaa",
               markersize=8, label="Dependency 낮음"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#aaaaaa",
               markersize=14, label="Dependency 높음"),
    ]
    ax.legend(handles=legend_elements + size_legend, loc="lower right",
              fontsize=9, frameon=True)

    path = OUT / "복지의 역설" / "복지공급검증" / "welfare_supply_disconnect_map.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  저장: {path}")


def plot_disconnect(merged):
    """Disconnect Score 순위 시각화. 막대 대신 점 순위로 표현."""
    fig, ax = plt.subplots(figsize=(12, 7))

    merged_sorted = merged.sort_values("Disconnect_100", ascending=True)
    colors = [Q_COLORS.get(row["Quadrant"], C_GRAY) for _, row in merged_sorted.iterrows()]
    y = np.arange(len(merged_sorted))
    ax.hlines(y, 0, merged_sorted["Disconnect_100"], color="#dfe6e9",
              linewidth=2.2, zorder=1)
    ax.scatter(merged_sorted["Disconnect_100"], y, color=colors, s=120,
               edgecolors="white", linewidths=1.0, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(merged_sorted["자치구"], fontsize=10)
    ax.set_xlabel("Disconnect Score (0~100)", fontsize=12)
    ax.set_title("연결 실패 의심 우선지역\n(시설 공급 밀도 + Avoidance가 함께 높은 곳)",
                 fontsize=14, fontweight="bold")

    for i, (_, row) in enumerate(merged_sorted.iterrows()):
        ax.text(row["Disconnect_100"] + 1, i,
                f"{row['Disconnect_100']:.0f} ({row['Quadrant']})",
                va="center", fontsize=9,
                fontweight="bold" if row["Quadrant"] == "Q1" else "normal")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=C_RED, label="Q1 의존+회피"),
                       Patch(facecolor=C_ORANGE, label="Q2 의존+연계가능군"),
                       Patch(facecolor=C_GREEN, label="Q3 상대안정"),
                       Patch(facecolor=C_BLUE, label="Q4 회피우세")]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    plt.tight_layout()
    path = OUT / "복지의 역설" / "복지공급검증" / "welfare_disconnect.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  저장: {path}")


# ── [7] 리포트 저장 ──────────────────────────────────────────────
def save_report(corr_df, plus50_result, quad_df, merged):
    corr_lines = "\n".join(
        f"| {row['시설군']} | {row['rho']:+.3f} | {row['sig']} |"
        for _, row in corr_df.iterrows()
    )
    quad_lines = "\n".join(
        f"| {row['시설군']} | {row['Q1_밀도']:.2f} | {row['Q3_밀도']:.2f} | {row['Q1/Q3']:.2f} |"
        for _, row in quad_df.iterrows()
    )

    top5 = merged.nlargest(5, "Disconnect_100")
    disc_lines = "\n".join(
        f"| {row['자치구']} | {row['Disconnect_100']:.0f} | {row['전체']} | {row['Avoidance']:.1f} | {row['Quadrant']} |"
        for _, row in top5.iterrows()
    )

    p50_text = ""
    if plus50_result:
        p50_text = f"""
50플러스 시설 있음: {plus50_result['n_has']}개 구, Avoidance 평균 {plus50_result['has_mean']:.1f}
50플러스 시설 없음: {plus50_result['n_no']}개 구, Avoidance 평균 {plus50_result['no_mean']:.1f}
Mann-Whitney p = {plus50_result['p']:.3f} ({plus50_result['sig']})
"""

    text = f"""\
# 복지시설 공급 레이어 검증 결과

## 핵심 결론

복지시설 공급량이 많은 자치구일수록 Avoidance Index가 낮아지는 보호 효과는 확인되지 않았다.
이는 복지의 역설이 단순한 시설 부족 문제가 아니라, 시설과 당사자가 연결되지 않는 문제일 가능성을 시사한다.

## 1. 시설 밀도 vs Avoidance 상관

| 시설군 | Spearman rho | 유의성 |
|--------|-------------|--------|
{corr_lines}

밀도 = 5060 남성 1인가구 1만 명당 시설 수 (통신정보 기반 실제 인구).
어떤 시설군도 Avoidance와 유의한 음의 상관을 보이지 않는다.
시설이 많다고 회피가 줄지 않는다.

## 2. 50플러스 시설 유무 vs Avoidance (보조)
{p50_text}
50플러스 시설 존재만으로 회피도가 달라지지 않는다. (N이 작아 보조 근거로만 해석)

## 3. Q1 vs Q3 시설 공급량

| 시설군 | Q1 밀도 | Q3 밀도 | Q1/Q3 |
|--------|---------|---------|-------|
{quad_lines}

Q1/Q3 비율이 1보다 낮아 Q1의 시설 밀도는 Q3보다 다소 낮다.
다만 Q1에도 일정 수준의 시설 공급은 존재하며, 전체 상관분석에서 공급량이 높을수록 회피가 낮아지는 보호 효과는 확인되지 않았다.
따라서 문제는 단순한 시설 부재라기보다 시설과 당사자가 연결되는 방식에 가깝다.

## 4. Disconnect Score 상위 5개

연결 실패 의심 우선지역 = 시설 공급 밀도와 Avoidance가 함께 높은 곳

| 자치구 | Disconnect | 시설수 | Avoidance | 사분면 |
|--------|-----------|--------|-----------|--------|
{disc_lines}

## 해석

- 공급만으로는 연결을 보장하지 않는다
- 시설 공급량만으로 설명되지 않는 Avoidance 축을 별도로 볼 필요가 있다
- Q1 지역에 대한 처방은 시설을 단순히 늘리는 것보다 접근 방식을 바꾸는 데 초점을 둬야 한다
"""
    path = OUT / "복지의 역설" / "복지공급검증" / "welfare_supply_analysis.md"
    path.write_text(text, encoding="utf-8")
    print(f"  저장: {path}")


# ── main ──────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("복지시설 공급 레이어 검증")
    print("=" * 60)

    print("\n-- 데이터 로딩 --")
    fac = load_facilities()
    idx = load_indices()

    corr_df, merged = analyze_main(fac, idx)
    plus50_result = analyze_50plus(merged)
    quad_df = analyze_quadrant(merged)
    merged = calculate_disconnect(merged)

    print("\n-- 시각화 --")
    plot_scatter(merged, corr_df)
    plot_quadrant_comparison(merged, quad_df)
    plot_disconnect_bubble(merged, corr_df)
    plot_disconnect(merged)

    print("\n-- 리포트 --")
    save_report(corr_df, plus50_result, quad_df, merged)

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
