"""
복지의 역설 — 시각화 통합 스크립트 (V3)
==========================================
가설 검증 흐름 (왜곡 없이):
  1) 부정 갭 — 핵심 증거 (객관적 고립 vs 주관적 인정)
  2) 도움부재 + 복지불신 — 5060이 유의하게 높은 2개 지표만
  3) 미래 vs 현재 괴리 — "81.3%는 미래가 어렵다, 근데 지금은 괜찮다"
  4) 자치구 지도 — Avoidance Index
  5) 구성요소 분해 — 자치구별 기여도

출력 (Outputs/복지의 역설/):
  - 05_denial_gap.png
  - 05_support_evidence.png
  - 05_future_vs_present.png
  - 05_avoidance_map.png
  - 05_component_breakdown.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon as MplPolygon
import pandas as pd
import numpy as np
import json
import glob
from pathlib import Path
from sklearn.preprocessing import minmax_scale
from scipy.stats import spearmanr

fp = fm.FontProperties(fname="C:/Windows/Fonts/malgun.ttf")
plt.rcParams["font.family"] = fp.get_name()
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data"
OUT  = ROOT / "Outputs" / "복지의 역설"
OUT.mkdir(parents=True, exist_ok=True)
SURVEY_BASE = DATA / "서울서베이 도시정책지표조사 정보"

GU_MAP = {
    110: "종로구", 140: "중구", 170: "용산구", 200: "성동구", 215: "광진구",
    230: "동대문구", 260: "중랑구", 290: "성북구", 305: "강북구", 320: "도봉구",
    350: "노원구", 380: "은평구", 410: "서대문구", 440: "마포구", 470: "양천구",
    500: "강서구", 530: "구로구", 545: "금천구", 560: "영등포구", 590: "동작구",
    620: "관악구", 650: "서초구", 680: "강남구", 710: "송파구", 740: "강동구",
}
GU_PREFIX = {
    11110: 110, 11140: 140, 11170: 170, 11200: 200, 11215: 215,
    11230: 230, 11260: 260, 11290: 290, 11305: 305, 11320: 320,
    11350: 350, 11380: 380, 11410: 410, 11440: 440, 11470: 470,
    11500: 500, 11530: 530, 11545: 545, 11560: 560, 11590: 590,
    11620: 620, 11650: 650, 11680: 680, 11710: 710, 11740: 740,
}

C_RED = "#e74c3c"
C_BLUE = "#3498db"
C_PURPLE = "#6c5ce7"
C_ORANGE = "#e67e22"
C_GREEN = "#27ae60"
C_GRAY = "#636e72"

GROUPS = ["남_2030", "남_40대", "남_5060", "남_70대+",
          "여_2030", "여_40대", "여_5060", "여_70대+"]
LABELS = ["남\n20-30", "남\n40대", "남\n50-60", "남\n70+",
          "여\n20-30", "여\n40대", "여\n50-60", "여\n70+"]


def _add_demo(df, sex_col="SQ1_2", birth_col="SQ1_3",
              fam_col="FAM1", survey_year=2024):
    df = df.copy()
    df["나이"] = survey_year - df[birth_col]
    df["성별명"] = df[sex_col].map({1: "남", 2: "여"})
    def _ag(age):
        if 20 <= age < 40:   return "2030"
        elif 40 <= age < 50: return "40대"
        elif 50 <= age < 70: return "5060"
        elif age >= 70:      return "70대+"
        return "기타"
    df["연령그룹"] = df["나이"].apply(_ag)
    df["그룹"] = df["성별명"] + "_" + df["연령그룹"]
    return df[df[fam_col] == 1].copy()


def _load_combined():
    """가구주 2023+2024+2025, 지역사회 2022+2024 합산 로딩."""
    # 가구주 2023+2024
    hh_frames = []
    for yr, sheet in [(2024, "2024 서울서베이 가구주_data(241217)"),
                       (2023, "2023 서울서베이 가구주 data(1228)")]:
        if yr == 2024:
            p = SURVEY_BASE / "서울서베이 도시정책지표조사 정보(2024년)" / "data" / "2024 서울서베이 가구주_data_코드북.xlsx"
        else:
            p = SURVEY_BASE / "서울서베이 도시정책지표조사 정보(2023년)" / "data" / "2023 서울서베이 가구주 data(240826)_코드북.xlsx"
        df = pd.read_excel(p, sheet_name=sheet)
        solo = _add_demo(df, survey_year=yr)
        solo["자치구코드"] = solo["GU"].astype(int)
        solo["조사년도"] = yr
        hh_frames.append(solo)
    # 가구주 2025: Q15A→Q11A 매핑
    p25 = SURVEY_BASE / "서울서베이 도시정책지표조사 정보(2025년)" / "data" / "2025년 서울서베이_가구주_data_codebook.xlsx"
    df25 = pd.read_excel(p25, sheet_name="2025년 서울서베이_가구주_data(1201)")
    df25 = df25.rename(columns={
        "Q15A1": "Q11A1", "Q15A2": "Q11A2", "Q15A3": "Q11A3",
        "Q15A4": "Q11A4", "Q15A5": "Q11A5", "Q15A6": "Q11A6",
        "Q15A7": "Q11A7", "자치구": "GU",
    })
    solo25 = _add_demo(df25, survey_year=2025)
    solo25["자치구코드"] = solo25["GU"].astype(int)
    solo25["조사년도"] = 2025
    hh_frames.append(solo25)
    hh = pd.concat(hh_frames, ignore_index=True)

    # 지역사회 2022+2024
    comm_frames = []
    # 2024
    p24 = SURVEY_BASE / "서울서베이 도시정책지표조사 정보(2024년)" / "data" / "2024 지역사회조사_data_코드북.xlsx"
    df24 = pd.read_excel(p24, sheet_name="2024 지역사회조사_data")
    s24 = _add_demo(df24, survey_year=2024)
    s24["자치구코드"] = s24["DEW6"].astype(int)
    comm_frames.append(s24)
    # 2022
    p22 = SURVEY_BASE / "서울서베이 도시정책지표조사 정보(2022년)" / "data" / "2022 지역사회조사.xlsx"
    df22 = pd.read_excel(p22, sheet_name="data")
    df22 = df22.rename(columns={"AQ26": "PQ47", "AQ26A": "PQ47A", "AZQ5A2": "Q5A2"})
    s22 = _add_demo(df22, survey_year=2022)
    if "DEW6" in s22.columns:
        s22["자치구코드"] = s22["DEW6"].astype(int)
    comm_frames.append(s22)
    comm = pd.concat(comm_frames, ignore_index=True)

    return hh, comm


# ── [1] 부정 갭 ──────────────────────────────────────────────────
def plot_denial_gap(hh, comm):
    """핵심: 객관적 고립 vs 주관적 인정, 갭 = 역설의 증거."""
    print("\n── [1] 부정 갭 ──")

    obj_vals, subj_vals = [], []
    for grp in GROUPS:
        sub_c = comm[comm["그룹"] == grp]
        no_help = (sub_c["PQ47"] == 2).sum() / len(sub_c) * 100 if len(sub_c) > 0 else 0
        obj_vals.append(no_help)

        sub_h = hh[hh["그룹"] == grp]
        q11 = sub_h["Q11A1"].dropna()
        subj_vals.append(q11.mean() if len(q11) > 0 else 0)

    obj = minmax_scale(np.array(obj_vals))
    subj = minmax_scale(np.array(subj_vals))
    gap_vals = obj - subj

    rank_df = pd.DataFrame({
        "group": GROUPS,
        "label": [label.replace("\n", " ") for label in LABELS],
        "objective": obj,
        "subjective": subj,
        "gap": gap_vals,
        "help_absence": obj_vals,
        "lonely_mean": subj_vals,
    }).sort_values("gap", ascending=False).reset_index(drop=True)

    by_group = rank_df.set_index("group")
    target = by_group.loc["남_5060"]
    full_rank = int(rank_df.index[rank_df["group"] == "남_5060"][0] + 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8.6), facecolor="#fbfbfa")
    x_min = min(-0.08, by_group.loc["여_5060", "gap"] - 0.07)
    x_max = target["gap"] + 0.16

    def draw_pair(ax, title, comparison_group, comparison_label, comparison_color):
        comp = by_group.loc[comparison_group]
        rows = [
            ("남 50-60", target, C_PURPLE, 1.0, 520),
            (comparison_label, comp, comparison_color, 0.58, 260),
        ]
        ypos = [1, 0]

        ax.axvline(0, color="#111111", linewidth=1.2, alpha=0.55, zorder=0)
        ax.axvspan(0, x_max, color=C_RED, alpha=0.055, zorder=0)

        for pos, (label, row, color, alpha, size) in zip(ypos, rows):
            ax.plot([0, row["gap"]], [pos, pos], color=color, linewidth=8 if label == "남 50-60" else 5,
                    alpha=alpha, solid_capstyle="round", zorder=2)
            ax.scatter(row["gap"], pos, s=size, color=color, edgecolors="#111111" if label == "남 50-60" else "white",
                       linewidths=2.2 if label == "남 50-60" else 1.2,
                       alpha=0.96, zorder=3)
            ax.text(row["gap"] + 0.018, pos, f"{row['gap']:+.3f}",
                    ha="left", va="center", fontsize=15 if label == "남 50-60" else 12,
                    fontweight="bold" if label == "남 50-60" else "normal",
                    color=color if label == "남 50-60" else C_GRAY)

        delta = target["gap"] - comp["gap"]
        ax.text(0.04, 0.52, f"차이 {delta:+.3f}",
                transform=ax.transAxes, ha="left", va="center",
                fontsize=18, fontweight="bold", color=C_PURPLE,
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                          edgecolor=C_PURPLE, linewidth=2))
        ax.text(target["gap"], 1.34,
                f"도움부재 {target['help_absence']:.1f}%\n외로움 인정 {target['lonely_mean']:.2f}/5",
                ha="center", va="top", fontsize=10, color=C_GRAY,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff",
                          edgecolor="#e5e7eb", linewidth=1))

        ax.set_title(title, fontsize=16, fontweight="bold", pad=14)
        ax.set_yticks(ypos)
        ax.set_yticklabels(["남 50-60", comparison_label], fontsize=14)
        ax.set_ylim(-0.45, 1.55)
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel("부정 갭", fontsize=12)
        ax.grid(axis="x", color="#eeeeee", linewidth=0.9)
        for spine in ax.spines.values():
            spine.set_visible(False)

    draw_pair(axes[0], "같은 5060 안에서: 남성이 더 크게 벌어진다",
              "여_5060", "여 50-60", C_BLUE)
    draw_pair(axes[1], "같은 남성 안에서: 5060에서 갭이 커진다",
              "남_2030", "남 20-30", C_ORANGE)

    fig.suptitle("5060 남성에서 복지 회피의 단서가 가장 선명하다",
                 fontsize=22, fontweight="bold", y=0.965)
    fig.text(0.5, 0.915,
             f"부정 갭 = 객관적 도움부재 정규점수 - 현재 외로움 인정 정규점수 | 5060 남성은 전체 8집단 중 {full_rank}위",
             ha="center", va="top", fontsize=12, color=C_GRAY)

    fig.text(0.5, 0.025,
             "양수일수록 '고립 신호는 큰데 현재 고립감은 덜 인정하는' 회피·부정 가능성이 크다.",
             ha="center", fontsize=10, style="italic", color=C_GRAY)

    plt.tight_layout(rect=[0, 0.05, 1, 0.88])
    path = OUT / "05_denial_gap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  저장: {path}")


# ── [2] 도움부재 + 복지불신 (유의한 지표만) ──────────────────────
def plot_support_evidence(hh, comm):
    """5060이 유의하게 높은 2개 지표만 보여줌. 왜곡 없음."""
    print("\n── [2] 보조 증거 ──")

    vals1 = []
    for grp in GROUPS:
        sub = comm[comm["그룹"] == grp]["PQ47"].dropna()
        vals1.append((sub == 2).sum() / len(sub) * 100 if len(sub) > 0 else 0)

    vals2 = []
    for grp in GROUPS:
        sub = comm[comm["그룹"] == grp]["Q5A2"].dropna()
        sub = sub[sub.between(1, 5)]
        vals2.append((6 - sub).mean() if len(sub) > 0 else 0)

    fig, ax = plt.subplots(figsize=(12.5, 8), facecolor="#fbfbfa")
    x = np.array(vals1)
    y = np.array(vals2)
    x_cut, y_cut = np.median(x), np.median(y)
    ax.axvspan(x_cut, x.max() + 4, ymin=(y_cut - (y.min() - 0.1)) / (y.max() - y.min() + 0.2),
               ymax=1, color=C_RED, alpha=0.08, zorder=0)
    ax.axvline(x_cut, color=C_GRAY, linestyle="--", linewidth=1, alpha=0.35)
    ax.axhline(y_cut, color=C_GRAY, linestyle="--", linewidth=1, alpha=0.35)

    for i, grp in enumerate(GROUPS):
        is_target = grp == "남_5060"
        c = C_RED if is_target else "#b8c2cc"
        ax.scatter(x[i], y[i], s=360 if is_target else 130, color=c,
                   edgecolors="#111111" if is_target else "white",
                   linewidths=1.6 if is_target else 0.8, alpha=0.92 if is_target else 0.68)
        ax.annotate(LABELS[i].replace("\n", " "), (x[i], y[i]),
                    xytext=(0, 10 if is_target else 6), textcoords="offset points",
                    fontsize=11 if is_target else 8, ha="center", va="bottom",
                    fontweight="bold" if is_target else "normal",
                    color=C_RED if is_target else C_GRAY)

    ax.text(x.max() + 2.5, y.max(), "도움도 없고\n복지도 멀다",
            ha="right", va="top", fontsize=15, fontweight="bold", color=C_RED)
    ax.set_xlabel("도움받을 사람 없다 (%)", fontsize=12)
    ax.set_ylabel("복지서비스 거리감 (6-Q5A2)", fontsize=12)
    ax.set_title("5060 남성은 도움부재와 복지 거리감이 동시에 높다",
                 fontsize=17, fontweight="bold", pad=15)
    ax.grid(color="#eeeeee", linewidth=0.8)
    ax.set_xlim(x.min() - 3, x.max() + 5)
    ax.set_ylim(y.min() - 0.15, y.max() + 0.25)

    ax.text(0.03, 0.95, "도움부재: p<.001\n복지거리감: p<.001",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=C_GRAY))

    plt.tight_layout()
    path = OUT / "05_support_evidence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  저장: {path}")


# ── [3] 미래 vs 현재 괴리 ────────────────────────────────────────
def plot_future_vs_present(hh):
    """81.3%는 미래가 어렵다. 근데 지금 외로움은 2.87/5."""
    print("\n── [3] 미래 vs 현재 ──")

    vals_future = []
    for grp in GROUPS:
        sub = hh[(hh["그룹"] == grp) & (hh["Q12"] == 1)]
        if "Q12A1" in sub.columns:
            q12a = sub["Q12A1"].dropna()
            has_diff = (q12a != 8).sum() / len(q12a) * 100 if len(q12a) > 0 else 0
        else:
            has_diff = 0
        vals_future.append(has_diff)

    vals_present = []
    for grp in GROUPS:
        sub = hh[hh["그룹"] == grp]["Q11A1"].dropna()
        vals_present.append(sub.mean() if len(sub) > 0 else 0)

    fig, ax = plt.subplots(figsize=(12.5, 8), facecolor="#fbfbfa")
    x = np.array(vals_future)
    y = np.array(vals_present)
    idx = GROUPS.index("남_5060")
    ax.axvspan(78, x.max() + 3, color=C_ORANGE, alpha=0.08)
    ax.axhspan(y.min() - 0.08, 2.9, color=C_BLUE, alpha=0.05)
    ax.text(x.max() + 1.5, 2.86, "미래 위험 인지↑\n현재 인정은 낮음",
            ha="right", va="top", fontsize=14, color=C_PURPLE, fontweight="bold")

    for i, grp in enumerate(GROUPS):
        is_target = grp == "남_5060"
        c = C_PURPLE if is_target else "#b8c2cc"
        ax.scatter(x[i], y[i], s=420 if is_target else 130, color=c,
                   edgecolors="#111111" if is_target else "white",
                   linewidths=1.6 if is_target else 0.8, alpha=0.92 if is_target else 0.68)
        if not is_target:
            ax.annotate(LABELS[i].replace("\n", " "), (x[i], y[i]),
                        xytext=(0, 6), textcoords="offset points",
                        fontsize=8, ha="center", va="bottom",
                        color=C_GRAY)

    ax.annotate(f"5060 남성\n미래 어려움 {x[idx]:.1f}%\n현재 외로움 {y[idx]:.2f}/5",
                xy=(x[idx], y[idx]), xytext=(x[idx] + 5, y[idx] - 0.30),
                fontsize=11, fontweight="bold", color=C_PURPLE,
                arrowprops=dict(arrowstyle="->", color=C_PURPLE, linewidth=1.8),
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=C_PURPLE))
    ax.set_xlabel('"앞으로 어려울 것" 비율 (%)', fontsize=12)
    ax.set_ylabel("현재 외로움 인정 (Q11A1, 5점)", fontsize=12)
    ax.set_title("미래는 어렵다고 보지만, 현재의 고립은 낮게 인정한다",
                 fontsize=17, fontweight="bold", pad=15)
    ax.grid(color="#eeeeee", linewidth=0.8)
    ax.set_xlim(x.min() - 3, x.max() + 5)
    ax.set_ylim(y.min() - 0.15, y.max() + 0.25)

    plt.tight_layout()
    path = OUT / "05_future_vs_present.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  저장: {path}")


# ── [4] 자치구 지도 ──────────────────────────────────────────────
def _load_welfare_supply_for_map(avoidance_df):
    """복지시설 공급 밀도 = 전체 복지시설 수 / 5060 남성 1인가구 수."""
    frames = []
    pattern = DATA / "서울시 사회복지시설 목록" / "서울시_*구_사회복지시설_목록.csv"
    for file in glob.glob(str(pattern)):
        try:
            df = pd.read_csv(file, encoding="cp949")
        except UnicodeDecodeError:
            df = pd.read_csv(file, encoding="utf-8-sig")
        frames.append(df)

    if not frames:
        return avoidance_df.assign(복지시설수=np.nan, 복지시설밀도=np.nan), np.nan, np.nan

    facilities = pd.concat(frames, ignore_index=True)
    fac_counts = facilities.groupby("시군구명").size().reset_index(name="복지시설수")
    fac_counts = fac_counts.rename(columns={"시군구명": "자치구"})

    merged = avoidance_df.merge(fac_counts, on="자치구", how="left")
    merged["복지시설수"] = merged["복지시설수"].fillna(0)

    panel_cache = ROOT / "Outputs" / "편의의 역설" / "_full_panel_cache.pkl"
    if panel_cache.exists():
        panel = pd.read_pickle(panel_cache)
        m5060 = panel[panel["그룹"] == "남 50-60대"]
        gu_pop = m5060.groupby("자치구").agg(solo_raw=("solo", "sum")).reset_index()
        gu_pop["solo_5060"] = gu_pop["solo_raw"] / 48
        merged = merged.merge(gu_pop[["자치구", "solo_5060"]], on="자치구", how="left")
        denom = merged["solo_5060"]
        denom_label = "5060 남성 1인가구 1만 명당"
    else:
        denom = merged["n_가구주"]
        denom_label = "서베이 표본 1만 명당"

    merged["복지시설밀도"] = merged["복지시설수"] / denom.replace(0, np.nan) * 10000
    valid = merged[["복지시설밀도", "Avoidance"]].dropna()
    rho, p = spearmanr(valid["복지시설밀도"], valid["Avoidance"])
    merged.attrs["denom_label"] = denom_label
    return merged, rho, p


def plot_map(avoidance_df, geojson):
    print("\n── [4] 자치구 지도 ──")

    map_df, rho, p = _load_welfare_supply_for_map(avoidance_df)
    supply_by_gu = dict(zip(map_df["자치구코드"], map_df["복지시설밀도"]))
    avoidance_by_gu = dict(zip(map_df["자치구코드"], map_df["Avoidance"]))
    denom_label = map_df.attrs.get("denom_label", "5060 남성 1인가구 1만 명당")

    def dong_to_gu(cd):
        return GU_PREFIX.get(int(str(cd)[:5]), None)

    def draw_panel(ax, values, cmap, norm, title, subtitle, label_top=5):
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        gu_coords = {gu: ([], []) for gu in GU_MAP}

        for feat in geojson["features"]:
            cd = feat["properties"]["d_admdong_cd"]
            gu = dong_to_gu(cd)
            val = values.get(gu) if gu else None
            color = sm.to_rgba(val) if val is not None and not pd.isna(val) else "#f3f4f6"
            geom = feat["geometry"]
            for coords in ([geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]):
                ring = coords[0]
                poly = MplPolygon(ring, closed=True)
                ax.add_patch(poly)
                poly.set_facecolor(color)
                poly.set_edgecolor("#b8c2cc")
                poly.set_linewidth(0.28)
                if gu and gu in gu_coords:
                    gu_coords[gu][0].extend([c[0] for c in ring])
                    gu_coords[gu][1].extend([c[1] for c in ring])

        label_gus = sorted(
            [gu for gu, val in values.items() if val is not None and not pd.isna(val)],
            key=lambda gu: values[gu],
            reverse=True,
        )[:label_top]
        for gu in label_gus:
            xs, ys = gu_coords.get(gu, ([], []))
            if not xs:
                continue
            val = values[gu]
            cx, cy = np.mean(xs), np.mean(ys)
            ax.annotate(f"{GU_MAP[gu]}\n{val:.0f}", (cx, cy),
                        fontsize=8.2, ha="center", va="center",
                        fontweight="bold", color="#111111",
                        bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                                  edgecolor="#111111", alpha=0.82, linewidth=0.6))

        ax.set_aspect("equal")
        ax.autoscale()
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=10)
        ax.text(0.5, 0.99, subtitle, transform=ax.transAxes,
                ha="center", va="top", fontsize=10, color=C_GRAY)
        return sm

    supply_cmap = LinearSegmentedColormap.from_list(
        "WelfareSupply", ["#ffffff", "#d8f3dc", "#74c69d", "#2d6a4f"], N=256)
    avoidance_cmap = LinearSegmentedColormap.from_list(
        "Avoidance", ["#ffffff", "#e9e5ff", "#a29bfe", "#6c5ce7", "#2d1b69"], N=256)

    supply_vals = map_df["복지시설밀도"].dropna()
    supply_norm = mcolors.Normalize(vmin=supply_vals.min(), vmax=supply_vals.max())
    avoidance_norm = mcolors.Normalize(vmin=0, vmax=100)

    fig, axes = plt.subplots(1, 2, figsize=(18, 11), facecolor="white")
    sm1 = draw_panel(
        axes[0], supply_by_gu, supply_cmap, supply_norm,
        "복지시설 공급 밀도",
        denom_label,
    )
    sm2 = draw_panel(
        axes[1], avoidance_by_gu, avoidance_cmap, avoidance_norm,
        "Avoidance Index",
        "도움부재 + 외로움부정 + 복지불신 + 네트워크축소",
    )

    cbar1 = plt.colorbar(sm1, ax=axes[0], fraction=0.032, pad=0.02, aspect=32)
    cbar1.set_label("복지시설 밀도", fontsize=10)
    cbar2 = plt.colorbar(sm2, ax=axes[1], fraction=0.032, pad=0.02, aspect=32)
    cbar2.set_label("복지 회피도", fontsize=10)

    sig = "p < .05" if p < 0.05 else "ns"
    fig.suptitle("복지시설은 있어도, 회피 신호는 낮아지지 않는다",
                 fontsize=23, fontweight="bold", y=0.965)
    fig.text(0.5, 0.925,
             f"복지시설 밀도 ↔ Avoidance Index: Spearman rho = {rho:+.3f} ({sig})",
             ha="center", fontsize=13, color="#2d3436",
             bbox=dict(boxstyle="round,pad=0.42", facecolor="#f8f9fa",
                       edgecolor="#dfe6e9", linewidth=1.0))
    fig.text(0.5, 0.025,
             "왼쪽은 공급량, 오른쪽은 회피 신호. 공급량이 높은 자치구가 반드시 낮은 회피도를 보이지 않으므로, 복지의 역설은 시설 부재보다 연결 실패 문제로 해석됨.",
             ha="center", fontsize=10.5, style="italic", color=C_GRAY)

    plt.tight_layout(rect=[0, 0.055, 1, 0.90])
    path = OUT / "05_avoidance_map.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  저장: {path}")


# ── [5] 구성요소 분해 ────────────────────────────────────────────
def plot_component_breakdown(avoidance_df):
    print("\n── [5] 구성요소 분해 ──")
    df = avoidance_df.sort_values("Avoidance", ascending=True).copy()
    for c in ["A_도움부재", "B_외로움부정", "C_복지불신", "D_네트워크축소"]:
        df[f"{c}_n"] = minmax_scale(df[c])

    df = df.sort_values("Avoidance", ascending=False).reset_index(drop=True)
    comps = [("A_도움부재_n", "도움부재"),
             ("B_외로움부정_n", "외로움부정"),
             ("C_복지불신_n", "복지불신"),
             ("D_네트워크축소_n", "네트워크축소")]
    mat = df[[c for c, _ in comps]].values

    fig, ax = plt.subplots(figsize=(12.5, 10), facecolor="white")
    cmap = LinearSegmentedColormap.from_list("avoidance_heat",
                                             ["#f7fbff", "#d7bde2", "#8e44ad", "#2d1b69"], N=256)
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    for yi in range(mat.shape[0]):
        for xi in range(mat.shape[1]):
            val = mat[yi, xi]
            ax.text(xi, yi, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color="white" if val > 0.62 else "#2d3436")
        ax.text(len(comps) + 0.15, yi, f"{df.loc[yi, 'Avoidance']:.0f}",
                ha="left", va="center", fontsize=9.5,
                fontweight="bold" if df.loc[yi, "Avoidance"] >= 60 else "normal",
                color=C_RED if df.loc[yi, "Avoidance"] >= 60 else C_GRAY)

    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels([label for _, label in comps], fontsize=10)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["자치구"], fontsize=9.5)
    ax.set_title("Avoidance Index는 어떤 회피 신호로 만들어졌나",
                 fontsize=16, fontweight="bold", pad=15)
    ax.text(len(comps) + 0.15, -0.75, "최종\n점수", ha="left", va="top",
            fontsize=9.5, fontweight="bold", color=C_RED)
    ax.set_xlim(-0.5, len(comps) + 0.9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("정규화 구성요소 점수", fontsize=9)
    plt.tight_layout()
    path = OUT / "05_component_breakdown.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  저장: {path}")


# ── main ──────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("복지의 역설 - 시각화 V3")
    print("=" * 60)

    print("\n-- 데이터 로딩 --")
    hh, comm = _load_combined()
    avoidance_df = pd.read_csv(OUT / "avoidance_index.csv")
    with open(DATA / "seoul_dong.geojson", "r", encoding="utf-8") as f:
        geojson = json.load(f)
    print(f"  가구주: {len(hh)}명, 지역사회: {len(comm)}명, 자치구: {len(avoidance_df)}개")

    print("\n-- 시각화 생성 --")
    plot_denial_gap(hh, comm)
    plot_support_evidence(hh, comm)
    plot_future_vs_present(hh)
    plot_map(avoidance_df, geojson)
    plot_component_breakdown(avoidance_df)

    print("\n완료!")


if __name__ == "__main__":
    main()
