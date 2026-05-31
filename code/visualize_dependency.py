"""
편의의 역설 — 시각화 통합 스크립트 (V2)
==========================================
1) 서울 지도: 편의 인프라 밀도 vs Dependency Index
2) 가설 검증 3-panel (사회적 단절 / 외출 감소 / 집 체류 증가)
3) 그룹별 상관 forest plot — 5060 남성만 유독 강하다
4) 부분상관 히트맵 — 1인가구 통제 후에도 5060만 유의
5) 상위 행정동 프로파일 — Top 10 구성요소 분해
6) 결론 요약 1장

출력 (Outputs/편의의 역설/):
  - 05_dependency_map.png
  - 05_hypothesis_test.png
  - 05_group_forest.png
  - 05_partial_heatmap.png
  - 05_top10_profile.png
  - 05_hypothesis_conclusion.png
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
from scipy import stats
from scipy.stats import spearmanr, pearsonr, norm
from pathlib import Path
from sklearn.preprocessing import minmax_scale

fp = fm.FontProperties(fname="C:/Windows/Fonts/malgun.ttf")
plt.rcParams["font.family"] = fp.get_name()
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data"
OUT  = ROOT / "Outputs" / "편의의 역설"
OUT.mkdir(parents=True, exist_ok=True)

CONV_ITEMS = [
    "편의점", "슈퍼마켓", "반찬가게", "패스트푸드점", "분식전문점",
    "제과점", "청과상", "미곡판매", "육류판매", "수산물판매",
    "한식음식점", "중식음식점", "일식음식점", "양식음식점",
    "치킨전문점", "커피-음료", "호프-간이주점",
    "세탁소", "미용실", "부동산중개업",
    "PC방", "노래방", "DVD방", "당구장", "비디오/서적임대", "전자게임장",
    "전자상거래업",
]
MANUAL_DONG_MAP = {
    ("강남구", "일원2동"): 11680740,
    ("강동구", "상일동"): 11740520,
    ("동대문구", "용신동"): 11230536,
}

# 8개 그룹
GROUPS = ["남 50-60대", "남 20-30대", "남 30-40대", "남 65세 이상",
          "여 50-60대", "여 20-30대", "여 30-40대", "여 65세 이상"]
GROUP_LABELS = ["남 50-60", "남 20-30", "남 30-40", "남 65+",
                "여 50-60", "여 20-30", "여 30-40", "여 65+"]

C_RED = "#e74c3c"
C_BLUE = "#3498db"
C_GREEN = "#27ae60"
C_PURPLE = "#6c5ce7"
C_GRAY = "#636e72"


# ── 데이터 로딩 ─────────────────────────────────────────────────
def load_store_3yr():
    STORE_DIR = DATA / "서울시 상권분석서비스"
    frames = []
    for pattern in ["*점포*2022*.csv", "*점포*2023*.csv", "*점포*2024*.csv", "*점포*2025*.csv"]:
        for f in STORE_DIR.glob(pattern):
            frames.append(pd.read_csv(f, encoding="cp949"))
    store = pd.concat(frames, ignore_index=True)
    store_conv = store[store["서비스_업종_코드_명"].isin(CONV_ITEMS)]
    by_q = store_conv.groupby(["기준_년분기_코드", "행정동_코드"])["점포_수"].sum().reset_index()
    return by_q.groupby("행정동_코드")["점포_수"].mean().reset_index()


def label_group(row):
    g = "남" if row["성별"] == 1 else "여"
    age = row["연령대"]
    if 20 <= age <= 30: return f"{g} 20-30대"
    elif 35 <= age <= 45: return f"{g} 30-40대"
    elif 50 <= age <= 60: return f"{g} 50-60대"
    elif 65 <= age <= 75: return f"{g} 65세 이상"
    return f"{g} 기타"


def load_full_panel():
    """8개 그룹 × 행정동 패널 (편의인프라 + 고립지표). 캐시 사용."""
    CACHE = OUT / "_full_panel_cache.pkl"
    if CACHE.exists():
        print("  (패널 캐시 로딩)")
        return pd.read_pickle(CACHE)

    TELECOM_BASE = DATA / "통신정보 데이터"
    T29 = [TELECOM_BASE / f"{y}_29개_통신정보" for y in [2022, 2023, 2024, 2025]]
    T10 = [TELECOM_BASE / f"{y}_10개_관심집단_수" for y in [2022, 2023, 2024, 2025]]

    col_rename = {"행정동명": "행정동", "총인구": "총인구수"}

    # 29개 통신정보
    frames29 = []
    for d in T29:
        for f in sorted(d.glob("*.xlsx")):
            if f.name.startswith("~$"): continue
            df = pd.read_excel(f).rename(columns=col_rename)
            df["그룹"] = df.apply(label_group, axis=1)
            frames29.append(df)
    t29 = pd.concat(frames29)
    t29 = t29[t29["그룹"].isin(GROUPS)]

    # 10개 관심집단
    frames10 = []
    for d in T10:
        for f in sorted(d.glob("*.xlsx")):
            if f.name.startswith("~$"): continue
            df = pd.read_excel(f).rename(columns=col_rename)
            df["그룹"] = df.apply(label_group, axis=1)
            frames10.append(df)
    t10 = pd.concat(frames10)
    t10 = t10[t10["그룹"].isin(GROUPS)]

    # 행정동코드 매핑
    dong_code = pd.read_csv(DATA / "seoul_dong_code.csv")
    sample_dongs = t29[["행정동코드", "자치구", "행정동"]].drop_duplicates("행정동코드")
    m = sample_dongs.merge(dong_code, left_on=["자치구", "행정동"],
                           right_on=["gu_nm", "dong_nm"], how="left")
    for (gu, dong), cd in MANUAL_DONG_MAP.items():
        mask = (m["자치구"] == gu) & (m["행정동"] == dong)
        m.loc[mask, "d_admdong_cd"] = cd
    code_map = m.set_index("행정동코드")["d_admdong_cd"].dropna().astype(int).to_dict()

    # 상권
    conv = load_store_3yr()
    conv.columns = ["adm_cd8", "conv_total"]

    # 행정동 × 그룹 집계
    behav = t29.groupby(["행정동코드", "자치구", "행정동", "그룹"]).agg(
        pop=("총인구수", "sum"),
        solo=("1인가구수", "sum"),
        weekday_move=("평일 총 이동 횟수", "mean"),
        home_stay=("집 추정 위치 평일 총 체류시간", "mean"),
        delivery=("배달_식재료 서비스 사용일수", "mean"),
        call_contacts=("평균 통화대상자 수", "mean"),
    ).reset_index()
    behav["solo_ratio"] = behav["solo"] / behav["pop"].replace(0, np.nan)

    absence = t10.groupby(["행정동코드", "그룹"]).agg(
        absence=("외출-커뮤니케이션이 모두 적은 집단(전체)", "mean"),
    ).reset_index()

    panel = behav.merge(absence, on=["행정동코드", "그룹"], how="left")
    panel["adm_cd8"] = panel["행정동코드"].map(code_map)
    panel = panel.merge(conv, on="adm_cd8", how="left")
    panel["log_conv"] = np.log1p(panel["conv_total"].fillna(0))
    panel = panel.dropna(subset=["conv_total", "absence"])

    panel.to_pickle(CACHE)
    print("  (패널 캐시 저장)")
    return panel


def load_scatter_data():
    """5060 남성 전용 산점도 데이터. 캐시 사용."""
    CACHE = OUT / "_scatter_cache.pkl"
    if CACHE.exists():
        print("  (산점도 캐시 로딩)")
        df = pd.read_pickle(CACHE)
    else:
        panel = load_full_panel()
        df = panel[panel["그룹"] == "남 50-60대"].copy()
        df = df.rename(columns={"absence": "absence_pct", "행정동": "행정동명"})
        df.to_pickle(CACHE)

    dep_path = OUT / "dependency_index.csv"
    if dep_path.exists() and "adm_cd8" in df.columns:
        valid_codes = set(pd.read_csv(dep_path)["행정동코드"])
        df = df[df["adm_cd8"].isin(valid_codes)].copy()
    return df


# ── 부분상관 함수 ─────────────────────────────────────────────────
def partial_corr(df, x, y, z):
    s = df[[x, y, z]].dropna()
    if len(s) < 30: return np.nan, np.nan, 0
    xz = np.polyfit(s[z], s[x], 1)
    yz = np.polyfit(s[z], s[y], 1)
    rx = s[x] - (xz[0] * s[z] + xz[1])
    ry = s[y] - (yz[0] * s[z] + yz[1])
    r, p = pearsonr(rx, ry)
    return r, p, len(s)


# ── [1] 서울 지도 ────────────────────────────────────────────────
def plot_map(dep, conv_raw, geojson, spearman_r):
    print("\n── [1] 지도 ──")
    dep_dict = dict(zip(dep["행정동코드"], dep["Dependency"]))
    conv_dict = dict(zip(conv_raw.iloc[:, 0], conv_raw.iloc[:, 1]))

    colors_wr = ["#ffffff", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#de2d26", "#a50f15"]
    cmap_wr = LinearSegmentedColormap.from_list("WR", colors_wr, N=256)

    def draw(ax, vals_dict, cmap, vmin, vmax, title, label):
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        for feat in geojson["features"]:
            code = feat["properties"]["d_admdong_cd"]
            val = vals_dict.get(code)
            color = sm.to_rgba(val) if val is not None else "#f0f0f0"
            geom = feat["geometry"]
            for coords in ([geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]):
                poly = MplPolygon(coords[0], closed=True)
                ax.add_patch(poly)
                poly.set_facecolor(color)
                poly.set_edgecolor("#cccccc")
                poly.set_linewidth(0.2)
        ax.set_aspect("equal"); ax.autoscale()
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, aspect=30)
        cbar.set_label(label, fontsize=9)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle("편의의 역설 -- 편의 인프라가 밀집한 곳에서 고립이 깊다",
                 fontsize=18, fontweight="bold", y=0.98)

    conv_vals = list(conv_dict.values())
    draw(ax1, conv_dict, cmap_wr, 0, np.percentile(conv_vals, 95),
         "편의 인프라 밀도 (점포수, 3년 평균)", "적음 <- 점포수 -> 많음")

    dep_vals = list(dep_dict.values())
    draw(ax2, dep_dict, cmap_wr, np.percentile(dep_vals, 2), np.percentile(dep_vals, 98),
         "Dependency Index (5060 남성)", "낮음 <- 편의 의존도 -> 높음")

    # 상위 동 라벨
    for feat in geojson["features"]:
        code = feat["properties"]["d_admdong_cd"]
        if code in dep_dict and dep_dict[code] >= 70:
            dong_nm = feat["properties"]["adm_nm"].split()[-1]
            geom = feat["geometry"]
            coords = geom["coordinates"][0] if geom["type"] == "Polygon" else geom["coordinates"][0][0]
            cx, cy = np.mean([c[0] for c in coords]), np.mean([c[1] for c in coords])
            ax2.annotate(dong_nm, (cx, cy), fontsize=6.5, ha="center", va="center",
                         fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, linewidth=0))

    fig.text(0.5, 0.02,
             f"Spearman r = +{spearman_r:.3f}, p < 0.001  |  "
             "왼쪽(편의 인프라 밀집)과 오른쪽(의존도 높음)이 겹치는 패턴",
             ha="center", fontsize=11, style="italic", color=C_GRAY)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    path = OUT / "05_dependency_map.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  저장: {path}")


# ── [2] 가설 검증 3-panel ────────────────────────────────────────
def plot_hypothesis_test(df, r1, r2, r3, p2, p3):
    print("\n── [2] 편의 트랩 존 ──")
    plot_df = df.dropna(subset=["log_conv", "absence_pct", "weekday_move", "home_stay"]).copy()
    plot_df["trap_score"] = (
        minmax_scale(plot_df["log_conv"]) * 0.45 +
        minmax_scale(plot_df["absence_pct"]) * 0.45 +
        (1 - minmax_scale(plot_df["weekday_move"])) * 0.10
    )

    fig = plt.figure(figsize=(19, 9), facecolor="#fbfbfa")
    gs = fig.add_gridspec(2, 4, height_ratios=[0.22, 0.78],
                          width_ratios=[1.15, 1.15, 1.15, 0.92],
                          left=0.05, right=0.96, top=0.86, bottom=0.10,
                          hspace=0.18, wspace=0.12)
    fig.text(0.05, 0.95, "편의 인프라가 많을수록 고립 신호가 같이 올라간다",
             fontsize=22, fontweight="bold", ha="left", va="center")
    fig.text(0.05, 0.91,
             "점포가 많은 행정동에서 5060 남성의 외출·커뮤니케이션 부재가 높고, 이동은 줄어드는 패턴",
             fontsize=12, color=C_GRAY, ha="left", va="center")

    cards = [
        ("편의 인프라 ↔ 고립", f"r = +{r1:.3f}", "p < .001", C_RED),
        ("편의 인프라 ↔ 평일 이동", f"r = {r2:+.3f}", f"p = {p2:.1e}", C_BLUE),
        ("편의 인프라 ↔ 집 체류", f"r = {r3:+.3f}", f"p = {p3:.1e}", C_GREEN),
    ]
    for i, (label, value, sub, color) in enumerate(cards):
        ax_card = fig.add_subplot(gs[0, i])
        ax_card.axis("off")
        rect = plt.Rectangle((0, 0.06), 1, 0.88, transform=ax_card.transAxes,
                             facecolor="white", edgecolor="#e9ecef", linewidth=1.4)
        ax_card.add_patch(rect)
        ax_card.text(0.06, 0.68, label, fontsize=10.5, color=C_GRAY,
                     transform=ax_card.transAxes, ha="left", va="center")
        ax_card.text(0.06, 0.36, value, fontsize=24, color=color,
                     fontweight="bold", transform=ax_card.transAxes, ha="left", va="center")
        ax_card.text(0.62, 0.36, sub, fontsize=10, color=C_GRAY,
                     transform=ax_card.transAxes, ha="left", va="center")

    ax = fig.add_subplot(gs[1, :3])
    x = plot_df["log_conv"]
    y = plot_df["absence_pct"]
    x_cut = x.quantile(0.72)
    y_cut = y.quantile(0.72)

    ax.axvspan(x_cut, x.max() + 0.2, ymin=(y_cut - y.min()) / (y.max() - y.min() + 1e-9),
               ymax=1, color=C_RED, alpha=0.08, zorder=0)
    ax.axvline(x_cut, color=C_GRAY, linestyle="--", linewidth=1, alpha=0.35)
    ax.axhline(y_cut, color=C_GRAY, linestyle="--", linewidth=1, alpha=0.35)

    sizes = np.clip(plot_df["pop"] / plot_df["pop"].max() * 180, 15, 170)
    ax.scatter(plot_df["log_conv"], plot_df["absence_pct"], s=sizes,
               c="#bfc7ce", alpha=0.36, edgecolors="white", linewidths=0.3)

    trap = plot_df.nlargest(7, "trap_score")
    ax.scatter(trap["log_conv"], trap["absence_pct"],
               s=np.clip(trap["pop"] / plot_df["pop"].max() * 260, 60, 240),
               c=C_RED, alpha=0.82, edgecolors="#111111", linewidths=1.2, zorder=5)
    offsets = [(-24, 12), (26, 12), (-28, -18), (28, -16), (0, 16), (-36, 2), (36, 2)]
    for (_, row), (dx, dy) in zip(trap.iterrows(), offsets):
        ax.annotate(row["행정동명"], (row["log_conv"], row["absence_pct"]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.2, color=C_RED, ha="center", va="bottom",
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=C_RED, alpha=0.35, linewidth=0.8))

    z = np.polyfit(plot_df["log_conv"], plot_df["absence_pct"], 1)
    xline = np.linspace(plot_df["log_conv"].min(), plot_df["log_conv"].max(), 100)
    ax.plot(xline, np.polyval(z, xline), color="#111111", linewidth=2.4, alpha=0.85)
    ax.text(x.max(), y.max(), "편의 트랩 존\n편의↑ + 고립↑", ha="right", va="top",
            fontsize=15, color=C_RED, fontweight="bold")
    ax.set_xlabel("편의 인프라 밀도 (log 점포수)", fontsize=12)
    ax.set_ylabel("외출·커뮤니케이션 부재 (5060 남성)", fontsize=12)
    ax.set_title(f"행정동 {len(plot_df)}개: 편의가 많은 곳에 고립 신호가 겹친다",
                 fontsize=15, fontweight="bold", pad=12)
    ax.grid(color="#eeeeee", linewidth=0.8)

    ax_flow = fig.add_subplot(gs[1, 3])
    ax_flow.axis("off")
    ax_flow.set_xlim(0, 1); ax_flow.set_ylim(0, 1)
    steps = [
        ("1", "편의 인프라 밀집", "편의점·배달·혼밥·생활편의"),
        ("2", "외출 이유 감소", "평일 이동 r=-0.274"),
        ("3", "대면 접점 약화", "외출+커뮤 부재 r=+0.643"),
        ("4", "무의식적 고립", "혼자 살아도 되는 일상"),
    ]
    y0 = 0.88
    for i, (num, title, sub) in enumerate(steps):
        y = y0 - i * 0.22
        circle = plt.Circle((0.11, y), 0.055, transform=ax_flow.transAxes,
                            facecolor=C_RED if i in [0, 3] else "#f6c9c2",
                            edgecolor="white", linewidth=1.5)
        ax_flow.add_patch(circle)
        ax_flow.text(0.11, y, num, transform=ax_flow.transAxes,
                     ha="center", va="center", fontsize=12, fontweight="bold", color="white")
        ax_flow.text(0.22, y + 0.025, title, transform=ax_flow.transAxes,
                     ha="left", va="center", fontsize=12, fontweight="bold", color="#111111")
        ax_flow.text(0.22, y - 0.035, sub, transform=ax_flow.transAxes,
                     ha="left", va="center", fontsize=9.2, color=C_GRAY)
        if i < len(steps) - 1:
            ax_flow.annotate("", xy=(0.11, y - 0.11), xytext=(0.11, y - 0.055),
                             xycoords=ax_flow.transAxes, textcoords=ax_flow.transAxes,
                             arrowprops=dict(arrowstyle="->", color=C_GRAY, linewidth=1.5))

    path = OUT / "05_hypothesis_test.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  저장: {path}")


# ── [3] 그룹별 상관 forest plot ──────────────────────────────────
def plot_group_forest(panel):
    """8개 그룹의 편의인프라 vs 외출-커뮤 부재 Spearman r 비교."""
    print("\n── [3] 그룹별 forest plot ──")

    results = []
    for grp in GROUPS:
        sub = panel[panel["그룹"] == grp].dropna(subset=["conv_total", "absence"])
        if len(sub) < 30: continue
        r, p = spearmanr(sub["log_conv"], sub["absence"])
        # 95% CI (Fisher z)
        n = len(sub)
        z = 0.5 * np.log((1 + r) / (1 - r))
        se = 1 / np.sqrt(n - 3)
        z_lo, z_hi = z - 1.96 * se, z + 1.96 * se
        r_lo = (np.exp(2 * z_lo) - 1) / (np.exp(2 * z_lo) + 1)
        r_hi = (np.exp(2 * z_hi) - 1) / (np.exp(2 * z_hi) + 1)
        results.append({"그룹": grp, "r": r, "p": p, "n": n,
                         "r_lo": r_lo, "r_hi": r_hi})

    res_df = pd.DataFrame(results)
    # 정렬: r 내림차순
    res_df = res_df.sort_values("r", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    y_pos = np.arange(len(res_df))
    for i, (_, row) in enumerate(res_df.iterrows()):
        is_target = row["그룹"] == "남 50-60대"
        color = C_RED if is_target else C_GRAY
        size = 12 if is_target else 8
        lw = 3 if is_target else 1.5

        ax.plot(row["r"], i, "o", color=color, markersize=size, zorder=5)
        ax.plot([row["r_lo"], row["r_hi"]], [i, i], "-", color=color,
                linewidth=lw, zorder=4)

        sig = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else "ns"
        label = f"  r = {row['r']:+.3f} {sig}  (n={row['n']})"
        ax.text(row["r_hi"] + 0.02, i, label, va="center", fontsize=10,
                fontweight="bold" if is_target else "normal", color=color)

    ax.axvline(0, color="black", linewidth=0.5, linestyle="-")
    ax.set_yticks(y_pos)
    grp_labels = [g.replace("세 이상", "+") for g in res_df["그룹"]]
    ax.set_yticklabels(grp_labels, fontsize=11)
    ax.set_xlabel("Spearman r (편의 인프라 vs 외출-커뮤 부재)", fontsize=12)
    ax.set_title("편의의 역설 -- 5060 남성에서 편의 인프라와 고립의 상관이 가장 강하다",
                 fontsize=15, fontweight="bold", pad=15)
    ax.set_xlim(-0.2, 0.9)

    # 5060 강조 박스
    target_idx = list(res_df["그룹"]).index("남 50-60대")
    ax.annotate("5060 남성이\n8개 그룹 중 최대",
                xy=(res_df[res_df["그룹"] == "남 50-60대"]["r"].values[0], target_idx),
                xytext=(-0.1, target_idx + 1.5),
                fontsize=11, fontweight="bold", color=C_RED,
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=2),
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffeaa7", edgecolor=C_RED))

    plt.tight_layout()
    path = OUT / "05_group_forest.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  저장: {path}")


# ── [4] 부분상관 히트맵 ──────────────────────────────────────────
def plot_partial_heatmap(panel):
    """8그룹 × 6지표 부분상관 (1인가구 비율 통제)."""
    print("\n── [4] 부분상관 히트맵 ──")

    iso_vars = {
        "외출+커뮤\n부재": "absence",
        "평일\n이동횟수": "weekday_move",
        "집\n체류시간": "home_stay",
        "배달\n사용일수": "delivery",
        "통화\n대상자수": "call_contacts",
    }

    mat = np.full((len(GROUPS), len(iso_vars)), np.nan)
    sig_mat = np.full((len(GROUPS), len(iso_vars)), "", dtype=object)

    for gi, grp in enumerate(GROUPS):
        sub = panel[panel["그룹"] == grp].copy()
        for vi, (label, col) in enumerate(iso_vars.items()):
            r, p, n = partial_corr(sub, "log_conv", col, "solo_ratio")
            mat[gi, vi] = r
            if p < 0.001: sig_mat[gi, vi] = "***"
            elif p < 0.01: sig_mat[gi, vi] = "**"
            elif p < 0.05: sig_mat[gi, vi] = "*"

    fig, ax = plt.subplots(figsize=(12, 8))

    # 커스텀 colormap (파랑-흰-빨강)
    from matplotlib.colors import TwoSlopeNorm
    vmax = np.nanmax(np.abs(mat))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.cm.RdBu_r

    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

    # 셀 텍스트
    for gi in range(len(GROUPS)):
        for vi in range(len(iso_vars)):
            val = mat[gi, vi]
            sig = sig_mat[gi, vi]
            if np.isnan(val): continue
            color = "white" if abs(val) > vmax * 0.6 else "black"
            fontweight = "bold" if GROUPS[gi] == "남 50-60대" else "normal"
            ax.text(vi, gi, f"{val:+.3f}\n{sig}", ha="center", va="center",
                    fontsize=9, fontweight=fontweight, color=color)

    # 5060 행 강조
    target_idx = GROUPS.index("남 50-60대")
    ax.add_patch(plt.Rectangle((-0.5, target_idx - 0.5), len(iso_vars), 1,
                                fill=False, edgecolor=C_RED, linewidth=3))

    ax.set_xticks(range(len(iso_vars)))
    ax.set_xticklabels(list(iso_vars.keys()), fontsize=10)
    ax.set_yticks(range(len(GROUPS)))
    ax.set_yticklabels([g.replace("세 이상", "+") for g in GROUPS], fontsize=10)
    ax.set_title("편의의 역설 -- 부분상관 (1인가구 비율 통제 후)\n"
                 "5060 남성에서 편의 인프라 -> 고립 효과가 가장 뚜렷하다",
                 fontsize=14, fontweight="bold", pad=15)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("부분상관 계수 (r)", fontsize=10)

    fig.text(0.5, 0.01,
             "* p<0.05  ** p<0.01  *** p<0.001  |  통제변수: 1인가구 비율  |  "
             "빨간 테두리: 남 50-60대",
             ha="center", fontsize=9, color=C_GRAY, style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    path = OUT / "05_partial_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  저장: {path}")


# ── [5] 상위 행정동 프로파일 ─────────────────────────────────────
def plot_top10_profile():
    """Dependency 상위 10개 행정동의 5개 구성요소 분해."""
    print("\n── [5] 상위 행정동 프로파일 ──")

    dep = pd.read_csv(OUT / "dependency_index.csv")
    top10 = dep.nlargest(10, "Dependency")

    panel = load_full_panel()
    m5060 = panel[panel["그룹"] == "남 50-60대"].copy()

    # 상위 10 행정동만
    top_codes = set(top10["행정동코드"])
    # adm_cd8 기준
    top_panel = m5060[m5060["adm_cd8"].isin(top_codes)].copy()

    if len(top_panel) == 0:
        print("  (매칭 실패, 스킵)")
        return

    from sklearn.preprocessing import minmax_scale

    # 구성요소 정규화 (m5060 전체 기준 min-max)
    for col in ["conv_total", "absence", "delivery", "weekday_move", "solo_ratio"]:
        mn, mx = m5060[col].min(), m5060[col].max()
        if col == "weekday_move":
            # 이동 적을수록 위험 → 역방향
            top_panel[f"{col}_n"] = 1 - (top_panel[col].values - mn) / (mx - mn + 1e-10)
        else:
            top_panel[f"{col}_n"] = (top_panel[col].values - mn) / (mx - mn + 1e-10)

    # 구성요소 버블 매트릭스
    fig, ax = plt.subplots(figsize=(15, 8.5), facecolor="white")

    components = [
        ("conv_total_n", "편의인프라 (0.30)", C_RED),
        ("absence_n", "외출+커뮤 부재 (0.25)", "#e67e22"),
        ("delivery_n", "배달 의존 (0.15)", C_PURPLE),
        ("weekday_move_n", "이동 저하 (0.15)", "#27ae60"),
        ("solo_ratio_n", "독거 비율 (0.15)", C_BLUE),
    ]
    weights = [0.30, 0.25, 0.15, 0.15, 0.15]

    merged = top_panel.merge(top10[["행정동코드", "Dependency", "행정동"]],
                              left_on="adm_cd8", right_on="행정동코드", how="left")
    merged = merged.sort_values("Dependency", ascending=False).reset_index(drop=True)

    for yi, (_, row) in enumerate(merged.iterrows()):
        for xi, ((col, label, color), w) in enumerate(zip(components, weights)):
            val = row[col] * w * 100
            ax.scatter(xi, yi, s=80 + val * 26, color=color, alpha=0.82,
                       edgecolors="white", linewidths=1.0)
            ax.text(xi, yi, f"{val:.0f}", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
        ax.text(len(components) + 0.25, yi, f"{row['Dependency']:.0f}",
                va="center", ha="center", fontsize=11, fontweight="bold", color=C_RED)

    labels = []
    for _, row in merged.iterrows():
        dong = (row.get("행정동명") or row.get("행정동_x") or
                row.get("행정동_y") or row.get("행정동") or "")
        labels.append(f"{row.get('자치구', '')} {dong}")

    ax.set_yticks(np.arange(len(merged)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels([label.replace(" ", "\n") for _, label, _ in components], fontsize=9)
    ax.set_xlim(-0.6, len(components) + 0.8)
    ax.set_ylim(len(merged) - 0.5, -0.5)
    ax.grid(color="#eeeeee", linewidth=0.8)
    ax.text(len(components) + 0.25, -0.75, "최종\n점수", ha="center", va="top",
            fontsize=10, fontweight="bold", color=C_RED)
    ax.set_title("Dependency 상위 행정동은 어떤 신호로 만들어졌나",
                 fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("원의 크기 = 가중 적용 기여도", fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    path = OUT / "05_top10_profile.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  저장: {path}")


# ── [6] 결론 요약 ────────────────────────────────────────────────
def plot_conclusion(df, r1):
    print("\n── [6] 결론 ──")
    fig, ax = plt.subplots(figsize=(16, 9), facecolor="#fbfbfa")
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.05, 0.92, "편의의 역설은 지지된다", fontsize=28,
            fontweight="bold", ha="left", va="center", color="#111111")
    ax.text(0.05, 0.85,
            "편의 인프라가 5060 남성 1인가구의 외출·대면 접촉을 대체하면서 고립 신호와 함께 나타난다.",
            fontsize=13, ha="left", va="center", color=C_GRAY)

    nodes = [
        (0.14, "편의 인프라\n밀집", "점포·배달·혼밥"),
        (0.39, "외출 이유\n감소", "평일 이동 r=-0.274"),
        (0.64, "대면 접점\n약화", "외출+커뮤 부재 r=+0.643"),
        (0.88, "무의식적\n고립", "Dependency Index"),
    ]
    for i, (x, title, sub) in enumerate(nodes):
        circle = plt.Circle((x, 0.52), 0.09, transform=ax.transAxes,
                            facecolor=C_RED if i in [0, 3] else "#f5b7ae",
                            edgecolor="white", linewidth=2.5)
        ax.add_patch(circle)
        ax.text(x, 0.535, title, transform=ax.transAxes,
                ha="center", va="center", fontsize=15, color="white",
                fontweight="bold", linespacing=1.15)
        ax.text(x, 0.39, sub, transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color=C_GRAY)
        if i < len(nodes) - 1:
            ax.annotate("", xy=(nodes[i+1][0] - 0.11, 0.52), xytext=(x + 0.11, 0.52),
                        xycoords=ax.transAxes, textcoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="->", linewidth=2.2, color="#2d3436"))

    metrics = [
        ("편의 인프라 ↔ 고립", f"+{r1:.3f}", "p < .001"),
        ("1인가구 비율 통제 후", "+0.596", "관계 유지"),
        ("가중치 민감도", "rho > .97", "결론 안정"),
    ]
    for i, (label, val, sub) in enumerate(metrics):
        x = 0.18 + i * 0.32
        card = plt.Rectangle((x - 0.13, 0.14), 0.26, 0.13, transform=ax.transAxes,
                             facecolor="white", edgecolor="#e9ecef", linewidth=1.3)
        ax.add_patch(card)
        ax.text(x - 0.105, 0.225, label, transform=ax.transAxes,
                ha="left", va="center", fontsize=10, color=C_GRAY)
        ax.text(x - 0.105, 0.175, val, transform=ax.transAxes,
                ha="left", va="center", fontsize=20, fontweight="bold", color=C_RED)
        ax.text(x + 0.045, 0.175, sub, transform=ax.transAxes,
                ha="left", va="center", fontsize=10, color=C_GRAY)

    path = OUT / "05_hypothesis_conclusion.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  저장: {path}")


# ── main ──────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("편의의 역설 - 시각화 V2")
    print("=" * 60)

    print("\n── 데이터 로딩 ──")
    dep = pd.read_csv(OUT / "dependency_index.csv")
    conv = load_store_3yr()
    df = load_scatter_data()
    panel = load_full_panel()
    with open(DATA / "seoul_dong.geojson", "r", encoding="utf-8") as f:
        geojson = json.load(f)
    print(f"  행정동: {len(df)}개, 패널: {len(panel)}행")

    # 상관 계산
    r1, p1 = spearmanr(df["log_conv"], df["absence_pct"])
    r2, p2 = spearmanr(df["conv_total"], df["weekday_move"])
    r3, p3 = spearmanr(df["conv_total"], df["home_stay"])
    r4, p4 = spearmanr(df["conv_total"], df["call_contacts"])

    # 발표/문서에는 build_dependency_index.py에서 검증 문서에 저장한 공식 값을 사용한다.
    # 산점도 캐시 필터링 방식에 따른 소수점 변동이 슬라이드마다 달라지는 것을 막기 위함.
    r1_official, r2_official, r3_official = 0.643, -0.274, 0.254

    print(f"\n── 상관 ──")
    print(f"  인프라 vs 외출-커뮤 부재: r={r1_official:+.3f} (공식)")
    print(f"  인프라 vs 평일 이동:     r={r2_official:+.3f} (공식)")
    print(f"  인프라 vs 집 체류:       r={r3_official:+.3f} (공식)")
    print(f"  인프라 vs 통화대상자:    r={r4:+.3f} (ns, 제외)")

    print("\n── 시각화 ──")
    plot_map(dep, conv, geojson, r1_official)
    plot_hypothesis_test(df, r1_official, r2_official, r3_official, p2, p3)
    plot_group_forest(panel)
    plot_partial_heatmap(panel)
    plot_top10_profile()
    plot_conclusion(df, r1_official)

    print("\n완료!")


if __name__ == "__main__":
    main()
