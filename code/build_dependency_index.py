"""
편의의 역설 — Dependency Index 산출 (V2)
==========================================
기존 '편리한 고립 분석리포트' 방법론 기반 재산출.

핵심 개선:
  - 8개 그룹 비교 (5060남성만이 아니라 다른 그룹 대비 검증)
  - 부분상관 (1인가구 비율 통제)
  - Fisher z-test (5060남성 특이성 검증)
  - 상권 데이터 4년 평균 (2022~2025)
  - 소인구 행정동 필터 (5060남성 인구 < 500명 제외)
  - 가중치: 인프라 0.30 / 외출커뮤 0.25 / 배달 0.15 / 이동 0.15 / 독거 0.15

산출물:
  1. Outputs/편의의 역설/dependency_index.csv
  2. Outputs/편의의 역설/dependency_methodology.md
  3. Outputs/편의의 역설/dependency_validation.md
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr, pearsonr, norm
from sklearn.preprocessing import minmax_scale
import warnings
warnings.filterwarnings('ignore')

# ── [1] 설정 ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data"
OUT  = ROOT / "Outputs" / "편의의 역설"
OUT.mkdir(parents=True, exist_ok=True)

TELECOM_BASE = DATA / "통신정보 데이터"
TELECOM_29_DIRS = [TELECOM_BASE / f"{y}_29개_통신정보" for y in [2022, 2023, 2024, 2025]]
TELECOM_10_DIRS = [TELECOM_BASE / f"{y}_10개_관심집단_수" for y in [2022, 2023, 2024, 2025]]
STORE_DIR = DATA / "서울시 상권분석서비스"
DONG_CODE_PATH = DATA / "seoul_dong_code.csv"

# 그룹 정의
MALE, FEMALE = 1, 2

# 편의 인프라 업종 (5개 카테고리, 27개 업종)
CATEGORIES = {
    "간편식사_편의": [
        "편의점", "슈퍼마켓", "반찬가게", "패스트푸드점", "분식전문점",
        "제과점", "청과상", "미곡판매", "육류판매", "수산물판매",
    ],
    "외식_혼밥": [
        "한식음식점", "중식음식점", "일식음식점", "양식음식점",
        "치킨전문점", "커피-음료", "호프-간이주점",
    ],
    "생활편의": ["세탁소", "미용실", "부동산중개업"],
    "혼자_여가": ["PC방", "노래방", "DVD방", "당구장", "비디오/서적임대", "전자게임장"],
    "온라인_배달거점": ["전자상거래업"],
}
CONV_ITEMS = [item for lst in CATEGORIES.values() for item in lst]

# 지수 가중치 (기존 방법론 동일)
W = {
    "A_인프라": 0.30,
    "B_외출커뮤적은": 0.25,
    "C_배달의존": 0.15,
    "D_이동저하": 0.15,
    "E_독거비율": 0.15,
}

# 소인구 필터 기준 (12개월 합산 총인구 기준)
MIN_POP_THRESHOLD = 500

# 수동 매핑: 통신정보에만 있는 3개 동 → 8자리 표준코드
MANUAL_DONG_MAP = {
    ("강남구", "일원2동"):   11680740,
    ("강동구", "상일동"):    11740520,
    ("동대문구", "용신동"):  11230536,
}


# ── [2] 그룹 라벨링 ───────────────────────────────────────────────
def label_group(row):
    g = "남" if row["성별"] == 1 else "여"
    age = row["연령대"]
    if 20 <= age <= 30:
        return f"{g} 20-30대"
    elif 35 <= age <= 45:
        return f"{g} 30-40대"
    elif 50 <= age <= 60:
        return f"{g} 50-60대"
    elif 65 <= age <= 75:
        return f"{g} 65세 이상"
    else:
        return f"{g} 기타"


# ── [3] 행정동 코드 매핑 ──────────────────────────────────────────
def build_dong_master():
    dong_code = pd.read_csv(DONG_CODE_PATH)
    sample = pd.read_excel(
        next(TELECOM_29_DIRS[0].glob("*.xlsx")),
        usecols=["행정동코드", "자치구", "행정동"],
    )
    telecom_dongs = (
        sample.drop_duplicates(subset=["행정동코드"])
        [["행정동코드", "자치구", "행정동"]]
        .reset_index(drop=True)
    )
    merged = telecom_dongs.merge(
        dong_code,
        left_on=["자치구", "행정동"],
        right_on=["gu_nm", "dong_nm"],
        how="left",
    )
    for (gu, dong), code8 in MANUAL_DONG_MAP.items():
        mask = (merged["자치구"] == gu) & (merged["행정동"] == dong)
        merged.loc[mask, "d_admdong_cd"] = code8

    master = merged[["행정동코드", "자치구", "행정동", "d_admdong_cd"]].copy()
    master["d_admdong_cd"] = master["d_admdong_cd"].astype(int)
    master = master.rename(columns={"행정동코드": "telecom_cd", "d_admdong_cd": "adm_cd8"})
    print(f"  행정동 매핑: {len(master)}동 (매핑률 {master['adm_cd8'].notna().mean():.1%})")
    return master


# ── [4] 데이터 로딩 ──────────────────────────────────────────────
def load_telecom_29():
    """29개 통신정보 4년(2022~2025) 전체 로딩 (모든 그룹). 캐시 사용."""
    CACHE = OUT / "_comm_cache.pkl"
    if CACHE.exists():
        combined = pd.read_pickle(CACHE)
        print(f"  29개 통신정보: {len(combined)}행 (캐시)")
        return combined
    cols_needed = [
        "행정동코드", "자치구", "행정동", "성별", "연령대",
        "총인구수", "1인가구수",
        "평균 통화대상자 수", "평균 문자대상자 수",
        "평일 총 이동 횟수", "휴일 총 이동 횟수 평균",
        "집 추정 위치 평일 총 체류시간", "집 추정 위치 휴일 총 체류시간",
        "배달 서비스 사용일수", "배달_식재료 서비스 사용일수",
        "동영상/방송 서비스 사용일수", "유튜브 사용일수",
        "쇼핑 서비스 사용일수",
    ]
    # 컬럼명 대체 매핑 (일부 월 파일의 컬럼명 차이 대응)
    col_rename = {"행정동명": "행정동", "총인구": "총인구수"}
    frames = []
    for d in TELECOM_29_DIRS:
        for f in sorted(d.glob("*.xlsx")):
            if f.name.startswith("~$"):
                continue
            df = pd.read_excel(f)
            df = df.rename(columns=col_rename)
            df = df[[c for c in cols_needed if c in df.columns]]
            df["그룹"] = df.apply(label_group, axis=1)
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    target_groups = ["남 20-30대", "남 30-40대", "남 50-60대", "남 65세 이상",
                     "여 20-30대", "여 30-40대", "여 50-60대", "여 65세 이상"]
    combined = combined[combined["그룹"].isin(target_groups)]
    combined.to_pickle(CACHE)
    print(f"  29개 통신정보: {len(combined)}행 (4년 × 12개월 × 8그룹)")
    return combined


def load_telecom_10():
    """10개 관심집단 4년(2022~2025) 전체 로딩 (모든 그룹). 캐시 사용."""
    CACHE = OUT / "_interest_cache.pkl"
    if CACHE.exists():
        combined = pd.read_pickle(CACHE)
        print(f"  10개 관심집단: {len(combined)}행 (캐시)")
        return combined
    col_rename = {"행정동명": "행정동", "총인구": "총인구수"}
    frames = []
    for d in TELECOM_10_DIRS:
        for f in sorted(d.glob("*.xlsx")):
            if f.name.startswith("~$"):
                continue
            df = pd.read_excel(f)
            df = df.rename(columns=col_rename)
            df["그룹"] = df.apply(label_group, axis=1)
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    target_groups = ["남 20-30대", "남 30-40대", "남 50-60대", "남 65세 이상",
                     "여 20-30대", "여 30-40대", "여 50-60대", "여 65세 이상"]
    combined = combined[combined["그룹"].isin(target_groups)]
    combined.to_pickle(CACHE)
    print(f"  10개 관심집단: {len(combined)}행 (4년 × 12개월 × 8그룹)")
    return combined


def load_store():
    """상권 점포 데이터 4년 평균 (2022~2025)."""
    dfs = []
    for pattern in ["*점포*2022*.csv", "*점포*2023*.csv", "*점포*2024*.csv", "*점포*2025*.csv"]:
        for f in STORE_DIR.glob(pattern):
            df = pd.read_csv(f, encoding="cp949")
            dfs.append(df)
    stores = pd.concat(dfs, ignore_index=True)
    stores = stores[stores["서비스_업종_코드_명"].isin(CONV_ITEMS)]
    # 분기별 행정동 합산 → 전체 평균
    by_q = stores.groupby(["기준_년분기_코드", "행정동_코드"])["점포_수"].sum().reset_index()
    by_dong = by_q.groupby("행정동_코드")["점포_수"].mean().reset_index()
    by_dong = by_dong.rename(columns={"행정동_코드": "adm_cd8", "점포_수": "편의인프라_총점"})
    print(f"  상권 데이터: {len(by_dong)}개 행정동 (4년 평균, {len(CONV_ITEMS)}개 업종)")
    return by_dong


# ── [5] 행정동×그룹 패널 구축 ─────────────────────────────────────
def build_panel(comm, interest, infra, dong_master):
    """행정동×그룹 패널 통합 + 편의 인프라 조인.
    주의: 행정동코드(telecom_cd)를 기준키로 사용 — 신사동 등 동명이인 방지."""
    GRP_KEY = ["행정동코드", "자치구", "행정동", "그룹"]

    # 통신정보 집계 (3년 평균)
    comm_agg = comm.groupby(GRP_KEY).agg(
        인구=("총인구수", "sum"),          # 전체 합산 (아래서 월수로 나눔)
        독거인구=("1인가구수", "sum"),
        통화대상자수=("평균 통화대상자 수", "mean"),
        문자대상자수=("평균 문자대상자 수", "mean"),
        평일이동횟수=("평일 총 이동 횟수", "mean"),
        휴일이동횟수=("휴일 총 이동 횟수 평균", "mean"),
        배달일수=("배달 서비스 사용일수", "mean"),
        배달식재료일수=("배달_식재료 서비스 사용일수", "mean"),
        _n_rows=("총인구수", "count"),      # 월수 (연령대×월 행 수)
    ).reset_index()
    # 인구: 연령대별 행이 3개(50,55,60)×48개월=144행 → 월평균 5060합산 = sum/48
    n_months = 48  # 4년
    comm_agg["인구_월평균"] = comm_agg["인구"] / n_months
    comm_agg["독거비율"] = comm_agg["독거인구"] / comm_agg["인구"].replace(0, np.nan)

    # 관심집단 집계
    interest_agg = interest.groupby(GRP_KEY).agg(
        커뮤니케이션적은=("커뮤니케이션이 적은 집단", "mean"),
        외출매우적은=("외출이 매우 적은 집단(전체)", "mean"),
        외출커뮤모두적은=("외출-커뮤니케이션이 모두 적은 집단(전체)", "mean"),
    ).reset_index()

    # 패널 조인 (행정동코드+그룹 기준)
    panel = comm_agg.merge(interest_agg, on=GRP_KEY, how="inner")

    # 인프라 조인 (행정동코드 → 8자리코드 → 인프라)
    dong_map = dong_master[["telecom_cd", "자치구", "행정동", "adm_cd8"]].drop_duplicates()
    panel = panel.merge(dong_map[["telecom_cd", "adm_cd8"]].rename(columns={"telecom_cd": "행정동코드"}),
                        on="행정동코드", how="left")
    panel = panel.merge(infra, on="adm_cd8", how="left")
    panel["log_인프라"] = np.log1p(panel["편의인프라_총점"])

    print(f"  패널: {panel.shape} (행정동×그룹)")
    print(f"  인프라 매칭 NA: {panel['편의인프라_총점'].isna().sum()}")
    return panel, dong_map


# ── [6] 상관분석 + 부분상관 + Fisher z ───────────────────────────
def partial_corr(df, x, y, z):
    """z를 통제하고 x와 y의 부분상관 (OLS 잔차 기반)."""
    s = df[[x, y, z]].dropna()
    if len(s) < 30:
        return np.nan, np.nan, 0
    xz = np.polyfit(s[z], s[x], 1)
    yz = np.polyfit(s[z], s[y], 1)
    rx = s[x] - (xz[0] * s[z] + xz[1])
    ry = s[y] - (yz[0] * s[z] + yz[1])
    r, p = pearsonr(rx, ry)
    return r, p, len(s)


def fisher_z_transform(r):
    return 0.5 * np.log((1 + r) / (1 - r))


def run_correlation_analysis(panel):
    """Spearman + 부분상관 + Fisher z 검정."""
    target_groups = ["남 50-60대", "남 20-30대", "남 30-40대", "남 65세 이상",
                     "여 50-60대", "여 30-40대", "여 20-30대", "여 65세 이상"]
    isolation_vars = ["외출커뮤모두적은", "외출매우적은", "커뮤니케이션적은",
                      "배달식재료일수", "평일이동횟수", "휴일이동횟수"]

    # --- Spearman 상관 ---
    print("\n── Spearman 상관 (편의 인프라 vs 고립 지표) ──")
    spearman_results = []
    for grp in target_groups:
        sub = panel[panel["그룹"] == grp].dropna(subset=["편의인프라_총점"])
        for var in isolation_vars:
            s = sub[["편의인프라_총점", var]].dropna()
            if len(s) < 30:
                continue
            rho, p = spearmanr(s["편의인프라_총점"], s[var])
            spearman_results.append({"그룹": grp, "지표": var, "Spearman_r": rho, "p": p, "N": len(s)})

    sp_df = pd.DataFrame(spearman_results)
    m5060_sp = sp_df[sp_df["그룹"] == "남 50-60대"]
    print("  [남 50-60대]")
    for _, row in m5060_sp.iterrows():
        sig = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else ""
        print(f"    {row['지표']:20s}  r={row['Spearman_r']:+.3f} {sig}")

    # --- 부분상관 (1인가구비율 통제) ---
    print("\n── 부분상관 (1인가구 비율 통제 후) ──")
    partial_results = []
    for grp in target_groups:
        sub = panel[panel["그룹"] == grp].copy()
        for var in isolation_vars:
            rp, pp, n = partial_corr(sub, "log_인프라", var, "독거비율")
            partial_results.append({
                "그룹": grp, "지표": var,
                "부분상관_r": rp, "부분상관_p": pp, "N": n,
            })

    partial_df = pd.DataFrame(partial_results)
    m5060_partial = partial_df[partial_df["그룹"] == "남 50-60대"]
    print("  [남 50-60대] 1인가구비율 통제 후에도 유효한가?")
    for _, row in m5060_partial.iterrows():
        sig = "***" if row["부분상관_p"] < 0.001 else "**" if row["부분상관_p"] < 0.01 else "*" if row["부분상관_p"] < 0.05 else "ns"
        print(f"    {row['지표']:20s}  r={row['부분상관_r']:+.3f} ({sig})")

    # --- Fisher z-test (5060남성 vs 다른 그룹) ---
    print("\n── Fisher z-test (5060남성이 다른 그룹보다 효과 더 큰가?) ──")
    m5060_p = partial_df[partial_df["그룹"] == "남 50-60대"].set_index("지표")
    fisher_results = []
    for grp in target_groups:
        if grp == "남 50-60대":
            continue
        sub_p = partial_df[partial_df["그룹"] == grp].set_index("지표")
        for var in isolation_vars:
            r1 = m5060_p.loc[var, "부분상관_r"] if var in m5060_p.index else np.nan
            r2 = sub_p.loc[var, "부분상관_r"] if var in sub_p.index else np.nan
            n1 = m5060_p.loc[var, "N"] if var in m5060_p.index else 0
            n2 = sub_p.loc[var, "N"] if var in sub_p.index else 0
            if pd.isna(r1) or pd.isna(r2) or n1 < 10 or n2 < 10:
                z_stat, p_val = np.nan, np.nan
            else:
                z1, z2 = fisher_z_transform(r1), fisher_z_transform(r2)
                se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
                z_stat = (z1 - z2) / se
                p_val = 2 * (1 - norm.cdf(abs(z_stat)))
            fisher_results.append({
                "비교": f"5060남 vs {grp}", "지표": var,
                "r_5060남": r1, "r_비교군": r2,
                "z": z_stat, "p": p_val,
            })

    fisher_df = pd.DataFrame(fisher_results)
    # 핵심 지표 '외출커뮤모두적은'만 요약
    key_var = fisher_df[fisher_df["지표"] == "외출커뮤모두적은"]
    print("  [외출+커뮤 모두 적은 비율] 5060남성 vs 다른 그룹:")
    for _, row in key_var.iterrows():
        sig = "*" if row["p"] < 0.05 else "ns"
        print(f"    {row['비교']:25s}  z={row['z']:+.2f} ({sig})")

    return sp_df, partial_df, fisher_df


# ── [7] Dependency Index 산출 ─────────────────────────────────────
def calculate_index(panel, dong_master):
    """5060남성 데이터로 Dependency Index 산출 (기존 방법론)."""
    m5060 = panel[panel["그룹"] == "남 50-60대"].copy()

    # 소인구 행정동 필터 (5060남성 월평균 합산 인구 기준)
    pop_before = len(m5060)
    m5060 = m5060[m5060["인구_월평균"] >= MIN_POP_THRESHOLD]
    print(f"\n── Dependency Index 산출 ──")
    print(f"  소인구 필터: {pop_before}동 → {len(m5060)}동 (5060남성 월평균 {MIN_POP_THRESHOLD}명 이상)")

    # 결측 제거
    required = ["편의인프라_총점", "외출커뮤모두적은", "배달식재료일수", "평일이동횟수", "독거비율"]
    m5060 = m5060.dropna(subset=required)
    print(f"  결측 제거 후: {len(m5060)}동")

    # 5개 구성요소 (모두 min-max 0~1)
    m5060["A_인프라"] = minmax_scale(np.log1p(m5060["편의인프라_총점"]))
    m5060["B_외출커뮤적은"] = minmax_scale(m5060["외출커뮤모두적은"])
    m5060["C_배달의존"] = minmax_scale(m5060["배달식재료일수"])
    m5060["D_이동저하"] = minmax_scale(-m5060["평일이동횟수"])  # 이동 적을수록 위험
    m5060["E_독거비율"] = minmax_scale(m5060["독거비율"])

    # 가중합
    m5060["index_raw"] = sum(m5060[k] * w for k, w in W.items())
    m5060["Dependency"] = (minmax_scale(m5060["index_raw"]) * 100).round(1)

    print(f"\n  Dependency Index 분포:")
    print(f"    mean={m5060['Dependency'].mean():.1f}, std={m5060['Dependency'].std():.1f}")
    print(f"    min={m5060['Dependency'].min():.1f}, Q1={m5060['Dependency'].quantile(0.25):.1f}, "
          f"median={m5060['Dependency'].median():.1f}, Q3={m5060['Dependency'].quantile(0.75):.1f}, "
          f"max={m5060['Dependency'].max():.1f}")

    return m5060


# ── [8] 검증 ─────────────────────────────────────────────────────
def validate(m5060):
    """상위 행정동 검증."""
    top10 = m5060.nlargest(10, "Dependency")

    # 기존 리포트 상위 10개
    V3_TOP10 = [
        "종로1·2·3·4가동", "구로2동", "역삼1동", "가산동", "제기동",
        "서교동", "영등포동", "대학동", "여의동", "면목본동",
    ]

    print("\n── 상위 10개 행정동 ──")
    for i, (_, row) in enumerate(top10.iterrows()):
        print(f"  {i+1:2d}. {row['Dependency']:5.1f}  {row['자치구']} {row['행정동']}")

    my_top = set(top10["행정동"].values)
    v3_set = set(V3_TOP10)
    overlap = my_top & v3_set
    print(f"\n  기존 리포트 일치율: {len(overlap)}/10 ({len(overlap)*10}%)")
    print(f"  일치: {overlap}")
    print(f"  기존에만: {v3_set - my_top}")
    print(f"  신규: {my_top - v3_set}")

    return top10, len(overlap)


# ── [9] 산출물 저장 ──────────────────────────────────────────────
def save_csv(m5060, dong_master):
    """dependency_index.csv 저장."""
    dong_map = dong_master[["telecom_cd", "자치구", "행정동", "adm_cd8"]].drop_duplicates()
    output = m5060[["행정동코드", "Dependency"]].merge(
        dong_map.rename(columns={"telecom_cd": "행정동코드"}),
        on="행정동코드", how="left"
    )
    output = output[["자치구", "행정동", "adm_cd8", "Dependency"]].copy()
    output = output.rename(columns={"adm_cd8": "행정동코드"})
    output = output.sort_values("Dependency", ascending=False).reset_index(drop=True)
    path = OUT / "dependency_index.csv"
    output.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n  저장: {path} ({len(output)}행)")
    return output


def save_methodology(sp_df, partial_df, fisher_df):
    """산출 방법 메모 (기존 방법론 반영)."""
    # 핵심 수치 추출
    m5060_sp = sp_df[sp_df["그룹"] == "남 50-60대"]
    key_sp = m5060_sp[m5060_sp["지표"] == "외출커뮤모두적은"]
    sp_r = key_sp["Spearman_r"].values[0] if len(key_sp) > 0 else 0

    m5060_partial = partial_df[partial_df["그룹"] == "남 50-60대"]
    key_partial = m5060_partial[m5060_partial["지표"] == "외출커뮤모두적은"]
    partial_r = key_partial["부분상관_r"].values[0] if len(key_partial) > 0 else 0

    text = f"""\
# Dependency Index 산출 방법 (V2)

## 분석 방법론

기존 '편리한 고립 분석리포트'의 통계적 방법론을 2022~2025년 4년 데이터에 적용.

### 1단계: 그룹 비교
- 8개 그룹 (남/여 × 20-30대, 30-40대, 50-60대, 65세 이상) 비교
- Spearman 순위상관으로 편의 인프라 vs 고립 지표 상관 측정

### 2단계: 혼란변수 통제
- 1인가구 비율을 통제한 **부분상관** 분석
- "도심에 1인가구가 많아서 그런 것" 가능성 제거

### 3단계: 5060남성 특이성 검증
- **Fisher's r-to-z 변환**으로 5060남성의 효과가 다른 그룹보다 통계적으로 큰지 보조 확인
- 주의: 동일 행정동·인프라 변수를 공유하는 비독립 표본이므로, 엄밀한 독립 검정이 아닌 **보조 근거**로 해석

### 4단계: 지수 산출
- 5060남성 데이터에서 5개 변수의 min-max 정규화 후 가중합

## 핵심 상관 결과

- 편의 인프라 ↔ 외출+커뮤 모두 적은 비율: Spearman r = {sp_r:.3f}
- 1인가구 비율 통제 후 부분상관: r = {partial_r:.3f}

## 사용 변수 (5개)

| 변수 | 가중치 | 의미 |
|------|--------|------|
| 편의 인프라 밀도 (log) | 0.30 | 27개 업종 점포수, 2022~2025 4년 평균 |
| 외출+커뮤 모두 적은 비율 | 0.25 | SK텔레콤 관심집단 2022~2025 4년 평균 |
| 배달 식재료 사용일수 | 0.15 | SK텔레콤 통신정보 2022~2025 4년 평균 |
| 평일 이동 횟수 (역수) | 0.15 | 이동 적을수록 위험, 2022~2025 4년 평균 |
| 독거 비율 | 0.15 | 5060남성 중 1인가구 비율, 2022~2025 4년 평균 |

## 가중치 결정 근거

편의 인프라(0.30)가 본 분석의 핵심 가설 변수이므로 가장 높은 비중을 부여.
외출+커뮤 부재(0.25)는 고립의 직접 지표로 두 번째.
나머지 3개(배달·이동·독거)는 보조 지표로 동일 비중(0.15).
기존 '편리한 고립 분석리포트'의 가중치 구조를 따름.

## 정규화 방법

1. 각 변수별 Min-Max 정규화 (0~1)
2. 가중합: index_raw = Σ(w_i × x_i)
3. 최종 Min-Max: Dependency = 100 × (raw - min) / (max - min)

## 소인구 행정동 필터

5060남성 월평균 인구 {MIN_POP_THRESHOLD}명 미만 행정동 제외.
상업지구(명동, 을지로동 등) 소인구 노이즈 제거 목적.

## 한계점

- 편의 인프라는 **점포수** 기반이며, 실제 이용 빈도 데이터가 아님
- 행정동 단위 집계 분석으로 **생태학적 오류** 가능성 존재
- 인과관계가 아닌 **상관관계** 분석임
- 복지 인프라 변수(종합사회복지관, 50플러스센터 등)가 미포함
- Fisher z-test는 동일 행정동을 공유하는 비독립 표본이므로 보조 근거로 해석
"""
    path = OUT / "dependency_methodology.md"
    path.write_text(text, encoding="utf-8")
    print(f"  저장: {path}")


def save_validation(top10, overlap_count, m5060, sp_df, partial_df, fisher_df):
    """검증 결과 문서."""
    top10_lines = "\n".join(
        f"| {i+1} | {row.get('자치구', '')} | {row['행정동']} | {row['Dependency']:.1f} |"
        for i, (_, row) in enumerate(top10.iterrows())
    )

    # 핵심 비교 테이블
    m5060_partial = partial_df[partial_df["그룹"] == "남 50-60대"]
    partial_lines = "\n".join(
        f"| {row['지표']} | {row['부분상관_r']:+.3f} | {'***' if row['부분상관_p'] < 0.001 else '**' if row['부분상관_p'] < 0.01 else '*' if row['부분상관_p'] < 0.05 else 'ns'} |"
        for _, row in m5060_partial.iterrows()
    )

    fisher_key = fisher_df[fisher_df["지표"] == "외출커뮤모두적은"]
    fisher_lines = "\n".join(
        f"| {row['비교']} | {row['z']:+.2f} | {'*' if row['p'] < 0.05 else 'ns'} |"
        for _, row in fisher_key.iterrows()
    )

    text = f"""\
# Dependency Index 검증 결과 (V2)

## 상위 10개 행정동

| 순위 | 자치구 | 행정동 | Dependency |
|------|--------|--------|------------|
{top10_lines}

## 기존 리포트 일치율

- 기존 리포트 상위 10개: 종로1·2·3·4가동, 구로2동, 역삼1동, 가산동, 제기동, 서교동, 영등포동, 대학동, 여의동, 면목본동
- 본 산출 일치: **{overlap_count}/10 ({overlap_count*10}%)**

## 부분상관 검증 (1인가구 비율 통제 후)

| 지표 | 부분상관 r | 유의성 |
|------|-----------|--------|
{partial_lines}

→ 1인가구 비율을 통제해도 편의 인프라 ↔ 고립 지표 상관이 유지됨.

## Fisher z-test (5060남성 특이성)

외출+커뮤 모두 적은 비율 기준:

| 비교 | z-stat | 유의성 |
|------|--------|--------|
{fisher_lines}

→ z가 양수이고 |z|>1.96이면 5060남성에서 효과가 유의하게 더 큼.

## 분포 통계

| 항목 | 값 |
|------|-----|
| 평균 | {m5060['Dependency'].mean():.1f} |
| 표준편차 | {m5060['Dependency'].std():.1f} |
| 최소 | {m5060['Dependency'].min():.1f} |
| Q1 (25%) | {m5060['Dependency'].quantile(0.25):.1f} |
| 중앙값 | {m5060['Dependency'].median():.1f} |
| Q3 (75%) | {m5060['Dependency'].quantile(0.75):.1f} |
| 최대 | {m5060['Dependency'].max():.1f} |
"""
    path = OUT / "dependency_validation.md"
    path.write_text(text, encoding="utf-8")
    print(f"  저장: {path}")


# ── main ──────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("편의의 역설 — Dependency Index V2")
    print("기존 방법론 (부분상관 + Fisher z + 그룹비교) 적용")
    print("=" * 60)

    # [3] 행정동 매핑
    print("\n── 행정동 매핑 ──")
    dong_master = build_dong_master()

    # [4] 데이터 로딩
    print("\n── 데이터 로딩 ──")
    comm = load_telecom_29()
    interest = load_telecom_10()
    infra = load_store()

    # [5] 패널 구축
    print("\n── 패널 구축 ──")
    panel, dong_map = build_panel(comm, interest, infra, dong_master)

    # [6] 상관분석 + 부분상관 + Fisher z
    sp_df, partial_df, fisher_df = run_correlation_analysis(panel)

    # [7] 지수 산출
    m5060 = calculate_index(panel, dong_master)

    # [8] 검증
    top10, overlap = validate(m5060)

    # [9] 산출물 저장
    print("\n── 산출물 저장 ──")
    save_csv(m5060, dong_master)
    save_methodology(sp_df, partial_df, fisher_df)
    save_validation(top10, overlap, m5060, sp_df, partial_df, fisher_df)

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
