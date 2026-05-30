"""
미래 고위험군 전이 예측 — Feature Engineering (Stage 1)
=========================================================
4년치 통신 패널(월별)에서 행정동별 시계열 특성을 추출하여
Dependency Index 전이 예측 모델의 입력 피처를 만든다.

분석 단위  : 행정동 (소인구 필터 후 ~419개)
시계열     : 월별 데이터 2022.01 ~ 2025.12 (최대 48개월)
타겟 그룹  : 남 50-60대 1인가구

피처 그룹:
  A. Level    — 현재 절대 수준 (최근 6개월 평균)
  B. Trend    — 방향성 (OLS slope, 핵심)
  C. Accel    — 가속도 (slope 변화)
  D. Volatility — 불안정성 (rolling std)
  E. Delta    — 최근 vs 초기 변화량
  F. Infra    — 편의 인프라 트렌드 (분기별 점포수)
  G. Annual   — 연도별 Dependency 점수 궤적

레이블 정의 (두 가지):
  label_slope : 연도별 Dependency slope > 0 이고 상위 50%에 해당 (빠른 악화)
  label_entry : 2024~2025 Dependency >= Q75(2022) 이면서 2022 < Q75 (Q1 진입)

산출물:
  Outputs/전이예측/transition_features.csv
  Outputs/전이예측/yearly_dependency.csv
  Outputs/전이예측/transition_label_report.md
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import linregress
from sklearn.preprocessing import minmax_scale
import warnings
warnings.filterwarnings("ignore")

# ── 설정 ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data"
OUT  = ROOT / "Outputs" / "전이예측"
OUT.mkdir(parents=True, exist_ok=True)

DEP_OUT   = ROOT / "Outputs" / "편의의 역설"
STORE_DIR = DATA / "서울시 상권분석서비스"

TELECOM_29_DIRS = {y: DATA / "통신정보 데이터" / f"{y}_29개_통신정보"   for y in range(2022, 2026)}
TELECOM_10_DIRS = {y: DATA / "통신정보 데이터" / f"{y}_10개_관심집단_수" for y in range(2022, 2026)}

MIN_POP  = 500   # 5060남 월평균 인구 하한
YEARS    = [2022, 2023, 2024, 2025]

CONV_ITEMS = [
    "편의점", "슈퍼마켓", "반찬가게", "패스트푸드점", "분식전문점",
    "제과점", "청과상", "미곡판매", "육류판매", "수산물판매",
    "한식음식점", "중식음식점", "일식음식점", "양식음식점",
    "치킨전문점", "커피-음료", "호프-간이주점",
    "세탁소", "미용실", "부동산중개업",
    "PC방", "노래방", "DVD방", "당구장", "비디오/서적임대", "전자게임장",
    "전자상거래업",
]

W_DEP = {          # Dependency 가중치 (build_dependency_index.py와 동일)
    "A_인프라":       0.30,
    "B_외출커뮤적은": 0.25,
    "C_배달의존":     0.15,
    "D_이동저하":     0.15,
    "E_독거비율":     0.15,
}


# ── [1] 유틸리티 ─────────────────────────────────────────────────────
def _parse_month(path: Path) -> int | None:
    """파일명에서 월 추출. 실패하면 None."""
    name = path.stem
    patterns = [
        r'(\d{4})년\s*(\d{1,2})월',    # 2022년01월
        r'_(\d{1,2})월',                # _01월
        r'(?:[-_])(\d{2})(?:[-_])',     # _01_ or -01-
        r'(?:M|m)(\d{2})',              # M01
        r'(\d{6})',                      # 202201 (연+월 6자리)
    ]
    for pat in patterns:
        m = re.search(pat, name)
        if m:
            val = int(m.group(m.lastindex))
            if pat.endswith(r'(\d{6})'):  # YYYYMM 형태
                val = val % 100
            if 1 <= val <= 12:
                return val
    return None


def _label_group(row) -> str:
    g   = "남" if row.get("성별", 0) == 1 else "여"
    age = row.get("연령대", 0)
    if   20 <= age <= 30: return f"{g} 20-30대"
    elif 35 <= age <= 45: return f"{g} 30-40대"
    elif 50 <= age <= 60: return f"{g} 50-60대"
    elif 65 <= age <= 75: return f"{g} 65세 이상"
    return f"{g} 기타"


def _ols_slope(series: pd.Series) -> float:
    """시계열 OLS slope. 길이 < 3이면 NaN."""
    s = series.dropna()
    if len(s) < 2:
        return np.nan
    x = np.arange(len(s), dtype=float)
    slope, *_ = linregress(x, s.values)
    return slope


def _safe_minmax(arr: np.ndarray) -> np.ndarray:
    """NaN 무시 minmax. 전체 NaN이면 0.5 반환."""
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0 or finite.max() == finite.min():
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - finite.min()) / (finite.max() - finite.min())


# ── [2] 월별 통신 데이터 로딩 (시간 인덱스 포함) ─────────────────────
def load_monthly_m5060() -> pd.DataFrame:
    """5060남 월별 통신 패널. 캐시 우선."""
    CACHE = OUT / "_monthly_m5060_cache.pkl"
    if CACHE.exists():
        df = pd.read_pickle(CACHE)
        print(f"  월별 패널 캐시: {len(df)}행  ({df['ym'].nunique()}개 월)")
        return df

    print("  월별 데이터 로딩 중 (최초 실행, 시간 소요)...")
    col_rename = {"행정동명": "행정동", "총인구": "총인구수"}
    cols_29 = [
        "행정동코드", "자치구", "행정동", "성별", "연령대",
        "총인구수", "1인가구수",
        "평일 총 이동 횟수", "배달_식재료 서비스 사용일수",
    ]
    cols_10 = [
        "행정동코드", "자치구", "행정동", "성별", "연령대",
        "외출-커뮤니케이션이 모두 적은 집단(전체)",
    ]

    frames29, frames10 = [], []

    for year, dir29 in TELECOM_29_DIRS.items():
        if not dir29.exists():
            print(f"    [SKIP] {dir29} 없음")
            continue
        files = sorted(f for f in dir29.glob("*.xlsx") if not f.name.startswith("~$"))
        for idx, f in enumerate(files):
            month = _parse_month(f)
            if month is None:
                month = idx + 1  # 정렬 순서로 대체
            try:
                df = pd.read_excel(f).rename(columns=col_rename)
                df = df[[c for c in cols_29 if c in df.columns]].copy()
                df["그룹"] = df.apply(_label_group, axis=1)
                df = df[df["그룹"] == "남 50-60대"]
                df["year"], df["month"] = year, month
                df["ym"] = year * 100 + month
                frames29.append(df)
            except Exception as e:
                print(f"    [WARN] {f.name}: {e}")

    for year, dir10 in TELECOM_10_DIRS.items():
        if not dir10.exists():
            continue
        files = sorted(f for f in dir10.glob("*.xlsx") if not f.name.startswith("~$"))
        for idx, f in enumerate(files):
            month = _parse_month(f)
            if month is None:
                month = idx + 1
            try:
                df = pd.read_excel(f).rename(columns=col_rename)
                df = df[[c for c in cols_10 if c in df.columns]].copy()
                df["그룹"] = df.apply(_label_group, axis=1)
                df = df[df["그룹"] == "남 50-60대"]
                df["year"], df["month"] = year, month
                df["ym"] = year * 100 + month
                frames10.append(df)
            except Exception as e:
                print(f"    [WARN] {f.name}: {e}")

    comm = pd.concat(frames29, ignore_index=True) if frames29 else pd.DataFrame()
    interest = pd.concat(frames10, ignore_index=True) if frames10 else pd.DataFrame()

    KEY = ["행정동코드", "자치구", "행정동", "year", "month", "ym"]

    # 연령대별 합산 (50, 55, 60대 → 행정동+월 기준 합산)
    comm_agg = (
        comm.groupby(KEY)
        .agg(
            인구=("총인구수", "sum"),
            독거인구=("1인가구수", "sum"),
            평일이동횟수=("평일 총 이동 횟수", "mean"),
            배달식재료일수=("배달_식재료 서비스 사용일수", "mean"),
        )
        .reset_index()
    )
    comm_agg["독거비율"] = comm_agg["독거인구"] / comm_agg["인구"].replace(0, np.nan)

    absence_col = "외출-커뮤니케이션이 모두 적은 집단(전체)"
    if absence_col in interest.columns:
        int_agg = (
            interest.groupby(KEY)[absence_col]
            .mean()
            .reset_index()
            .rename(columns={absence_col: "외출커뮤적은"})
        )
        merged = comm_agg.merge(int_agg, on=KEY, how="left")
    else:
        merged = comm_agg
        merged["외출커뮤적은"] = np.nan

    merged.to_pickle(CACHE)
    print(f"  월별 패널 저장: {len(merged)}행  ({merged['ym'].nunique()}개 월)")
    return merged


# ── [3] 분기별 편의 인프라 로딩 ──────────────────────────────────────
def load_quarterly_store() -> pd.DataFrame:
    """분기별 행정동 편의 인프라 점포수. 캐시 우선."""
    CACHE = OUT / "_quarterly_store_cache.pkl"
    if CACHE.exists():
        df = pd.read_pickle(CACHE)
        print(f"  분기별 상권 캐시: {len(df)}행")
        return df

    dfs = []
    for pat in ["*점포*2022*.csv", "*점포*2023*.csv", "*점포*2024*.csv", "*점포*2025*.csv"]:
        for f in STORE_DIR.glob(pat):
            try:
                dfs.append(pd.read_csv(f, encoding="cp949"))
            except UnicodeDecodeError:
                dfs.append(pd.read_csv(f, encoding="utf-8-sig"))

    if not dfs:
        print("  [WARN] 상권 데이터 없음")
        return pd.DataFrame()

    stores = pd.concat(dfs, ignore_index=True)
    stores = stores[stores["서비스_업종_코드_명"].isin(CONV_ITEMS)]

    qcode_col = "기준_년분기_코드"
    dong_col  = "행정동_코드"
    cnt_col   = "점포_수"

    by_q = (
        stores.groupby([qcode_col, dong_col])[cnt_col]
        .sum()
        .reset_index()
        .rename(columns={qcode_col: "yyqq", dong_col: "행정동코드8", cnt_col: "점포수"})
    )
    # yyqq: 20221 → year=2022, quarter=1
    by_q["year"]    = by_q["yyqq"] // 10
    by_q["quarter"] = by_q["yyqq"] % 10

    by_q.to_pickle(CACHE)
    print(f"  분기별 상권 저장: {len(by_q)}행")
    return by_q


# ── [4] 연도별 Dependency 점수 계산 ──────────────────────────────────
def compute_yearly_dependency(monthly: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    """각 연도(2022~2025)별로 Dependency 점수를 산출.

    전체 연도 풀링 후 전역 정규화 → 연도 간 점수 직접 비교 가능.
    (연도별 정규화 시 2022점수=75와 2025점수=75가 다른 의미가 됨을 방지)
    """
    print("\n── [4] 연도별 Dependency 산출 (전역 정규화) ──")

    KEY = ["행정동코드", "자치구", "행정동"]

    if not store.empty:
        store_yearly = (
            store.groupby(["year", "행정동코드8"])["점포수"]
            .mean()
            .reset_index()
            .rename(columns={"행정동코드8": "행정동코드_store"})
        )
    else:
        store_yearly = pd.DataFrame()

    # Step 1: 연도별 raw 집계 (정규화 없이)
    year_frames = []
    for year in YEARS:
        sub = monthly[monthly["year"] == year].copy()
        if sub.empty:
            print(f"  {year}: 데이터 없음 → skip")
            continue

        agg = (
            sub.groupby(KEY)
            .agg(
                인구_연평균=("인구", "mean"),
                독거비율=("독거비율", "mean"),
                평일이동횟수=("평일이동횟수", "mean"),
                배달식재료일수=("배달식재료일수", "mean"),
                외출커뮤적은=("외출커뮤적은", "mean"),
            )
            .reset_index()
        )
        agg["year"] = year

        before = len(agg)
        agg = agg[agg["인구_연평균"] >= MIN_POP]
        print(f"  {year}: {before}동 → 소인구 필터 후 {len(agg)}동")

        if not store_yearly.empty:
            sy = store_yearly[store_yearly["year"] == year].copy()
            agg["_cd_str"] = agg["행정동코드"].astype(str).str.zfill(8)
            sy["_cd_str"]  = sy["행정동코드_store"].astype(str).str.zfill(8)
            agg = agg.merge(sy[["_cd_str", "점포수"]], on="_cd_str", how="left").drop(columns=["_cd_str"])
        else:
            agg["점포수"] = np.nan

        agg = agg.dropna(subset=["외출커뮤적은", "배달식재료일수", "평일이동횟수", "독거비율"])
        if len(agg) < 10:
            print(f"  {year}: 유효 행정동 {len(agg)}개 → 너무 적어 skip")
            continue

        year_frames.append(agg)

    if not year_frames:
        raise RuntimeError("연도별 Dependency 계산 실패 — 월별 데이터 확인 필요")

    # Step 2: 전체 연도 풀링 후 전역 정규화
    pooled = pd.concat(year_frames, ignore_index=True)
    pooled["A"] = _safe_minmax(np.log1p(pooled["점포수"].fillna(0).values))
    pooled["B"] = _safe_minmax(pooled["외출커뮤적은"].values)
    pooled["C"] = _safe_minmax(pooled["배달식재료일수"].values)
    pooled["D"] = _safe_minmax(-pooled["평일이동횟수"].values)
    pooled["E"] = _safe_minmax(pooled["독거비율"].values)
    pooled["dep_raw"] = (
        pooled["A"] * W_DEP["A_인프라"] +
        pooled["B"] * W_DEP["B_외출커뮤적은"] +
        pooled["C"] * W_DEP["C_배달의존"] +
        pooled["D"] * W_DEP["D_이동저하"] +
        pooled["E"] * W_DEP["E_독거비율"]
    )
    pooled["dep_score"] = (_safe_minmax(pooled["dep_raw"].values) * 100).round(1)
    print(f"\n  전역 정규화 완료 (N={len(pooled)}, 연도 간 점수 직접 비교 가능)")

    # Step 3: 연도별 pivot → wide format
    base = pooled[KEY].drop_duplicates().copy()
    for year in YEARS:
        yr = pooled[pooled["year"] == year][KEY + ["dep_score"]].rename(
            columns={"dep_score": f"dep_{year}"})
        if len(yr):
            base = base.merge(yr, on=KEY, how="outer")

    dep_cols = [c for c in base.columns if c.startswith("dep_")]
    avail = [c for c in dep_cols if base[c].notna().sum() > 50]
    print(f"  연도별 Dependency 컬럼: {avail}")
    print(f"  유효 행정동 수: {base[avail].dropna().shape[0]}")

    return base, year_frames


# ── [5] 시계열 피처 추출 ──────────────────────────────────────────────
def extract_time_features(monthly: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    """행정동별 시계열 피처 추출 (A~F 그룹).

    각 행정동에 대해 48개월 시계열을 분석하여 ~25개 피처 벡터를 생성.
    """
    print("\n── [5] 시계열 피처 추출 ──")

    KEY = ["행정동코드", "자치구", "행정동"]
    VARS = ["외출커뮤적은", "배달식재료일수", "평일이동횟수", "독거비율", "인구"]

    # 인프라 분기별 slope (행정동코드8 기준)
    infra_slope_map = {}
    if not store.empty:
        for cd, grp in store.groupby("행정동코드8"):
            grp_s = grp.sort_values("yyqq")
            infra_slope_map[cd] = {
                "infra_mean":   grp_s["점포수"].mean(),
                "infra_slope":  _ols_slope(grp_s["점포수"]),
                "infra_recent": grp_s["점포수"].iloc[-1] if len(grp_s) else np.nan,
                "infra_delta":  (grp_s["점포수"].iloc[-4:].mean() -
                                 grp_s["점포수"].iloc[:4].mean()),
            }

    dong_groups = monthly.sort_values("ym").groupby(KEY)
    feature_rows = []

    for (cd, gu, dong), grp in dong_groups:
        grp = grp.sort_values("ym")
        n   = len(grp)

        row = {"행정동코드": cd, "자치구": gu, "행정동": dong, "n_months": n}

        # ── A. Level (최근 6개월 평균) ──────────────────────────────
        recent = grp.tail(6)
        for v in VARS:
            if v in grp.columns:
                row[f"level_{v}"] = recent[v].mean()

        # ── B. Trend (전체 기간 OLS slope) ─────────────────────────
        for v in VARS:
            if v in grp.columns:
                row[f"slope_{v}"] = _ols_slope(grp[v])

        # 이동횟수는 감소가 위험 → 부호 반전하여 "위험도 slope"로 변환
        if "slope_평일이동횟수" in row:
            row["slope_이동저하"] = -row.pop("slope_평일이동횟수")

        # ── C. Acceleration (slope 변화율) ──────────────────────────
        half = n // 2
        for v in ["외출커뮤적은", "배달식재료일수", "독거비율"]:
            if v in grp.columns:
                s_early = _ols_slope(grp[v].iloc[:half])
                s_late  = _ols_slope(grp[v].iloc[half:])
                row[f"accel_{v}"] = s_late - s_early  # 양수 = 가속화

        # ── D. Volatility (전체 기간 std) ───────────────────────────
        for v in ["외출커뮤적은", "배달식재료일수"]:
            if v in grp.columns:
                row[f"vol_{v}"] = grp[v].std()

        # ── E. Delta (최근 12개월 평균 vs 초기 12개월 평균) ───────────
        early = grp.head(12)
        late  = grp.tail(12)
        for v in ["외출커뮤적은", "배달식재료일수", "인구"]:
            if v in grp.columns and len(early) >= 3 and len(late) >= 3:
                row[f"delta_{v}"] = late[v].mean() - early[v].mean()

        # ── F. Infra (편의 인프라 트렌드) ──────────────────────────
        # 행정동코드 8자리로 매핑 시도
        cd8_str = str(cd).zfill(8)
        if cd8_str in infra_slope_map:
            row.update(infra_slope_map[cd8_str])
        else:
            # 끝 7자리 일치 시도 (telecom 7자리 vs 상권 8자리)
            for k in infra_slope_map:
                if k.endswith(str(cd)[-7:]):
                    row.update(infra_slope_map[k])
                    break

        feature_rows.append(row)

    feat = pd.DataFrame(feature_rows)
    print(f"  추출 완료: {len(feat)}행 × {len(feat.columns)}열")
    return feat


# ── [6] 레이블 정의 ──────────────────────────────────────────────────
def define_labels(feat: pd.DataFrame, yearly: pd.DataFrame) -> pd.DataFrame:
    """전이 레이블 두 가지 정의.

    label_slope : 연도별 Dependency slope가 양수 + 상위 50% (빠른 악화)
    label_entry : 최신 연도 Dependency >= Q75(기준연도) 이면서 기준연도 < Q75
    """
    print("\n── [6] 레이블 정의 ──")
    KEY = ["행정동코드", "자치구", "행정동"]

    merged = feat.merge(yearly, on=KEY, how="inner")

    dep_cols = sorted([c for c in yearly.columns if c.startswith("dep_20")])
    print(f"  사용 연도: {dep_cols}")

    if len(dep_cols) < 2:
        print("  [WARN] 연도별 Dep 컬럼 부족 — 레이블 생성 불가")
        merged["label_slope"] = np.nan
        merged["label_entry"] = np.nan
        return merged

    # ── label_slope: 연도별 Dependency OLS slope 상위 50% ─────────
    def _row_slope(row):
        vals = [row[c] for c in dep_cols if pd.notna(row[c])]
        if len(vals) < 2:
            return np.nan
        return _ols_slope(pd.Series(vals))

    merged["dep_slope_annual"] = merged.apply(_row_slope, axis=1)
    # 전체 중앙값 대신 양수 slope 중 중앙값 사용 (음수 slope 행정동이 레이블 1이 되는 문제 방지)
    pos_mask = merged["dep_slope_annual"] > 0
    slope_pos_median = (
        merged.loc[pos_mask, "dep_slope_annual"].median()
        if pos_mask.sum() > 0 else 0.0
    )
    merged["label_slope"] = (
        pos_mask & (merged["dep_slope_annual"] >= slope_pos_median)
    ).astype(int)

    # ── label_entry: Q1 진입 (기준연도 Q75 미만 → 최신연도 Q75 이상) ─
    ref_col  = dep_cols[0]   # 2022 기준
    last_col = dep_cols[-1]  # 최신년도

    q75_ref  = merged[ref_col].quantile(0.75)
    not_q1_now   = merged[ref_col] < q75_ref
    entered_q1   = merged[last_col] >= q75_ref
    merged["label_entry"] = (not_q1_now & entered_q1).astype(int)

    # 분포 출력
    n = len(merged)
    n_slope = merged["label_slope"].sum()
    n_entry = merged["label_entry"].sum()
    print(f"\n  label_slope  (빠른 악화): {n_slope}/{n} = {n_slope/n:.1%}")
    print(f"  label_entry  (Q1 진입)  : {n_entry}/{n} = {n_entry/n:.1%}")
    print(f"  두 레이블 동시 해당    : {((merged['label_slope']==1)&(merged['label_entry']==1)).sum()}개")

    return merged


# ── [7] 저장 ─────────────────────────────────────────────────────────
def save_outputs(full: pd.DataFrame, yearly_records: list, yearly_wide: pd.DataFrame):
    """피처 CSV + 연도별 Dependency + 리포트 저장."""

    # 피처 CSV
    feat_cols = (
        ["행정동코드", "자치구", "행정동", "n_months"] +
        [c for c in full.columns if any(c.startswith(p)
         for p in ["level_", "slope_", "accel_", "vol_", "delta_",
                   "infra_", "dep_20", "dep_slope", "label_"])]
    )
    feat_cols = [c for c in feat_cols if c in full.columns]
    feat_df = full[feat_cols].copy()
    feat_df.to_csv(OUT / "transition_features.csv", index=False, encoding="utf-8-sig")
    print(f"\n  저장: transition_features.csv  ({len(feat_df)}행 × {len(feat_df.columns)}열)")

    # 연도별 Dependency CSV
    dep_cols = ["행정동코드", "자치구", "행정동"] + [c for c in yearly_wide.columns if c.startswith("dep_")]
    dep_wide = yearly_wide[[c for c in dep_cols if c in yearly_wide.columns]].copy()
    dep_wide.to_csv(OUT / "yearly_dependency.csv", index=False, encoding="utf-8-sig")
    print(f"  저장: yearly_dependency.csv  ({len(dep_wide)}행)")

    # 리포트
    dep_year_cols = sorted([c for c in dep_wide.columns if c.startswith("dep_20")])
    n_dong = len(feat_df)
    n_feat = len([c for c in feat_df.columns if any(c.startswith(p)
                  for p in ["level_", "slope_", "accel_", "vol_", "delta_", "infra_"])])

    slope_pos  = (feat_df["dep_slope_annual"] > 0).sum() if "dep_slope_annual" in feat_df else "N/A"
    n_label_s  = feat_df["label_slope"].sum() if "label_slope" in feat_df else "N/A"
    n_label_e  = feat_df["label_entry"].sum() if "label_entry" in feat_df else "N/A"

    report = f"""\
# 전이 예측 피처 엔지니어링 결과

## 데이터 요약

| 항목 | 값 |
|------|-----|
| 행정동 수 | {n_dong}개 |
| 월 시계열 | {feat_df["n_months"].median():.0f}개월 (중앙값) |
| 추출 피처 수 | {n_feat}개 |
| 연도별 Dep 컬럼 | {dep_year_cols} |

## 레이블 분포

| 레이블 | 정의 | 해당 수 | 비율 |
|--------|------|---------|------|
| label_slope | 연도별 Dep slope 양수 + 상위 50% | {n_label_s} | {int(n_label_s)/n_dong:.1%} |
| label_entry | 최신연도 Dep >= Q75(기준연도) & 기준연도 < Q75 | {n_label_e} | {int(n_label_e)/n_dong:.1%} |

## 피처 그룹

| 그룹 | 피처 예시 | 의미 |
|------|---------|------|
| A. Level | level_외출커뮤적은 | 최근 6개월 평균 수준 |
| B. Trend | slope_외출커뮤적은 | OLS slope (전체 기간) |
| C. Accel | accel_외출커뮤적은 | 후반부 slope - 전반부 slope |
| D. Volatility | vol_외출커뮤적은 | 전체 std |
| E. Delta | delta_외출커뮤적은 | 최근12개월 평균 - 초기12개월 평균 |
| F. Infra | infra_slope | 편의 인프라 점포수 추세 |
| G. Annual | dep_2022, dep_slope_annual | 연도별 Dep 점수 및 slope |

## 다음 단계

train_transition_model.py 실행:
  - 입력: transition_features.csv
  - 레이블: label_slope (주), label_entry (보조)
  - 모델: Logistic Regression (기준) + XGBoost
"""
    (OUT / "transition_label_report.md").write_text(report, encoding="utf-8")
    print(f"  저장: transition_label_report.md")


# ── main ──────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("전이 예측 — Feature Engineering")
    print("=" * 60)

    print("\n── [2] 월별 통신 데이터 로딩 ──")
    monthly = load_monthly_m5060()

    print("\n── [3] 분기별 상권 데이터 로딩 ──")
    store = load_quarterly_store()

    yearly_wide, yearly_records = compute_yearly_dependency(monthly, store)

    feat = extract_time_features(monthly, store)

    full = define_labels(feat, yearly_wide)

    print("\n── [7] 저장 ──")
    save_outputs(full, yearly_records, yearly_wide)

    print("\n" + "=" * 60)
    print("완료! 다음: python code/train_transition_model.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
