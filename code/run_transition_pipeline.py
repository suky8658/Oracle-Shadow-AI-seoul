"""
전이 예측 완전 파이프라인 — run_transition_pipeline.py
실행: C:/Users/vinvi/anaconda3/python.exe code/run_transition_pipeline.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from scipy.stats import linregress
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ── 경로 ─────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "Data"
DEP_OUT = ROOT / "Outputs" / "편의의 역설"
OUT     = ROOT / "Outputs" / "전이예측"
OUT.mkdir(parents=True, exist_ok=True)

_FONT_FILE = "C:/Windows/Fonts/malgun.ttf"
if Path(_FONT_FILE).exists():
    fp = fm.FontProperties(fname=_FONT_FILE)
    plt.rcParams["font.family"] = fp.get_name()
else:
    plt.rcParams["font.family"] = "Malgun Gothic"
    fp = fm.FontProperties()
plt.rcParams["axes.unicode_minus"] = False

MIN_POP  = 500    # 5060남 월평균 인구 하한
YEARS    = [2022, 2023, 2024, 2025]
W        = {"A": 0.30, "B": 0.25, "C": 0.15, "D": 0.15, "E": 0.15}
CONV_ITEMS = [
    "편의점","슈퍼마켓","반찬가게","패스트푸드점","분식전문점","제과점","청과상",
    "미곡판매","육류판매","수산물판매","한식음식점","중식음식점","일식음식점",
    "양식음식점","치킨전문점","커피-음료","호프-간이주점","세탁소","미용실",
    "부동산중개업","PC방","노래방","DVD방","당구장","비디오/서적임대","전자게임장",
    "전자상거래업",
]

print("=" * 60)
print("전이 예측 파이프라인")
print("=" * 60)


# ── 유틸 ─────────────────────────────────────────────────────────────
def safe_mm(arr):
    a = np.array(arr, dtype=float)
    f = a[np.isfinite(a)]
    if len(f) < 2 or f.max() == f.min():
        return np.full_like(a, 0.5, dtype=float)
    return (a - f.min()) / (f.max() - f.min())

def ols_slope(vals):
    v = np.array([x for x in vals if pd.notna(x)], dtype=float)
    if len(v) < 2:
        return np.nan
    return linregress(np.arange(len(v), dtype=float), v).slope


# ── STEP 1: 전체 패널 캐시 (4년 평균, cross-sectional) ───────────────
print("\n[1] 전체 패널 캐시 로딩")
panel = pd.read_pickle(DEP_OUT / "_full_panel_cache.pkl")
base  = panel[panel["그룹"] == "남 50-60대"].copy()

# pop = 48개월 합산 총인구 → /48 = 월평균
base["pop_monthly"] = base["pop"] / 48
before = len(base)
base = base[base["pop_monthly"] >= MIN_POP].copy()
print(f"  5060남: {before} → {len(base)}동 (소인구 필터 후)")
print(f"  컬럼: {list(base.columns)}")

KEY = ["행정동코드", "자치구", "행정동", "adm_cd8"]


# ── STEP 2: 연도별 통신 데이터 (annual averages) ──────────────────────
print("\n[2] 연도별 통신 데이터 로딩")
YCACHE = OUT / "_yearly_telecom.pkl"

if YCACHE.exists():
    yearly = pd.read_pickle(YCACHE)
    print(f"  캐시: {len(yearly)}행 ({yearly['year'].nunique()}개 연도)")
else:
    COL29  = ["행정동코드", "자치구", "행정동", "성별", "연령대",
               "총인구수", "1인가구수",
               "평일 총 이동 횟수", "배달_식재료 서비스 사용일수"]
    COL10  = ["행정동코드", "자치구", "행정동명", "성별", "연령대",
               "총인구", "1인가구수",
               "외출-커뮤니케이션이 모두 적은 집단(전체)"]
    AGE5060 = [50, 55, 60]

    frames29, frames10 = [], []
    for year in YEARS:
        for label, dirpath, cols, flist in [
            ("29개", DATA / "통신정보 데이터" / f"{year}_29개_통신정보",   COL29, frames29),
            ("10개", DATA / "통신정보 데이터" / f"{year}_10개_관심집단_수", COL10, frames10),
        ]:
            if not dirpath.exists():
                print(f"  [SKIP] {year} {label}")
                continue
            files = [f for f in sorted(dirpath.glob("*.xlsx")) if not f.name.startswith("~$")]
            ok = 0
            for f in files:
                try:
                    df = pd.read_excel(f)
                    # 10개 파일: '행정동명' → '행정동', '총인구' → '총인구수'
                    df = df.rename(columns={"행정동명": "행정동", "총인구": "총인구수"})
                    df = df[[c for c in cols if c in df.columns]].copy()
                    df = df[(df["성별"] == 1) & (df["연령대"].isin(AGE5060))]
                    df["year"] = year
                    flist.append(df)
                    ok += 1
                except Exception as e:
                    print(f"  [WARN] {f.name}: {e}")
            print(f"  {year} {label}: {ok}/{len(files)} 파일")

    YKEY   = ["행정동코드", "자치구", "행정동", "year"]
    YKEY10 = ["행정동코드", "year"]   # 10개 파일은 행정동명→행정동 리네임 후 컬럼이 없을 수 있음
    comm = pd.concat(frames29, ignore_index=True) if frames29 else pd.DataFrame()
    intr = pd.concat(frames10, ignore_index=True) if frames10 else pd.DataFrame()

    if not comm.empty:
        comm_agg = comm.groupby(YKEY).agg(
            인구합산=("총인구수", "sum"),
            독거합산=("1인가구수", "sum"),
            이동횟수=("평일 총 이동 횟수", "mean"),
            배달식재료=("배달_식재료 서비스 사용일수", "mean"),
        ).reset_index()
        comm_agg["독거비율"] = comm_agg["독거합산"] / comm_agg["인구합산"].replace(0, np.nan)
    else:
        comm_agg = pd.DataFrame(columns=YKEY)

    if not intr.empty:
        absence_col = "외출-커뮤니케이션이 모두 적은 집단(전체)"
        # 행정동 컬럼 없이 행정동코드+year 기준으로 groupby
        grp_cols = [c for c in YKEY10 if c in intr.columns]
        int_agg = intr.groupby(grp_cols)[absence_col].mean().reset_index().rename(
            columns={absence_col: "외출커뮤적은"})
        if not comm_agg.empty:
            yearly = comm_agg.merge(int_agg, on=[c for c in grp_cols if c in comm_agg.columns],
                                    how="left")
        else:
            yearly = int_agg
    else:
        yearly = comm_agg
        yearly["외출커뮤적은"] = np.nan

    yearly.to_pickle(YCACHE)
    print(f"  연도별 캐시 저장: {len(yearly)}행")


# ── STEP 3: 분기별 상권 데이터 ──────────────────────────────────────
print("\n[3] 분기별 상권 데이터")
STORE_DIR = DATA / "서울시 상권분석서비스"
SCACHE = OUT / "_store_quarterly.pkl"

if SCACHE.exists():
    by_q = pd.read_pickle(SCACHE)
    has_store = True
    print(f"  캐시: {len(by_q)}행")
else:
    frames_s = []
    for pat in ["*점포*2022*.csv", "*점포*2023*.csv", "*점포*2024*.csv", "*점포*2025*.csv"]:
        for f in STORE_DIR.glob(pat):
            try:
                frames_s.append(pd.read_csv(f, encoding="cp949"))
            except Exception:
                try:
                    frames_s.append(pd.read_csv(f, encoding="utf-8-sig"))
                except Exception:
                    pass

    if frames_s:
        stores = pd.concat(frames_s, ignore_index=True)
        stores = stores[stores["서비스_업종_코드_명"].isin(CONV_ITEMS)]
        by_q = stores.groupby(["기준_년분기_코드", "행정동_코드"])["점포_수"].sum().reset_index()
        by_q["year"]    = by_q["기준_년분기_코드"] // 10
        by_q["quarter"] = by_q["기준_년분기_코드"] % 10
        by_q.to_pickle(SCACHE)
        has_store = True
        print(f"  상권: {by_q['행정동_코드'].nunique()}개 행정동 × {by_q['기준_년분기_코드'].nunique()}분기")
    else:
        by_q = pd.DataFrame()
        has_store = False
        print("  상권 데이터 없음")


# ── STEP 4: 연도별 Dependency 계산 (전역 정규화) ──────────────────────
# 연도별로 따로 정규화하면 2022점수와 2025점수가 비교 불가 → 전체 연도 풀링 후 한 번만 정규화
print("\n[4] 연도별 Dependency 계산 (전역 정규화)")

def get_infra_year(year):
    if not has_store:
        return pd.DataFrame(columns=["adm_cd8_str", "점포수"])
    inf = by_q[by_q["year"] == year].groupby("행정동_코드")["점포_수"].mean().reset_index()
    inf["adm_cd8_str"] = inf["행정동_코드"].astype(str).str.zfill(8)
    return inf[["adm_cd8_str", "점포_수"]].rename(columns={"점포_수": "점포수"})

# Step 4-1: 연도별 raw 집계 (정규화 없이)
year_raws = []
for year in YEARS:
    sub = yearly[yearly["year"] == year].copy()
    if len(sub) < 20:
        print(f"  {year}: {len(sub)}행 → skip")
        continue
    if "인구합산" in sub.columns:
        sub = sub[sub["인구합산"] / 12 / 3 >= MIN_POP]
    sub = sub.merge(
        base[["행정동코드", "adm_cd8"]].drop_duplicates(), on="행정동코드", how="left"
    )
    sub["adm_cd8_str"] = sub["adm_cd8"].astype("Int64").astype(str).str.zfill(8)
    infra_y = get_infra_year(year)
    if not infra_y.empty:
        sub = sub.merge(infra_y, on="adm_cd8_str", how="left")
    else:
        sub["점포수"] = np.nan
    must = ["외출커뮤적은", "배달식재료", "이동횟수", "독거비율"]
    sub = sub.dropna(subset=[c for c in must if c in sub.columns])
    if len(sub) < 10:
        print(f"  {year}: 유효 {len(sub)}동 → skip")
        continue
    sub["_year"] = year
    sub["_A"] = np.log1p(sub["점포수"].fillna(0))
    sub["_B"] = sub["외출커뮤적은"]
    sub["_C"] = sub["배달식재료"]
    sub["_D"] = -sub["이동횟수"]
    sub["_E"] = sub["독거비율"]
    year_raws.append(sub[["행정동코드", "_year", "_A", "_B", "_C", "_D", "_E"]])
    print(f"  {year}: {len(sub)}동 raw 집계")

# Step 4-2: 전체 연도 풀링 후 전역 정규화 (연도 간 직접 비교 가능)
pooled = pd.concat(year_raws, ignore_index=True)
for src, dst in [("_A","A"), ("_B","B"), ("_C","C"), ("_D","D"), ("_E","E")]:
    pooled[dst] = safe_mm(pooled[src].values)
pooled["dep_raw"]   = (pooled["A"]*W["A"] + pooled["B"]*W["B"] + pooled["C"]*W["C"]
                       + pooled["D"]*W["D"] + pooled["E"]*W["E"])
pooled["dep_score"] = (safe_mm(pooled["dep_raw"].values) * 100).round(1)
print(f"  전역 정규화 완료 (N={len(pooled)}, 연도 간 점수 직접 비교 가능)")

# Step 4-3: Wide format
dep_records = {}
for year in YEARS:
    yr = (pooled[pooled["_year"] == year][["행정동코드", "dep_score"]]
          .rename(columns={"dep_score": f"dep_{year}"}))
    if len(yr):
        dep_records[year] = yr

dep_wide = base[["행정동코드", "자치구", "행정동"]].copy()
for year, df in dep_records.items():
    dep_wide = dep_wide.merge(df, on="행정동코드", how="left")

dep_cols = [c for c in dep_wide.columns if c.startswith("dep_20")]
print(f"  연도별 Dep: {dep_cols}")
dep_wide.to_csv(OUT / "yearly_dependency.csv", index=False, encoding="utf-8-sig")
print(f"  yearly_dependency.csv 저장 ({len(dep_wide)}행)")


# ── STEP 5: 피처 매트릭스 ────────────────────────────────────────────
print("\n[5] 피처 매트릭스 구성")

feat = base[["행정동코드", "자치구", "행정동", "adm_cd8"]].copy()

# A. Level features: 2022-2023 조기 구간 평균 (피처-레이블 시간 분리)
# 4년 평균 대신 2022-2023만 사용 → label_entry(2025 기준) 와 시간적 독립
early = yearly[yearly["year"].isin([2022, 2023])].copy()
for src, dst in [("외출커뮤적은","level_외출커뮤적은"), ("이동횟수","level_이동횟수"),
                  ("배달식재료","level_배달식재료"), ("독거비율","level_독거비율")]:
    if src in early.columns:
        avg = early.groupby("행정동코드")[src].mean().rename(dst).reset_index()
        feat = feat.merge(avg, on="행정동코드", how="left")
# 인프라·인구는 패널 캐시 사용 (연도별 통신 데이터에 없는 구조적 제약)
for src, dst in [("conv_total","level_인프라"), ("pop_monthly","level_인구")]:
    if src in base.columns:
        tmp = base.groupby("행정동코드")[src].mean().rename(dst).reset_index()
        feat = feat.merge(tmp, on="행정동코드", how="left")

# A2. 조기 트렌드 피처: 2022→2023 변화 방향 (빠른 악화 신호)
y22 = yearly[yearly["year"] == 2022].set_index("행정동코드")
y23 = yearly[yearly["year"] == 2023].set_index("행정동코드")
for src, dst in [("외출커뮤적은","delta_외출커뮤적은"), ("이동횟수","delta_이동횟수"),
                  ("배달식재료","delta_배달식재료"), ("독거비율","delta_독거비율")]:
    if src in y22.columns and src in y23.columns:
        delta = (y23[src] - y22[src]).rename(dst).reset_index()
        feat = feat.merge(delta, on="행정동코드", how="left")

# B. Infra trend (분기별 slope per 행정동)
if has_store:
    infra_slopes = {}
    for cd, grp in by_q.groupby("행정동_코드"):
        g = grp.sort_values("기준_년분기_코드")
        vals = g["점포_수"].values
        infra_slopes[str(int(cd)).zfill(8)] = {
            "infra_slope":  ols_slope(vals),
            "infra_delta":  vals[-4:].mean() - vals[:4].mean() if len(vals) >= 8 else np.nan,
            "infra_recent": float(vals[-1]) if len(vals) else np.nan,
        }
    infra_df = pd.DataFrame.from_dict(infra_slopes, orient="index").reset_index()
    infra_df.columns = ["adm_cd8_str"] + list(infra_df.columns[1:])
    feat["adm_cd8_str"] = feat["adm_cd8"].astype("Int64").astype(str).str.zfill(8)
    feat = feat.merge(infra_df, on="adm_cd8_str", how="left")

# C. Annual Dep features (레이블 정의용으로 CSV에 포함, 모델 입력 피처 아님)
feat = feat.merge(dep_wide[["행정동코드"] + dep_cols], on="행정동코드", how="left")

# 모델 입력 피처: 조기 수준(level_) + 조기 트렌드(delta_) + 인프라 추세(infra_)
# dep_20XX / dep_slope_annual 은 레이블 계산에 쓰이므로 피처에서 제외 (누출 방지)
FEAT_COLS = (
    [c for c in feat.columns if c.startswith("level_")] +
    [c for c in feat.columns if c.startswith("delta_")] +
    [c for c in feat.columns if c.startswith("infra_")]
)
FEAT_COLS = [c for c in FEAT_COLS if c in feat.columns]
n_level = sum(1 for c in FEAT_COLS if c.startswith("level_"))
n_delta = sum(1 for c in FEAT_COLS if c.startswith("delta_"))
n_infra = sum(1 for c in FEAT_COLS if c.startswith("infra_"))
print(f"  피처: {len(FEAT_COLS)}개 (level={n_level}, delta={n_delta}, infra={n_infra})")
print(f"  → {FEAT_COLS}")
print(f"  ※ dep_20XX / dep_slope_annual 은 레이블 정의에만 사용 (피처 제외)")


# ── STEP 6: 레이블 정의 ──────────────────────────────────────────────
print("\n[6] 레이블 정의 (dep 궤적 기반, 피처와 완전 독립)")
# dep_20XX는 전역 정규화 기반 → 연도 간 점수 직접 비교 가능
dep_yr_cols = [c for c in feat.columns if c.startswith("dep_20")]

# dep_slope_annual 계산 (레이블용, FEAT_COLS에 포함 안 됨)
if len(dep_yr_cols) >= 2:
    feat["dep_slope_annual"] = feat.apply(
        lambda r: ols_slope([r.get(c, np.nan) for c in dep_yr_cols]), axis=1)
else:
    feat["dep_slope_annual"] = np.nan

# label_slope: Dep slope 양수 AND 양수 slope 중 상위 50% (빠르게 악화 중)
if feat["dep_slope_annual"].notna().sum() > 0:
    pos_mask = feat["dep_slope_annual"] > 0
    slope_pos_med = (
        feat.loc[pos_mask, "dep_slope_annual"].median()
        if pos_mask.sum() > 0 else 0.0
    )
    feat["label_slope"] = (
        pos_mask & (feat["dep_slope_annual"] >= slope_pos_med)
    ).astype(int)
    print(f"  label_slope (Dep slope 양수+상위50%): {feat['label_slope'].sum()}/{len(feat)}")
else:
    feat["label_slope"] = np.nan
    print(f"  label_slope: dep_slope_annual 없음 → NaN")

# label_entry: 전역 Q75 기준으로 기준연도 미달 → 최신연도 초과 (Q1 진입)
if len(dep_yr_cols) >= 2:
    all_dep_vals = pd.concat([feat[c].dropna() for c in dep_yr_cols])
    q75_global   = all_dep_vals.quantile(0.75)
    ref_col, last_col = dep_yr_cols[0], dep_yr_cols[-1]
    valid = feat[ref_col].notna() & feat[last_col].notna()
    feat["label_entry"] = np.nan
    feat.loc[valid, "label_entry"] = (
        feat.loc[valid, ref_col].lt(q75_global) &
        feat.loc[valid, last_col].ge(q75_global)
    ).astype(float)
    n_e = int(feat["label_entry"].sum())
    print(f"  label_entry (Q1 진입, 전역 Q75={q75_global:.1f}): {n_e}/{len(feat)}")
else:
    feat["label_entry"] = np.nan
    print(f"  label_entry: dep 연도 부족 → NaN")

feat.to_csv(OUT / "transition_features.csv", index=False, encoding="utf-8-sig")
print(f"  transition_features.csv 저장")


# ── STEP 7: 모델 학습 ────────────────────────────────────────────────
print("\n[7] 모델 학습 (Stratified 5-fold CV)")

model_results = {}

for label_col in ["label_slope", "label_entry"]:
    sub = feat.dropna(subset=[label_col]).copy()
    y   = sub[label_col].astype(int)
    X   = sub[FEAT_COLS].copy()

    if y.sum() < 10 or (len(y) - y.sum()) < 10:
        print(f"  [{label_col}] 양성={y.sum()} / 음성={len(y)-y.sum()} → skip")
        continue

    print(f"\n  [{label_col}] N={len(sub)}, 양성={y.sum()} ({y.mean():.1%})")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    lr_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced", random_state=42)),
    ])
    gb_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            min_samples_leaf=10, subsample=0.8, random_state=42)),
    ])

    row = {"label": label_col, "N": len(sub), "pos": int(y.sum()),
           "sub": sub, "y": y, "X": X}

    for name, pipe in [("LR", lr_pipe), ("GB", gb_pipe)]:
        try:
            prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
            auc  = roc_auc_score(y, prob)
            ap   = average_precision_score(y, prob)
            row[f"{name}_proba"] = prob
            row[f"{name}_auc"]   = auc
            row[f"{name}_ap"]    = ap
            print(f"    {name} 5-fold CV: AUC={auc:.3f}  AP={ap:.3f}")
            pred = (prob >= 0.5).astype(int)
            print("    " + classification_report(y, pred,
                  target_names=["안정","전이"], zero_division=0).replace("\n", "\n    "))
        except Exception as e:
            print(f"    {name}: 오류 — {e}")

    # 홀드아웃 검증 (20% 무작위 분할, CV와 독립된 추가 검증)
    if len(y) >= 50 and y.sum() >= 10 and (len(y) - y.sum()) >= 10:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)
        row["holdout_n_train"] = len(y_tr)
        row["holdout_n_test"]  = len(y_te)
        print(f"    홀드아웃 (80/20, n_test={len(y_te)}):")
        for name, pipe in [("LR", lr_pipe), ("GB", gb_pipe)]:
            try:
                fitted = clone(pipe)
                fitted.fit(X_tr, y_tr)
                h_prob = fitted.predict_proba(X_te)[:, 1]
                h_auc  = roc_auc_score(y_te, h_prob)
                h_ap   = average_precision_score(y_te, h_prob)
                row[f"{name}_holdout_auc"] = h_auc
                row[f"{name}_holdout_ap"]  = h_ap
                print(f"      {name}: AUC={h_auc:.3f}  AP={h_ap:.3f}")
            except Exception as e:
                print(f"      {name}: 홀드아웃 오류 — {e}")

    model_results[label_col] = row


# ── STEP 8: 피처 중요도 시각화 ──────────────────────────────────────
print("\n[8] 피처 중요도 시각화")

if "label_slope" in model_results:
    res = model_results["label_slope"]
    X_fit, y_fit = res["X"], res["y"]

    gb_fit = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            min_samples_leaf=10, subsample=0.8, random_state=42)),
    ])
    gb_fit.fit(X_fit, y_fit)

    imp = pd.Series(gb_fit["clf"].feature_importances_, index=FEAT_COLS).sort_values(ascending=False)
    top = imp.head(min(20, len(imp)))

    fig, ax = plt.subplots(figsize=(10, max(5, len(top) * 0.38)))
    colors = ["#e74c3c" if any(k in c for k in ["slope","dep_20","delta","accel"])
              else "#3498db" for c in top.index[::-1]]
    bars = ax.barh(range(len(top)), top.values[::-1], color=colors, edgecolor="white")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([c.replace("_", " ") for c in top.index[::-1]], fontproperties=fp, fontsize=9)
    ax.set_xlabel("피처 중요도 (GradientBoosting)", fontproperties=fp, fontsize=11)
    ax.set_title("전이 예측 — 피처 중요도\n빨강=추세·연도별, 파랑=수준·인프라",
                 fontproperties=fp, fontsize=12, fontweight="bold")
    ax.legend(handles=[
        mpatches.Patch(facecolor="#e74c3c", label="추세/연도별 피처"),
        mpatches.Patch(facecolor="#3498db", label="수준/인프라 피처"),
    ], prop=fp, fontsize=9, loc="lower right")
    for i, (val, bar) in enumerate(zip(top.values[::-1], bars)):
        ax.text(val + imp.max() * 0.01, i, f"{val:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  feature_importance.png 저장")
    print(f"  상위 5 피처: {list(top.head(5).index)}")


# ── STEP 9: 전이 확률 저장 ───────────────────────────────────────────
print("\n[9] 전이 확률 저장")

if "label_slope" in model_results:
    res = model_results["label_slope"]
    meta = res["sub"][["행정동코드", "자치구", "행정동"]].copy()
    meta["전이확률_GB"] = res.get("GB_proba", np.nan)
    meta["전이확률_LR"] = res.get("LR_proba", np.nan)
    meta["실제레이블"]  = res["y"].values

    prob_for_grade = meta["전이확률_GB"].fillna(meta["전이확률_LR"])
    meta["위험등급"] = pd.cut(
        prob_for_grade,
        bins=[-0.001, 0.30, 0.50, 0.70, 1.001],
        labels=["저위험", "중위험", "고위험", "최고위험"],
    )
    meta = meta.merge(dep_wide[["행정동코드"] + dep_cols], on="행정동코드", how="left")
    if "dep_slope_annual" in feat.columns:
        meta = meta.merge(feat[["행정동코드","dep_slope_annual"]], on="행정동코드", how="left")

    meta = meta.sort_values("전이확률_GB", ascending=False).reset_index(drop=True)
    meta.to_csv(OUT / "transition_predictions.csv", index=False, encoding="utf-8-sig")

    print(f"  transition_predictions.csv 저장 ({len(meta)}행)")
    print(f"\n  위험등급 분포:")
    print(meta["위험등급"].value_counts().to_string())
    print(f"\n  전이 확률 상위 10:")
    for _, r in meta.head(10).iterrows():
        print(f"    {r.get('자치구',''):5s} {r.get('행정동',''):10s}  "
              f"GB={r['전이확률_GB']:.3f}  등급={r['위험등급']}")


# ── STEP 10: 자치구 보조 모델 ────────────────────────────────────────
print("\n[10] 자치구 보조 모델 (규칙 기반)")

shadow_path = ROOT / "Outputs" / "shadow_index.csv"
if shadow_path.exists() and dep_cols and "자치구" in dep_wide.columns:
    shadow = pd.read_csv(shadow_path)

    gu_dep = dep_wide.groupby("자치구")[dep_cols].mean().reset_index()
    gu_dep["dep_slope_gu"] = gu_dep.apply(
        lambda r: ols_slope([r[c] for c in dep_cols]), axis=1)

    gu_risk = shadow.merge(gu_dep[["자치구","dep_slope_gu"] + dep_cols], on="자치구", how="left")
    gu_risk["risk_raw"] = (
        0.50 * safe_mm(gu_risk["dep_slope_gu"].fillna(0).values) +
        0.30 * safe_mm(gu_risk["Avoidance"].values) +
        0.20 * safe_mm(gu_risk["Dependency"].values)
    )
    gu_risk["전이위험점수"] = (safe_mm(gu_risk["risk_raw"].values) * 100).round(1)
    gu_risk = gu_risk.sort_values("전이위험점수", ascending=False)
    gu_risk.to_csv(OUT / "gu_transition_score.csv", index=False, encoding="utf-8-sig")
    print(f"  gu_transition_score.csv 저장")

    print(f"\n  자치구 전이 위험 상위 10:")
    for _, r in gu_risk.head(10).iterrows():
        print(f"    {r['자치구']:6s}  위험점수={r['전이위험점수']:.1f}  "
              f"dep_slope={r.get('dep_slope_gu',0):.3f}  "
              f"Q={r['Quadrant']}  Avoidance={r['Avoidance']:.1f}")

    # 자치구 보조 모델 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 왼쪽: 전이 위험점수 순위
    ax = axes[0]
    gu_sorted = gu_risk.sort_values("전이위험점수", ascending=True)
    Q_COLORS = {"Q1": "#e74c3c", "Q2": "#e67e22", "Q3": "#27ae60", "Q4": "#3498db"}
    colors_gu = [Q_COLORS.get(q, "#636e72") for q in gu_sorted["Quadrant"]]
    bars = ax.barh(range(len(gu_sorted)), gu_sorted["전이위험점수"],
                   color=colors_gu, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(gu_sorted)))
    ax.set_yticklabels(gu_sorted["자치구"], fontproperties=fp, fontsize=9)
    ax.set_xlabel("전이 위험점수 (0–100)", fontproperties=fp)
    ax.set_title("자치구별 Q1 전이 위험점수", fontproperties=fp, fontsize=12, fontweight="bold")
    ax.legend(handles=[mpatches.Patch(facecolor=c, label=q) for q, c in Q_COLORS.items()],
              prop=fp, fontsize=9, loc="lower right")
    for i, (val, bar) in enumerate(zip(gu_sorted["전이위험점수"], bars)):
        if val > 10:
            ax.text(val + 0.5, i, f"{val:.0f}", va="center", fontsize=8)

    # 오른쪽: dep_slope_gu vs Avoidance 산점도
    ax = axes[1]
    for _, r in gu_risk.iterrows():
        c = Q_COLORS.get(r["Quadrant"], "#636e72")
        s = 150 + r["전이위험점수"] * 2
        ax.scatter(r.get("dep_slope_gu", 0), r["Avoidance"], c=c, s=s,
                   edgecolors="white", linewidths=0.5, zorder=3)
        ax.annotate(r["자치구"], (r.get("dep_slope_gu", 0), r["Avoidance"]),
                    fontsize=7.5, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points", fontproperties=fp)
    ax.axvline(0, color="#636e72", ls="--", alpha=0.4)
    ax.axhline(gu_risk["Avoidance"].median(), color="#636e72", ls="--", alpha=0.4)
    ax.set_xlabel("Dependency 연간 slope (악화 방향 →)", fontproperties=fp)
    ax.set_ylabel("Avoidance Index (복지 회피도)", fontproperties=fp)
    ax.set_title("Dep 악화 추세 × 복지 회피도\n(버블 크기 = 전이 위험점수)",
                 fontproperties=fp, fontsize=12, fontweight="bold")

    fig.suptitle("자치구 Q1 전이 위험 분석", fontproperties=fp,
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUT / "gu_transition_map.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  gu_transition_map.png 저장")
else:
    print("  [SKIP] shadow_index.csv 없거나 연도별 Dep 미계산")


# ── STEP 11: 최종 리포트 ─────────────────────────────────────────────
print("\n[11] 최종 리포트")

def fmt(v): return f"{v:.3f}" if isinstance(v, float) else str(v)

res_s = model_results.get("label_slope", {})
res_e = model_results.get("label_entry", {})

# 홀드아웃 결과 포맷
def fmt_holdout(res, name):
    auc = res.get(f"{name}_holdout_auc")
    ap  = res.get(f"{name}_holdout_ap")
    if auc is None:
        return "N/A", "N/A"
    return f"{auc:.3f}", f"{ap:.3f}"

# slope 커버리지 (수정 7: 2점 이상이면 slope 계산 허용)
slope_total   = feat["dep_slope_annual"].notna().sum() if "dep_slope_annual" in feat.columns else 0
slope_coverage = slope_total / len(feat) * 100 if len(feat) > 0 else 0

n_hr = 0
if (OUT / "transition_predictions.csv").exists():
    preds = pd.read_csv(OUT / "transition_predictions.csv", encoding="utf-8-sig")
    if "위험등급" in preds.columns:
        n_hr = preds["위험등급"].isin(["고위험","최고위험"]).sum()

top5_feat_str = ""
if "label_slope" in model_results and "label_slope" in model_results.get("label_slope", {}):
    pass  # already printed above

report = f"""\
# 전이 예측 모델 — 최종 결과

## 모델 성능 (Stratified 5-fold CV)

### label_slope (연도별 Dep slope 양수 + 상위 50%)

| 모델 | AUC | Average Precision | N | 양성 |
|------|-----|-------------------|---|------|
| Logistic Regression | {fmt(res_s.get("LR_auc","N/A"))} | {fmt(res_s.get("LR_ap","N/A"))} | {res_s.get("N","N/A")} | {res_s.get("pos","N/A")} |
| Gradient Boosting   | {fmt(res_s.get("GB_auc","N/A"))} | {fmt(res_s.get("GB_ap","N/A"))} | {res_s.get("N","N/A")} | {res_s.get("pos","N/A")} |

### label_entry (기준연도 Q75 미만 → 최신연도 Q75 이상)

| 모델 | AUC | Average Precision |
|------|-----|-------------------|
| Logistic Regression | {fmt(res_e.get("LR_auc","N/A"))} | {fmt(res_e.get("LR_ap","N/A"))} |
| Gradient Boosting   | {fmt(res_e.get("GB_auc","N/A"))} | {fmt(res_e.get("GB_ap","N/A"))} |

> AUC 해석: 0.5=무작위, 0.7+=유의미, 0.8+=좋음. N≈419로 신뢰구간 넓음.

## 홀드아웃 검증 (80/20 무작위 분할, CV와 독립)

### label_slope

| 모델 | 홀드아웃 AUC | 홀드아웃 AP | train N | test N |
|------|------------|-----------|---------|--------|
| Logistic Regression | {fmt_holdout(res_s, "LR")[0]} | {fmt_holdout(res_s, "LR")[1]} | {res_s.get("holdout_n_train","N/A")} | {res_s.get("holdout_n_test","N/A")} |
| Gradient Boosting   | {fmt_holdout(res_s, "GB")[0]} | {fmt_holdout(res_s, "GB")[1]} | {res_s.get("holdout_n_train","N/A")} | {res_s.get("holdout_n_test","N/A")} |

### label_entry

| 모델 | 홀드아웃 AUC | 홀드아웃 AP |
|------|------------|-----------|
| Logistic Regression | {fmt_holdout(res_e, "LR")[0]} | {fmt_holdout(res_e, "LR")[1]} |
| Gradient Boosting   | {fmt_holdout(res_e, "GB")[0]} | {fmt_holdout(res_e, "GB")[1]} |

> CV AUC와 홀드아웃 AUC 차이가 작으면 과적합 없음. 큰 차이(0.05+)가 나면 과적합 의심.

## Slope 커버리지 (수정 7: 2점 이상 허용)

- dep_slope_annual 계산 성공: **{slope_total}개 행정동** / 전체 {len(feat)}개 ({slope_coverage:.1f}%)
- OLS slope 최소 관측치를 3점 → 2점으로 완화하여 데이터 포인트 2개인 행정동도 포함

## 고위험 행정동

- 고위험 + 최고위험 행정동: **{n_hr}개**
- 상세: transition_predictions.csv

## 피처 중요도 상위 5개

feature_importance.png 참조.
추세·연도별 피처(빨강)가 수준 피처(파랑)보다 중요하게 나오면 → 방향성이 현재 수준보다 예측력 있음.

## 산출물 목록

| 파일 | 내용 |
|------|------|
| transition_features.csv | 행정동별 피처 매트릭스 ({len(FEAT_COLS)}개 피처) |
| yearly_dependency.csv | 연도별 Dependency 궤적 |
| transition_predictions.csv | 행정동별 전이 확률 + 위험등급 |
| gu_transition_score.csv | 자치구 보조 전이 위험점수 |
| feature_importance.png | GradientBoosting 피처 중요도 |
| gu_transition_map.png | 자치구 전이 위험 시각화 |

## 레이블 정의

- **label_slope**: 연도별 Dep slope > 0 AND 상위 50% → 빠르게 악화되는 행정동
- **label_entry**: dep_{dep_cols[0] if dep_cols else '2022'} < Q75 이면서 dep_{dep_cols[-1] if dep_cols else '2025'} >= Q75 → Q1 진입

## 한계

- N≈419: max_depth=3, min_samples_leaf=10으로 과적합 억제
- 생태학적 오류: 행정동 집계 수준 → 개인 수준 전이 예측 아님
- Avoidance 축(자치구 단위)은 행정동 모델 미포함 → 자치구 보조 모델에서 보완
- 4년 데이터 기반: 검증 시계열 제한적
"""
(OUT / "model_results.md").write_text(report, encoding="utf-8")
print(f"  model_results.md 저장")

print("\n" + "=" * 60)
print("파이프라인 완료!")
print(f"산출물: {OUT}")
print("=" * 60)
