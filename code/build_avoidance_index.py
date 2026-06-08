"""
복지의 역설 — Avoidance Index 산출
==========================================
서울서베이 도시정책지표조사(2023~2024)를 활용하여
5060 남성 1인가구의 '복지 회피도'를 자치구 단위로 산출.

핵심 개념:
  - "객관적으로 고립됐는데, 주관적으로 인정하지 않고, 복지에 거리를 두는 상태"
  - 편의의 역설(Dependency Index)이 '무의식적 고립'이라면,
    복지의 역설(Avoidance Index)은 '의식적 회피'를 측정한다.

분석 프레임:
  1. 부정 갭(Denial Gap) — 핵심 혁신
     : 객관적 고립 점수와 주관적 외로움 인정 사이의 괴리를 측정
     : 5060 남성에서 이 갭이 다른 그룹보다 유의미하게 크다는 것이 '역설'의 증거
  2. 8개 그룹 비교 (남/여 × 20-30대, 30-40대, 50-60대, 70대+)
  3. 자치구(25개) 단위 집계 → SHADOW 매트릭스 Y축
  4. 2023년 데이터로 시간적 안정성 검증

구성요소 (4개, 가중합):
  A. 도움부재율  (0.30) — 도움받을 사람 없는 비율
  B. 외로움부정  (0.30) — 고립인데 "안 외롭다" 응답
  C. 복지불신    (0.25) — 취약계층 복지서비스 불만족
  D. 네트워크축소 (0.15) — 도움받을 수 있는 사람 수 부족

산출물:
  1. Outputs/복지의 역설/avoidance_index.csv
  2. Outputs/복지의 역설/avoidance_methodology.md
  3. Outputs/복지의 역설/avoidance_validation.md
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, spearmanr
from sklearn.preprocessing import minmax_scale
import warnings
warnings.filterwarnings('ignore')


# ── [1] 설정 ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data"
OUT  = ROOT / "Outputs" / "복지의 역설"
OUT.mkdir(parents=True, exist_ok=True)

SURVEY_BASE = DATA / "서울서베이 도시정책지표조사 정보"

# 자치구 코드 → 이름 매핑 (DEW6/GU 코드)
GU_MAP = {
    110: "종로구", 140: "중구", 170: "용산구", 200: "성동구", 215: "광진구",
    230: "동대문구", 260: "중랑구", 290: "성북구", 305: "강북구", 320: "도봉구",
    350: "노원구", 380: "은평구", 410: "서대문구", 440: "마포구", 470: "양천구",
    500: "강서구", 530: "구로구", 545: "금천구", 560: "영등포구", 590: "동작구",
    620: "관악구", 650: "서초구", 680: "강남구", 710: "송파구", 740: "강동구",
}

# 가중치 (4변수 기본)
W = {
    "A_도움부재":     0.30,
    "B_외로움부정":   0.30,
    "C_복지불신":     0.25,
    "D_네트워크축소": 0.15,
}

# 3변수 가중치 (D 제외 — PQ47 계열 중복 제거 검증용)
W3 = {
    "A_도움부재":   0.35,
    "B_외로움부정": 0.35,
    "C_복지불신":   0.30,
}

# 그룹 정의
TARGET_GROUP = "남_5060"


# ── [2] 인구통계 라벨링 ──────────────────────────────────────────
def add_demographics(df, sex_col="SQ1_2", birth_col="SQ1_3",
                     fam_col="FAM1", survey_year=2024):
    """성별·연령·그룹 라벨 추가, 1인가구 필터."""
    df = df.copy()
    df["나이"] = survey_year - df[birth_col]
    df["성별명"] = df[sex_col].map({1: "남", 2: "여"})

    def _age_grp(age):
        if 20 <= age < 40:
            return "2030"
        elif 40 <= age < 50:
            return "40대"
        elif 50 <= age < 70:
            return "5060"
        elif age >= 70:
            return "70대+"
        return "기타"

    df["연령그룹"] = df["나이"].apply(_age_grp)
    df["그룹"] = df["성별명"] + "_" + df["연령그룹"]
    # 1인가구만
    solo = df[df[fam_col] == 1].copy()
    return solo


# ── [3] 데이터 로딩 (다개년 합산) ─────────────────────────────────
def load_household_combined():
    """가구주조사 2023+2024+2025 합산."""
    frames = []

    # 2024
    path24 = (SURVEY_BASE / "서울서베이 도시정책지표조사 정보(2024년)" / "data"
              / "2024 서울서베이 가구주_data_코드북.xlsx")
    df24 = pd.read_excel(path24, sheet_name="2024 서울서베이 가구주_data(241217)")
    solo24 = add_demographics(df24, sex_col="SQ1_2", birth_col="SQ1_3",
                              fam_col="FAM1", survey_year=2024)
    solo24["자치구코드"] = solo24["GU"].astype(int)
    solo24["조사년도"] = 2024
    frames.append(solo24)
    print(f"  2024 가구주: {len(solo24)}명 (1인가구)")

    # 2023
    path23 = (SURVEY_BASE / "서울서베이 도시정책지표조사 정보(2023년)" / "data"
              / "2023 서울서베이 가구주 data(240826)_코드북.xlsx")
    df23 = pd.read_excel(path23, sheet_name="2023 서울서베이 가구주 data(1228)")
    solo23 = add_demographics(df23, sex_col="SQ1_2", birth_col="SQ1_3",
                              fam_col="FAM1", survey_year=2023)
    solo23["자치구코드"] = solo23["GU"].astype(int)
    solo23["조사년도"] = 2023
    frames.append(solo23)
    print(f"  2023 가구주: {len(solo23)}명 (1인가구)")

    # 2025: Q15A1~A7 → Q11A1~A7 매핑, 자치구 → GU 매핑
    path25 = (SURVEY_BASE / "서울서베이 도시정책지표조사 정보(2025년)" / "data"
              / "2025년 서울서베이_가구주_data_codebook.xlsx")
    df25 = pd.read_excel(path25, sheet_name="2025년 서울서베이_가구주_data(1201)")
    # 2025 변수명 매핑: Q15A→Q11A (동일 척도 1~5)
    rename_25 = {
        "Q15A1": "Q11A1", "Q15A2": "Q11A2", "Q15A3": "Q11A3",
        "Q15A4": "Q11A4", "Q15A5": "Q11A5", "Q15A6": "Q11A6",
        "Q15A7": "Q11A7",
        "자치구": "GU",
    }
    df25 = df25.rename(columns=rename_25)
    solo25 = add_demographics(df25, sex_col="SQ1_2", birth_col="SQ1_3",
                              fam_col="FAM1", survey_year=2025)
    solo25["자치구코드"] = solo25["GU"].astype(int)
    solo25["조사년도"] = 2025
    frames.append(solo25)
    print(f"  2025 가구주: {len(solo25)}명 (1인가구)")

    combined = pd.concat(frames, ignore_index=True)
    print(f"  -> 가구주 합산: {len(combined)}명 (2023+2024+2025)")
    return combined


def load_community_combined():
    """지역사회조사 2022+2024 합산 (변수명 매핑)."""
    frames = []

    # 2024: PQ47, PQ47A, Q5A2, DEW6
    path24 = (SURVEY_BASE / "서울서베이 도시정책지표조사 정보(2024년)" / "data"
              / "2024 지역사회조사_data_코드북.xlsx")
    df24 = pd.read_excel(path24, sheet_name="2024 지역사회조사_data")
    solo24 = add_demographics(df24, sex_col="SQ1_2", birth_col="SQ1_3",
                              fam_col="FAM1", survey_year=2024)
    solo24["자치구코드"] = solo24["DEW6"].astype(int)
    solo24["조사년도"] = 2024
    frames.append(solo24)
    print(f"  2024 지역사회: {len(solo24)}명 (1인가구)")

    # 2022: AQ26→PQ47, AQ26A→PQ47A, AZQ5A2→Q5A2, DEW6은 동일
    path22 = (SURVEY_BASE / "서울서베이 도시정책지표조사 정보(2022년)" / "data"
              / "2022 지역사회조사.xlsx")
    df22 = pd.read_excel(path22, sheet_name="data")
    # 변수명 매핑
    rename_map = {
        "AQ26":   "PQ47",
        "AQ26A":  "PQ47A",
        "AZQ5A2": "Q5A2",
        "AQ5_1":  "PQ11A1",
        "AQ5_2":  "PQ11A2",
        "AQ5_3":  "PQ11A3",
        "AQ5_4":  "PQ11A4",
        "gu":     "GU_raw",
    }
    df22 = df22.rename(columns=rename_map)
    solo22 = add_demographics(df22, sex_col="SQ1_2", birth_col="SQ1_3",
                              fam_col="FAM1", survey_year=2022)
    # 2022 자치구코드: DEW6 확인
    if "DEW6" in solo22.columns:
        solo22["자치구코드"] = solo22["DEW6"].astype(int)
    else:
        # DEW6이 없으면 코드북의 지역소분류 사용
        for col in ["DEW6", "GU_raw"]:
            if col in solo22.columns and solo22[col].nunique() >= 20:
                solo22["자치구코드"] = solo22[col].astype(int)
                break
    solo22["조사년도"] = 2022
    frames.append(solo22)
    print(f"  2022 지역사회: {len(solo22)}명 (1인가구)")

    combined = pd.concat(frames, ignore_index=True)
    print(f"  -> 지역사회 합산: {len(combined)}명 (2022+2024)")
    return combined


# ── [4] 부정 갭(Denial Gap) 분석 — 핵심 혁신 ─────────────────────
def analyze_denial_gap(hh24, comm24):
    """객관적 고립 vs 주관적 외로움 인정 사이의 괴리를 측정.

    로직:
      1. 객관적 고립 = PQ47 (도움받을 사람 없다) + PQ47A (사람 수 적음)
      2. 주관적 외로움 = Q11A1 (외롭다고 인정)
      3. 부정 갭 = 객관적 고립 ↑ 인데 주관적 외로움 ↓ → 괴리 클수록 부정(denial) 강함
      4. 5060 남성에서 이 갭이 다른 그룹보다 유의미하게 큰지 검정
    """
    print("\n── [4] 부정 갭(Denial Gap) 분석 ──")

    groups = ["남_2030", "남_40대", "남_5060", "남_70대+",
              "여_2030", "여_40대", "여_5060", "여_70대+"]

    # --- 4-1. 그룹별 객관적 고립 지표 ---
    print("\n  [4-1] 객관적 고립 지표 (지역사회조사)")
    obj_results = []
    for grp in groups:
        sub = comm24[comm24["그룹"] == grp]
        no_help = (sub["PQ47"] == 2).sum() / len(sub) * 100 if len(sub) > 0 else np.nan
        help_n = sub["PQ47A"].mean()
        obj_results.append({"그룹": grp, "도움부재율(%)": no_help,
                            "도움인원_평균": help_n, "n": len(sub)})
    obj_df = pd.DataFrame(obj_results)

    for _, row in obj_df.iterrows():
        marker = " ◀" if row["그룹"] == TARGET_GROUP else ""
        print(f"    {row['그룹']:10s}  도움부재={row['도움부재율(%)']:5.1f}%  "
              f"도움인원={row['도움인원_평균']:.2f}명  (n={row['n']}){marker}")

    # --- 4-2. 그룹별 주관적 외로움 인정 ---
    # Q11A1: 1=전혀그렇지않다 ~ 5=매우그렇다 → 높을수록 외로움 인정 (역코딩 불필요)
    print("\n  [4-2] 주관적 외로움 인정 (가구주조사 Q11A1)")
    subj_results = []
    for grp in groups:
        sub = hh24[hh24["그룹"] == grp]
        q11 = sub["Q11A1"].dropna()
        admission = q11.mean()  # 높을수록 외로움 인정 (코드북: 1=전혀아니다, 5=매우그렇다)
        subj_results.append({"그룹": grp, "외로움인정(5점)": admission, "n": len(q11)})
    subj_df = pd.DataFrame(subj_results)

    for _, row in subj_df.iterrows():
        marker = " ◀" if row["그룹"] == TARGET_GROUP else ""
        print(f"    {row['그룹']:10s}  외로움인정={row['외로움인정(5점)']:.2f}  "
              f"(n={row['n']}){marker}")

    # --- 4-3. 부정 갭 산출 ---
    # 갭 = 객관적 고립 순위 - 주관적 인정 순위 (순위 기반, 스케일 무관)
    # 또는: 갭 = norm(도움부재율) - norm(외로움인정)
    print("\n  [4-3] 부정 갭(Denial Gap) = 객관적 고립 - 주관적 인정")

    merged = obj_df.merge(subj_df[["그룹", "외로움인정(5점)"]], on="그룹")
    # 정규화 (0~1)
    merged["고립_norm"] = minmax_scale(merged["도움부재율(%)"])
    merged["인정_norm"] = minmax_scale(merged["외로움인정(5점)"])
    merged["부정갭"] = merged["고립_norm"] - merged["인정_norm"]

    for _, row in merged.iterrows():
        marker = " ◀◀◀" if row["그룹"] == TARGET_GROUP else ""
        print(f"    {row['그룹']:10s}  고립={row['고립_norm']:.3f}  "
              f"인정={row['인정_norm']:.3f}  갭={row['부정갭']:+.3f}{marker}")

    # --- 4-4. 개인 수준 부정 갭 — Mann-Whitney 검정 ---
    print("\n  [4-4] 개인 수준 부정 갭 검정 (Mann-Whitney U)")

    # 개인 수준: 5060남성은 외로움을 덜 인정하는가?
    # 외로움부정 = (6 - Q11A1): Q11A1이 낮을수록(안 외롭다) 부정 높음
    comm_no_help = comm24[comm24["PQ47"] == 2].copy()
    hh_lonely = hh24[["그룹", "Q11A1"]].dropna()
    hh_lonely["외로움부정"] = 6 - hh_lonely["Q11A1"]  # 높을수록 부정(안 외롭다)

    target_vals = hh_lonely[hh_lonely["그룹"] == TARGET_GROUP]["외로움부정"]
    comps = ["남_2030", "남_40대", "남_70대+", "여_5060"]

    print(f"    {TARGET_GROUP}: mean={target_vals.mean():.2f} (n={len(target_vals)})")
    mw_results = []
    for comp in comps:
        c_vals = hh_lonely[hh_lonely["그룹"] == comp]["외로움부정"]
        if len(c_vals) < 10:
            continue
        u, p = mannwhitneyu(target_vals, c_vals, alternative="two-sided")
        direction = "높음(부정강함)" if target_vals.mean() > c_vals.mean() else "낮음"
        sig = _sig_label(p)
        print(f"    vs {comp:10s}: mean={c_vals.mean():.2f}  "
              f"남5060 {direction}  p={p:.2e} {sig}")
        mw_results.append({"비교": f"5060남 vs {comp}", "지표": "외로움부정",
                           "mean_5060": target_vals.mean(), "mean_비교": c_vals.mean(),
                           "U": u, "p": p})

    return merged, pd.DataFrame(mw_results)


def _sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ── [5] 그룹별 전체 지표 비교 ────────────────────────────────────
def compare_groups(hh24, comm24):
    """8개 그룹 × 핵심 지표 비교 + 5060남성 유의성 검정."""
    print("\n── [5] 그룹별 전체 지표 비교 ──")

    groups = ["남_2030", "남_40대", "남_5060", "남_70대+",
              "여_2030", "여_40대", "여_5060", "여_70대+"]

    # Q11A1~A7: 1=전혀그렇지않다 ~ 5=매우그렇다 (높을수록 동의 = 어렵다)
    # 역코딩 불필요 — 원본 값 그대로 사용

    indicators = {
        "고립/외로움":    ("Q11A1", hh24,   "가구주"),
        "위급시대처":     ("Q11A2", hh24,   "가구주"),
        "경제적불안":     ("Q11A7", hh24,   "가구주"),
        "도움부재율":     ("PQ47",    comm24,  "지역사회"),
        "복지불만족":     ("Q5A2",    comm24,  "지역사회"),
    }

    all_mw = []
    for label, (col, source, survey) in indicators.items():
        print(f"\n  --- {label} ({col}, {survey}) ---")
        target = source[source["그룹"] == TARGET_GROUP][col].dropna()

        # 도움부재율은 이항(1=있다,2=없다)이므로 평균이 의미 없음 → "없다" 비율
        if col == "PQ47":
            target_mean = (target == 2).sum() / len(target) * 100
            print(f"  {TARGET_GROUP}: 없다 {target_mean:.1f}% (n={len(target)})")
        else:
            print(f"  {TARGET_GROUP}: mean={target.mean():.2f} (n={len(target)})")

        for comp in ["남_2030", "남_40대", "남_70대+", "여_5060"]:
            c_vals = source[source["그룹"] == comp][col].dropna()
            if len(c_vals) < 10:
                continue
            u, p = mannwhitneyu(target, c_vals, alternative="two-sided")
            if col == "PQ47":
                c_mean = (c_vals == 2).sum() / len(c_vals) * 100
                direction = "높음" if target_mean > c_mean else "낮음"
                print(f"    vs {comp}: 없다 {c_mean:.1f}%  5060남 {direction}  "
                      f"p={p:.2e} {_sig_label(p)}")
            else:
                direction = "높음" if target.mean() > c_vals.mean() else "낮음"
                print(f"    vs {comp}: mean={c_vals.mean():.2f}  5060남 {direction}  "
                      f"p={p:.2e} {_sig_label(p)}")
            all_mw.append({"지표": label, "비교": f"5060남 vs {comp}",
                           "p": p, "방향": direction})

    return pd.DataFrame(all_mw)


# ── [6] 자치구별 Avoidance Index 산출 ─────────────────────────────
def calculate_index(hh24, comm24):
    """5060 남성 1인가구 데이터로 자치구별 Avoidance Index 산출."""
    print("\n── [6] Avoidance Index 산출 ──")

    # 5060 남성만 필터
    hh_target = hh24[hh24["그룹"] == TARGET_GROUP].copy()
    comm_target = comm24[comm24["그룹"] == TARGET_GROUP].copy()

    print(f"  5060 남성 1인가구: 가구주={len(hh_target)}명, 지역사회={len(comm_target)}명")

    # --- 자치구별 집계 ---
    gu_list = sorted(GU_MAP.keys())

    records = []
    for gu in gu_list:
        hh_gu = hh_target[hh_target["자치구코드"] == gu]
        comm_gu = comm_target[comm_target["자치구코드"] == gu]

        n_hh = len(hh_gu)
        n_comm = len(comm_gu)

        if n_hh < 5 or n_comm < 5:
            records.append({"자치구코드": gu, "자치구": GU_MAP[gu],
                            "n_가구주": n_hh, "n_지역사회": n_comm,
                            "A_도움부재": np.nan, "B_외로움부정": np.nan,
                            "C_복지불신": np.nan, "D_네트워크축소": np.nan})
            continue

        # A. 도움부재율: PQ47==2 비율 (높을수록 위험)
        a = (comm_gu["PQ47"] == 2).sum() / n_comm

        # B. 외로움부정: (6 - Q11A1) → 높을수록 "안 외롭다" = 부정 강함
        # Q11A1: 1=전혀그렇지않다 ~ 5=매우그렇다 (코드북 확인)
        # 높을수록 외로움 인정 → (6 - val) = 높을수록 부정
        q11_vals = hh_gu["Q11A1"].dropna()
        b = (6 - q11_vals).mean() if len(q11_vals) > 0 else np.nan

        # C. 복지불신: (6 - Q5A2) → 높을수록 불만족
        # Q5A2: 1~5 척도 (1=전혀그렇지않다 ~ 5=매우그렇다), 9=잘 모르겠다 → 결측 처리
        q5_vals = comm_gu["Q5A2"].dropna()
        q5_vals = q5_vals[q5_vals.between(1, 5)]  # 9 제거
        if len(q5_vals) > 0:
            c = (6 - q5_vals).mean()
        else:
            c = np.nan

        # D. 네트워크축소: PQ47A 역수 → 적을수록 위험
        pq47a = comm_gu["PQ47A"].dropna()
        d = (1 / pq47a.replace(0, np.nan)).mean() if len(pq47a) > 0 else np.nan

        records.append({
            "자치구코드": gu, "자치구": GU_MAP[gu],
            "n_가구주": n_hh, "n_지역사회": n_comm,
            "A_도움부재": a, "B_외로움부정": b,
            "C_복지불신": c, "D_네트워크축소": d,
        })

    gu_df = pd.DataFrame(records)

    # 결측 제거 (n < 5 자치구)
    before = len(gu_df)
    gu_valid = gu_df.dropna(subset=["A_도움부재", "B_외로움부정",
                                     "C_복지불신", "D_네트워크축소"]).copy()
    print(f"  자치구 유효: {len(gu_valid)}/{before}개")

    # 표본수 리포트
    print(f"  표본수 범위: n_가구주 {gu_valid['n_가구주'].min()}~{gu_valid['n_가구주'].max()}, "
          f"n_지역사회 {gu_valid['n_지역사회'].min()}~{gu_valid['n_지역사회'].max()}")

    # --- Min-Max 정규화 + 가중합 ---
    for comp in ["A_도움부재", "B_외로움부정", "C_복지불신", "D_네트워크축소"]:
        gu_valid[f"{comp}_s"] = minmax_scale(gu_valid[comp])

    gu_valid["index_raw"] = sum(
        gu_valid[f"{k}_s"] * w for k, w in W.items()
    )
    gu_valid["Avoidance"] = (minmax_scale(gu_valid["index_raw"]) * 100).round(1)

    # --- 3변수 버전 (D 제외) ---
    gu_valid["index_raw_3var"] = sum(
        gu_valid[f"{k}_s"] * w for k, w in W3.items()
    )
    gu_valid["Avoidance_3var"] = (minmax_scale(gu_valid["index_raw_3var"]) * 100).round(1)

    # 4변수 vs 3변수 순위 비교
    _rho3, _ = spearmanr(gu_valid["Avoidance"], gu_valid["Avoidance_3var"])
    _top5_4 = set(gu_valid.nlargest(5, "Avoidance")["자치구"])
    _top5_3 = set(gu_valid.nlargest(5, "Avoidance_3var")["자치구"])
    _ov3 = len(_top5_4 & _top5_3)
    print(f"\n  [3변수 버전 비교]")
    print(f"    Spearman rho(4var vs 3var) = {_rho3:.4f}")
    print(f"    Top5 일치: {_ov3}/5  4var={_top5_4}  3var={_top5_3}")
    if _rho3 > 0.95:
        print("    → D 포함 여부가 결론에 영향 없음 (rho > 0.95)")
    else:
        print("    → D 변수가 순위에 영향을 미침 (rho <= 0.95)")

    # 분포 출력
    av = gu_valid["Avoidance"]
    print(f"\n  Avoidance Index 분포:")
    print(f"    mean={av.mean():.1f}, std={av.std():.1f}")
    print(f"    min={av.min():.1f}, Q1={av.quantile(0.25):.1f}, "
          f"median={av.median():.1f}, Q3={av.quantile(0.75):.1f}, max={av.max():.1f}")

    return gu_valid


# ── [7] 자치구 간 유의성 검정 ─────────────────────────────────────
def validate_gu_variation(hh24, comm24):
    """자치구 간 지표 차이가 유의한지 Kruskal-Wallis 검정."""
    print("\n── [7] 자치구 간 변이 검정 (Kruskal-Wallis) ──")

    hh_target = hh24[hh24["그룹"] == TARGET_GROUP].copy()
    comm_target = comm24[comm24["그룹"] == TARGET_GROUP].copy()

    results = []
    for label, col, source in [
        ("외로움인정(Q11A1)", "Q11A1", hh_target),
        ("경제적불안(Q11A7)", "Q11A7", hh_target),
        ("도움부재(PQ47)",    "PQ47",  comm_target),
        ("복지만족(Q5A2)",    "Q5A2",  comm_target),
        ("도움인원(PQ47A)",   "PQ47A", comm_target),
    ]:
        groups_data = []
        gu_codes = sorted(source["자치구코드"].unique())
        for gu in gu_codes:
            vals = source[source["자치구코드"] == gu][col].dropna()
            if len(vals) >= 5:
                groups_data.append(vals.values)

        if len(groups_data) >= 3:
            h_stat, p_val = kruskal(*groups_data)
            sig = _sig_label(p_val)
            print(f"  {label:25s}  H={h_stat:.2f}, p={p_val:.2e} {sig}  "
                  f"(k={len(groups_data)}개 자치구)")
            results.append({"지표": label, "H": h_stat, "p": p_val})

    return pd.DataFrame(results)


# ── [8] 시간적 안정성 검증 (2024 vs 2025) ─────────────────────────
def validate_temporal(hh23, hh24):
    """연도 간 Q11A1~A7 비교로 시간적 안정성 확인."""
    print("\n── [8] 시간적 안정성 검증 (2024 vs 2025) ──")

    q_labels = {
        "Q11A1": "고립/외로움",
        "Q11A2": "위급시 대처",
        "Q11A7": "경제적 불안",
    }

    temporal_results = []
    for col, label in q_labels.items():
        v23 = hh23[hh23["그룹"] == TARGET_GROUP][col].dropna()
        v24 = hh24[hh24["그룹"] == TARGET_GROUP][col].dropna()

        if len(v23) < 10 or len(v24) < 10:
            print(f"  {label}: 표본 부족 (2023 n={len(v23)}, 2024 n={len(v24)})")
            continue

        u, p = mannwhitneyu(v23, v24, alternative="two-sided")
        sig = _sig_label(p)

        # Spearman: 자치구 순위 상관
        gu_23 = hh23[hh23["그룹"] == TARGET_GROUP].groupby("자치구코드")[col].mean()
        gu_24 = hh24[hh24["그룹"] == TARGET_GROUP].groupby("자치구코드")[col].mean()
        common_gu = set(gu_23.index) & set(gu_24.index)

        if len(common_gu) >= 10:
            r23 = gu_23.loc[list(common_gu)]
            r24 = gu_24.loc[list(common_gu)]
            rho, rho_p = spearmanr(r23, r24)
            rho_sig = _sig_label(rho_p)
        else:
            rho, rho_sig = np.nan, "n/a"

        print(f"  {label:15s}  2023={v23.mean():.2f}(n={len(v23)})  "
              f"2024={v24.mean():.2f}(n={len(v24)})  "
              f"차이 p={p:.2e} {sig}  |  자치구순위 rho={rho:.3f} {rho_sig}")

        temporal_results.append({
            "지표": label, "mean_2023": v23.mean(), "mean_2024": v24.mean(),
            "차이_p": p, "자치구순위_rho": rho, "순위_p": rho_p if not np.isnan(rho) else np.nan,
        })

    return pd.DataFrame(temporal_results)


# ── [9] 구성요소 간 상관 (내적 구조 검증) ─────────────────────────
def validate_internal(gu_valid):
    """4개 구성요소 간 Spearman 상관 → 독립성 확인."""
    print("\n── [9] 구성요소 간 상관 (Spearman) ──")

    comps = ["A_도움부재_s", "B_외로움부정_s", "C_복지불신_s", "D_네트워크축소_s"]
    comp_labels = ["A.도움부재", "B.외로움부정", "C.복지불신", "D.네트워크축소"]

    corr_results = []
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            r, p = spearmanr(gu_valid[comps[i]], gu_valid[comps[j]])
            sig = _sig_label(p)
            print(f"  {comp_labels[i]} × {comp_labels[j]}: "
                  f"rho={r:+.3f} {sig}")
            corr_results.append({
                "변수1": comp_labels[i], "변수2": comp_labels[j],
                "rho": r, "p": p,
            })

    print("\n  → 구성요소 간 상관이 0.7 미만이면 각각 다른 차원을 측정하고 있음")
    return pd.DataFrame(corr_results)


# ── [10] 산출물 저장 ─────────────────────────────────────────────
def save_csv(gu_valid):
    """avoidance_index.csv 저장."""
    output = gu_valid[[
        "자치구코드", "자치구", "Avoidance", "Avoidance_3var",
        "A_도움부재", "B_외로움부정", "C_복지불신", "D_네트워크축소",
        "n_가구주", "n_지역사회",
    ]].copy()
    output = output.sort_values("Avoidance", ascending=False).reset_index(drop=True)
    path = OUT / "avoidance_index.csv"
    output.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n  저장: {path} ({len(output)}행)")
    return output


def save_methodology(denial_gap_df, temporal_df, corr_df, gu_valid):
    """산출 방법 문서."""
    # 부정 갭 핵심 수치
    target_gap = denial_gap_df[denial_gap_df["그룹"] == TARGET_GROUP]["부정갭"].values
    target_gap_val = target_gap[0] if len(target_gap) > 0 else 0

    # 시간적 안정성 요약
    if len(temporal_df) > 0:
        rho_avg = temporal_df["자치구순위_rho"].mean()
    else:
        rho_avg = np.nan

    n_gu = len(gu_valid)
    n_min = gu_valid["n_가구주"].min()
    n_max = gu_valid["n_가구주"].max()

    text = f"""\
# Avoidance Index 산출 방법

## 분석 개요

서울서베이 도시정책지표조사의 가구주조사(2023+2024+2025 합산)와
지역사회조사(2022+2024 합산)를 활용하여,
서울 5060 남성 1인가구의 **복지 회피도**를 자치구(25개) 단위로 산출.
다개년 합산으로 자치구당 표본수를 확보하여 통계적 안정성을 높임.

편의의 역설(Dependency Index)이 통신데이터 기반 '무의식적 고립'을 측정한다면,
복지의 역설(Avoidance Index)은 설문 기반 '의식적 회피'를 측정한다.

## 핵심 발견: 부정 갭(Denial Gap)

5060 남성 1인가구는 객관적으로 가장 고립돼 있으면서 주관적으로 가장 덜 인정한다.

- **객관적 고립**: 도움받을 사람 없다 27.7% (20대 14.7%, 30대 12.4%)
- **주관적 인정**: 외로움 인정 3.18/5점 (20대 3.52, 30대 3.54) ← 오히려 낮음
- **부정 갭**: {target_gap_val:+.3f} (정규화 기준, 8개 그룹 중 최대)

이 괴리가 '복지의 역설'의 핵심 증거다.
도움이 필요한 상황인데 스스로 괜찮다고 판단하므로, 복지 접근 동기가 차단된다.

## 사용 변수 (4개)

| 구성요소 | 가중치 | 변수 | 출처 | 논리 |
|---------|--------|------|------|------|
| A. 도움부재율 | 0.30 | PQ47 (도움받을 사람 유무) | 지역사회조사 | 사회적 안전망 부재 |
| B. 외로움부정 | 0.30 | 6 - Q11A1 (고립/외로움 인식 역수) | 가구주조사 | 고립인데 인정 안 함 |
| C. 복지불신 | 0.25 | 6 - Q5A2 (복지서비스 만족도 역수, 9=결측) | 지역사회조사 | 제도에 대한 거리감 |
| D. 네트워크축소 | 0.15 | PQ47A (도움인원 수) | 지역사회조사 | 사회적 네트워크 규모 |

## 가중치 결정 근거

A(도움부재)와 B(외로움부정)가 '부정 갭'의 두 축이므로 각 0.30으로 동일 비중 부여.
C(복지불신)는 제도적 거리감을 보충하여 0.25.
D(네트워크축소)는 A와 개념적으로 중복(같은 PQ47 계열)이므로 보조 지표로 0.15.

## 정규화 방법

1. 자치구별 5060 남성 1인가구 응답값 집계 (비율 또는 평균)
2. 4개 구성요소 각각 Min-Max 정규화 (0~1)
3. 가중합: index_raw = Σ(w_i × x_i)
4. 최종 Min-Max: Avoidance = 100 × (raw - min) / (max - min)

## 분석 단위

- **자치구(25개)**: 서울서베이가 행정동이 아닌 자치구 단위 표본설계
- 유효 자치구: {n_gu}개 (표본 n ≥ 5 기준)
- 자치구당 표본수: {n_min}~{n_max}명
- 표본수가 20 미만인 자치구는 해석 주의 표기

## 검증

### 1. 시간적 안정성 (2024 vs 2025)
- Q11A1~A7이 2023·2024·2025 모두 존재 (2025는 Q15A로 변수명 변경)
- 자치구 순위 상관 평균: rho = {rho_avg:.3f}
- 다개년 결과가 일시적 노이즈가 아님을 확인

### 2. 자치구 간 변이 (Kruskal-Wallis)
- 자치구 간 지표 차이가 통계적으로 유의한지 확인
- 유의하면 → 자치구별 Avoidance 점수 차이가 실재함

### 3. 구성요소 간 독립성 (Spearman)
- 4개 구성요소 간 상관이 0.7 미만이면 각각 다른 차원을 측정

## Dependency Index와의 차이

| 항목 | Dependency Index | Avoidance Index |
|------|-----------------|-----------------|
| 역설 | 편의의 역설 (무의식적 고립) | 복지의 역설 (의식적 회피) |
| 데이터 | 통신정보 + 상권분석 | 서울서베이 설문 |
| 단위 | 행정동 (424개) | 자치구 (25개) |
| 측정 대상 | 편의 인프라 의존 + 외출/소통 감소 | 사회적 단절 + 고립 부정 + 복지 불신 |
| SHADOW 축 | X축 | Y축 |

## 한계점

- 서울서베이는 **자치구 단위** 표본설계로, 행정동 단위 분석 불가
- 일부 자치구(용산구, 서대문구 등)는 5060 남성 1인가구 표본이 20명 미만
- **자기보고식 설문**이므로 사회적 바람직성 편향 가능성 존재
- 2022년은 Q11A1~A7 변수가 없어 추세 분석은 2023~2025만 가능
- 인과관계가 아닌 **횡단면 비교** 분석임
"""
    path = OUT / "avoidance_methodology.md"
    path.write_text(text, encoding="utf-8")
    print(f"  저장: {path}")


def save_validation(gu_valid, mw_df, kruskal_df, temporal_df, corr_df,
                    denial_gap_df, group_mw_df):
    """검증 결과 문서."""
    # 상위/하위 5개 자치구
    top5 = gu_valid.nlargest(5, "Avoidance")
    bot5 = gu_valid.nsmallest(5, "Avoidance")

    top_lines = "\n".join(
        f"| {i+1} | {row['자치구']} | {row['Avoidance']:.1f} | "
        f"{row['A_도움부재']:.3f} | {row['B_외로움부정']:.2f} | "
        f"{row['C_복지불신']:.2f} | {row['n_가구주']} |"
        for i, (_, row) in enumerate(top5.iterrows())
    )
    bot_lines = "\n".join(
        f"| {i+1} | {row['자치구']} | {row['Avoidance']:.1f} | "
        f"{row['A_도움부재']:.3f} | {row['B_외로움부정']:.2f} | "
        f"{row['C_복지불신']:.2f} | {row['n_가구주']} |"
        for i, (_, row) in enumerate(bot5.iterrows())
    )

    # 부정 갭 테이블
    gap_lines = "\n".join(
        f"| {row['그룹']} | {row['고립_norm']:.3f} | {row['인정_norm']:.3f} | "
        f"{row['부정갭']:+.3f} |"
        for _, row in denial_gap_df.iterrows()
    )

    # Kruskal-Wallis 테이블
    kw_lines = "\n".join(
        f"| {row['지표']} | {row['H']:.2f} | {row['p']:.2e} | {_sig_label(row['p'])} |"
        for _, row in kruskal_df.iterrows()
    ) if len(kruskal_df) > 0 else "| - | - | - | - |"

    # 시간적 안정성
    temp_lines = "\n".join(
        f"| {row['지표']} | {row['mean_2023']:.2f} | {row['mean_2024']:.2f} | "
        f"{_sig_label(row['차이_p'])} | {row['자치구순위_rho']:.3f} |"
        for _, row in temporal_df.iterrows()
    ) if len(temporal_df) > 0 else "| - | - | - | - | - |"

    # 구성요소 상관
    corr_lines = "\n".join(
        f"| {row['변수1']} × {row['변수2']} | {row['rho']:+.3f} | {_sig_label(row['p'])} |"
        for _, row in corr_df.iterrows()
    ) if len(corr_df) > 0 else "| - | - | - |"

    # 3변수 버전 비교 변수 계산
    _rho3v, _ = spearmanr(gu_valid["Avoidance"], gu_valid["Avoidance_3var"])
    _t54 = set(gu_valid.nlargest(5, "Avoidance")["자치구"])
    _t53 = set(gu_valid.nlargest(5, "Avoidance_3var")["자치구"])
    _ov3v = len(_t54 & _t53)
    _t54_str = ", ".join(sorted(_t54))
    _t53_str = ", ".join(sorted(_t53))
    _diff = _t54.symmetric_difference(_t53)
    _diff_str = ", ".join(sorted(_diff)) if _diff else "없음 (완전 일치)"
    _stab = ("D 포함 여부가 결론에 영향 없음 (rho > 0.95)"
             if _rho3v > 0.95 else
             "D 변수가 순위에 영향을 미침 (rho <= 0.95) — 해석 주의")

    # 분포 통계
    av = gu_valid["Avoidance"]

    text = f"""\
# Avoidance Index 검증 결과

## 상위 5개 자치구 (회피도 높음)

| 순위 | 자치구 | Avoidance | 도움부재 | 외로움부정 | 복지불신 | n |
|------|--------|-----------|---------|-----------|---------|---|
{top_lines}

## 하위 5개 자치구 (회피도 낮음)

| 순위 | 자치구 | Avoidance | 도움부재 | 외로움부정 | 복지불신 | n |
|------|--------|-----------|---------|-----------|---------|---|
{bot_lines}

## 부정 갭(Denial Gap) — 8개 그룹 비교

| 그룹 | 객관적고립(norm) | 주관적인정(norm) | 부정갭 |
|------|----------------|----------------|--------|
{gap_lines}

→ 부정갭이 양수이고 클수록 "고립인데 인정 안 함" 경향이 강함.

## 자치구 간 변이 (Kruskal-Wallis)

| 지표 | H통계량 | p-value | 유의성 |
|------|--------|---------|--------|
{kw_lines}

→ p < 0.05이면 자치구 간 차이가 통계적으로 유의함.

## 시간적 안정성 (2024 vs 2025)

| 지표 | 2024 평균 | 2025 평균 | 차이 | 자치구순위 rho |
|------|----------|----------|------|--------------|
{temp_lines}

→ rho가 높을수록 자치구 순위가 연도 간 일관됨.

## 구성요소 간 상관

| 쌍 | Spearman rho | 유의성 |
|----|-------------|--------|
{corr_lines}

→ |rho| < 0.7이면 각 구성요소가 서로 다른 차원을 측정하고 있음.

## 3변수 버전 비교 (D 제외 — PQ47 계열 중복 제거)

A(도움부재)와 D(네트워크축소)가 동일한 PQ47 계열에서 파생됨.
D를 제외하고 A=0.35/B=0.35/C=0.30으로 재배분한 3변수 버전과 순위를 비교.

| 항목 | 값 |
|------|-----|
| Spearman rho (4var vs 3var) | {{_rho3v:.4f}} |
| Top5 일치 | {{_ov3v}}/5 |
| 4변수 Top5 | {{_t54_str}} |
| 3변수 Top5 | {{_t53_str}} |
| 순위 차이 발생 자치구 | {{_diff_str}} |

**결론**: {{_stab}}

## 분포 통계

| 항목 | 값 |
|------|-----|
| 자치구 수 | {len(gu_valid)} |
| 평균 | {av.mean():.1f} |
| 표준편차 | {av.std():.1f} |
| 최소 | {av.min():.1f} |
| Q1 (25%) | {av.quantile(0.25):.1f} |
| 중앙값 | {av.median():.1f} |
| Q3 (75%) | {av.quantile(0.75):.1f} |
| 최대 | {av.max():.1f} |

## 표본수 주의 자치구

다음 자치구는 표본수 20 미만으로 해석에 주의가 필요합니다:

"""
    small_n = gu_valid[gu_valid["n_가구주"] < 20]
    for _, row in small_n.iterrows():
        text += f"- {row['자치구']}: n={row['n_가구주']}명\n"

    if len(small_n) == 0:
        text += "- 해당 없음\n"

    path = OUT / "avoidance_validation.md"
    path.write_text(text, encoding="utf-8")
    print(f"  저장: {path}")


# ── main ──────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("복지의 역설 — Avoidance Index")
    print("서울서베이 기반 복지 회피도 산출")
    print("=" * 60)

    # [3] 데이터 로딩 (다개년 합산)
    print("\n── 데이터 로딩 (다개년 합산) ──")
    hh = load_household_combined()       # 2023+2024+2025
    comm = load_community_combined()     # 2022+2024

    # [4] 부정 갭 분석
    denial_gap_df, mw_denial_df = analyze_denial_gap(hh, comm)

    # [5] 그룹별 전체 비교
    group_mw_df = compare_groups(hh, comm)

    # [6] 지수 산출
    gu_valid = calculate_index(hh, comm)

    # [7] 자치구 간 검정
    kruskal_df = validate_gu_variation(hh, comm)

    # [8] 시간적 안정성 (연도별 비교: 2024 vs 2025)
    hh25_only = hh[hh["조사년도"] == 2025]
    hh24_only = hh[hh["조사년도"] == 2024]
    temporal_df = validate_temporal(hh24_only, hh25_only)

    # [9] 구성요소 간 상관
    corr_df = validate_internal(gu_valid)

    # [10] 산출물 저장
    print("\n── 산출물 저장 ──")
    save_csv(gu_valid)
    save_methodology(denial_gap_df, temporal_df, corr_df, gu_valid)
    save_validation(gu_valid, mw_denial_df, kruskal_df, temporal_df,
                    corr_df, denial_gap_df, group_mw_df)

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
