"""
섀도우 AI — 핵심 계산 모듈
=========================
행정동 단위 전이 확률(주축, 가중 0.75) +
자치구 단위 복지 회피 인덱스(보정축, 가중 0.25)를
가중 결합하여 섀도우 처방 점수와 처방 등급을 산출한다.
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "Outputs"

W_DEP   = 0.75
W_AVOID = 0.25

# 행정동 등급 (419개 분포 기준 고정 임계값) — (등급, 하한 점수)
GRADE_THRESHOLDS = [("최고위험", 80), ("고위험", 65), ("중위험", 50), ("저위험", 0)]

# "고위험 행정동" 정의 — 자치구 보조배지(고위험 동 수·비율) 계산용
HIGH_RISK_GRADES = {"최고위험", "고위험"}


def _assign_grade(score: float) -> str:
    for grade, threshold in GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "저위험"


def build_shadow_scores() -> pd.DataFrame:
    # 1. 행정동 전이확률 로드
    pred = pd.read_csv(OUT / "전이예측" / "risk_predictions_final.csv")

    # 2. 전이확률_GB → 0~100 Min-Max 정규화
    p_min, p_max = pred["전이확률_GB"].min(), pred["전이확률_GB"].max()
    pred["전이확률_정규화"] = (pred["전이확률_GB"] - p_min) / (p_max - p_min) * 100

    # 3. 자치구 Avoidance join (행정동 → 자치구 매핑)
    avo = pd.read_csv(OUT / "복지의 역설" / "avoidance_index.csv")[["자치구", "Avoidance"]]
    pred = pred.merge(avo, on="자치구", how="left")

    n_missing = pred["Avoidance"].isna().sum()
    if n_missing:
        fallback = avo["Avoidance"].mean()
        print(f"  [경고] Avoidance 누락 {n_missing}개 행정동 → 평균값({fallback:.1f})으로 대체")
        pred["Avoidance"] = pred["Avoidance"].fillna(fallback)

    # 4. Shadow Score
    pred["Shadow_Score"] = (
        pred["전이확률_정규화"] * W_DEP + pred["Avoidance"] * W_AVOID
    ).round(2)

    # 5. 위험등급
    pred["위험등급"] = pred["Shadow_Score"].apply(_assign_grade)

    cols = [
        "행정동코드", "자치구", "행정동",
        "전이확률_GB", "전이확률_정규화",
        "Avoidance", "Shadow_Score", "위험등급",
        "dep_2025", "dep_slope_annual",
    ]
    return pred[cols].sort_values("Shadow_Score", ascending=False).reset_index(drop=True)


def _recalibrate_gu_grades(scores: pd.Series) -> list:
    """자치구 Shadow_Score 분포의 사분위로 등급 임계값을 재보정한다.

    행정동용 80/65/50은 419개 분포에 맞춘 값이라, 25개 자치구 평균에
    그대로 쓰면 중앙으로 압축돼 '최고위험'이 사라진다.
    → 자치구 분포의 Q75/Q50/Q25를 하한으로 재정의(대략 6/6/6/7 균형).
    반환: [(등급, 하한), ...] 내림차순.
    """
    q75 = float(scores.quantile(0.75))
    q50 = float(scores.quantile(0.50))
    q25 = float(scores.quantile(0.25))
    return [("최고위험", q75), ("고위험", q50), ("중위험", q25), ("저위험", float("-inf"))]


def build_gu_aggregate(df_dong: pd.DataFrame):
    """행정동 점수 → 자치구 단위 집계.

    대표점수(Shadow_Score) = 소속 행정동 Shadow_Score 평균.
      (Avoidance가 자치구 내 상수이므로
       mean(행정동 Shadow) = mean(Dep)×0.75 + Avo×0.25 로 1:1 정의)
    보조지표 = 고위험(최고위험·고위험) 행정동 수·비율.

    반환: (자치구 25행 DataFrame, 재보정 임계값 list)
    """
    rows = []
    for gu, g in df_dong.groupby("자치구"):
        g_sorted   = g.sort_values("Shadow_Score", ascending=False)
        n_dong     = len(g)
        n_high     = int(g["위험등급"].isin(HIGH_RISK_GRADES).sum())
        high_dongs = g_sorted[g_sorted["위험등급"].isin(HIGH_RISK_GRADES)]["행정동"].tolist()
        rows.append({
            "자치구":            gu,
            "전이확률_정규화_평균": round(float(g["전이확률_정규화"].mean()), 2),
            "Avoidance":         round(float(g["Avoidance"].mean()), 2),
            "Shadow_Score":      round(float(g["Shadow_Score"].mean()), 2),
            "행정동수":          n_dong,
            "고위험_행정동수":    n_high,
            "고위험_비율":        round(n_high / n_dong * 100, 1) if n_dong else 0.0,
            "대표행정동":         g_sorted.iloc[0]["행정동"],
            "대표행정동점수":      round(float(g_sorted.iloc[0]["Shadow_Score"]), 2),
            "고위험동_목록":      ";".join(high_dongs),
        })

    gu_df = pd.DataFrame(rows).sort_values("Shadow_Score", ascending=False).reset_index(drop=True)

    # 자치구 분포로 등급 재보정
    thresholds = _recalibrate_gu_grades(gu_df["Shadow_Score"])

    def assign(score):
        for grade, lo in thresholds:
            if score >= lo:
                return grade
        return "저위험"

    gu_df["위험등급"] = gu_df["Shadow_Score"].apply(assign)

    cols = [
        "자치구", "Shadow_Score", "위험등급",
        "전이확률_정규화_평균", "Avoidance",
        "행정동수", "고위험_행정동수", "고위험_비율",
        "대표행정동", "대표행정동점수", "고위험동_목록",
    ]
    return gu_df[cols], thresholds
