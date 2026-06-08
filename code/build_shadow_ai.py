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

# (등급, 하한 점수)
GRADE_THRESHOLDS = [("최고위험", 80), ("고위험", 65), ("중위험", 50), ("저위험", 0)]


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
