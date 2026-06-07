# 모델 카드 — 서울 5060 남성 1인가구 Q1 전이 예측 모델

## 버전
- 생성일: 2026-06-04 17:15
- 데이터: 서울시 통신정보 2022~2025 (4년)

## 문제 정의

서울 5060 남성 1인가구 행정동 단위에서,
현재 상대적으로 낮은 Dependency 수준에 있는 행정동이
**향후 Q1(최고위험군)으로 전이될 확률**을 예측.

## 레이블 정의

**label_entry** (주 레이블):
- 양성(1): 기준연도 Dep < 전역 Q75 이면서 최신연도 Dep >= 전역 Q75 → Q1으로 전이
- 음성(0): 그 외 (이미 Q1이었거나, 전이 없음)
- 양성 비율: 10.6%
- 전역 정규화 기반 점수를 사용하므로 연도 간 비교 유효

## 모델 성능 (Stratified 5-fold CV)

| 모델 | AUC | Average Precision |
|------|-----|-------------------|
| **Gradient Boosting (권장)** | **0.844** | **0.302** |
| Logistic Regression (기준선) | 0.842 | 0.271 |

- **AUC 0.844**: 우수 (0.8+)
- N=359으로 작아 신뢰구간 넓음 — 방향성 참고용으로 해석

## 입력 피처 (13개)

| 그룹 | 피처 예시 | 의미 |
|------|---------|------|
| level_ | level_외출커뮤적은 | 4년 평균 행동 수준 (현재 상태) |
| level_ | level_배달식재료 | 배달 의존도 수준 |
| level_ | level_이동횟수 | 평일 이동 횟수 수준 |
| level_ | level_독거비율 | 독거 비율 수준 |
| infra_ | infra_slope | 편의 인프라 점포수 추세 |
| infra_ | infra_delta | 최근 4분기 vs 초기 4분기 점포수 변화 |

※ dep_20XX / dep_slope_annual 은 레이블 정의에만 사용 (피처 제외, 누출 방지)

## 피처 중요도 상위 5개

  1. `infra_recent` (0.1969)
  2. `level_외출커뮤적은` (0.1807)
  3. `level_독거비율` (0.1396)
  4. `level_인프라` (0.1111)
  5. `level_이동횟수` (0.0730)

## 출력

- `전이확률_GB` (0~1): GradientBoosting 예측 확률
- `전이확률_LR` (0~1): Logistic Regression 예측 확률
- `위험등급`: 저위험(0~0.3) / 중위험(0.3~0.5) / 고위험(0.5~0.7) / 최고위험(0.7~1.0)

## 사용 방법

```bash
C:/Users/vinvi/anaconda3/python.exe code/predict_risk.py
```

또는 새 행정동 데이터 입력:
```python
from predict_risk import predict_dong
result = predict_dong(transition_features_csv_path)
```

## 한계

- **분석 단위**: 행정동(집계) — 개인 수준 예측 아님 (생태학적 오류 주의)
- **Avoidance 미포함**: 자치구 보조 모델(`gu_transition_score.csv`) 별도 참조
- **시계열 제한**: 4년 데이터, 검증 횟수 부족
- **N=419**: 과적합 억제 위해 max_depth=3, min_samples_leaf=10

## 파일 목록

```
Outputs/전이예측/
├── model/
│   ├── gb_model.pkl          ← GradientBoosting (권장)
│   ├── lr_model.pkl          ← LogisticRegression (기준선)
│   ├── model_meta.json       ← 피처 목록, 성능 지표, 파라미터
│   └── feat_stats.json       ← 피처 분포 통계
├── transition_features.csv   ← 학습 피처
├── yearly_dependency.csv     ← 연도별 Dep 궤적
├── transition_predictions.csv ← 행정동별 전이 확률
├── gu_transition_score.csv   ← 자치구 보조 위험점수
├── feature_importance.png    ← 피처 중요도 차트
├── gu_transition_map.png     ← 자치구 전이 위험 지도
└── model_results.md          ← 분석 결과 리포트
```
