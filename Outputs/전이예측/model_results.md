# 전이 예측 모델 — 최종 결과

## 모델 성능 (Stratified 5-fold CV)

### label_slope (연도별 Dep slope 양수 + 상위 50%)

| 모델 | AUC | Average Precision | N | 양성 |
|------|-----|-------------------|---|------|
| Logistic Regression | 0.768 | 0.606 | 419 | 146 |
| Gradient Boosting   | 0.759 | 0.628 | 419 | 146 |

### label_entry (기준연도 Q75 미만 → 최신연도 Q75 이상)

| 모델 | AUC | Average Precision |
|------|-----|-------------------|
| Logistic Regression | 0.842 | 0.271 |
| Gradient Boosting   | 0.844 | 0.302 |

> AUC 해석: 0.5=무작위, 0.7+=유의미, 0.8+=좋음. N≈419로 신뢰구간 넓음.

## 홀드아웃 검증 (80/20 무작위 분할, CV와 독립)

### label_slope

| 모델 | 홀드아웃 AUC | 홀드아웃 AP | train N | test N |
|------|------------|-----------|---------|--------|
| Logistic Regression | 0.720 | 0.551 | 335 | 84 |
| Gradient Boosting   | 0.707 | 0.584 | 335 | 84 |

### label_entry

| 모델 | 홀드아웃 AUC | 홀드아웃 AP |
|------|------------|-----------|
| Logistic Regression | 0.887 | 0.391 |
| Gradient Boosting   | 0.889 | 0.467 |

> CV AUC와 홀드아웃 AUC 차이가 작으면 과적합 없음. 큰 차이(0.05+)가 나면 과적합 의심.

## Slope 커버리지 (수정 7: 2점 이상 허용)

- dep_slope_annual 계산 성공: **379개 행정동** / 전체 419개 (90.5%)
- OLS slope 최소 관측치를 3점 → 2점으로 완화하여 데이터 포인트 2개인 행정동도 포함

## 고위험 행정동

- 고위험 + 최고위험 행정동: **103개**
- 상세: transition_predictions.csv

## 피처 중요도 상위 5개

feature_importance.png 참조.
추세·연도별 피처(빨강)가 수준 피처(파랑)보다 중요하게 나오면 → 방향성이 현재 수준보다 예측력 있음.

## 산출물 목록

| 파일 | 내용 |
|------|------|
| transition_features.csv | 행정동별 피처 매트릭스 (13개 피처) |
| yearly_dependency.csv | 연도별 Dependency 궤적 |
| transition_predictions.csv | 행정동별 전이 확률 + 위험등급 |
| gu_transition_score.csv | 자치구 보조 전이 위험점수 |
| feature_importance.png | GradientBoosting 피처 중요도 |
| gu_transition_map.png | 자치구 전이 위험 시각화 |

## 레이블 정의

- **label_slope**: 연도별 Dep slope > 0 AND 상위 50% → 빠르게 악화되는 행정동
- **label_entry**: dep_dep_2022 < Q75 이면서 dep_dep_2025 >= Q75 → Q1 진입

## 한계

- N≈419: max_depth=3, min_samples_leaf=10으로 과적합 억제
- 생태학적 오류: 행정동 집계 수준 → 개인 수준 전이 예측 아님
- Avoidance 축(자치구 단위)은 행정동 모델 미포함 → 자치구 보조 모델에서 보완
- 4년 데이터 기반: 검증 시계열 제한적
