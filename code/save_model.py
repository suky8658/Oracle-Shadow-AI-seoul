"""
시제품 모델 저장 — save_model.py
run_transition_pipeline.py 실행 후 호출.
학습된 GradientBoosting 모델을 pickle로 저장하고 모델 카드를 생성.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
)
import warnings
warnings.filterwarnings("ignore")

ROOT   = Path(__file__).resolve().parent.parent
OUT    = ROOT / "Outputs" / "전이예측"
MODEL_DIR = OUT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("시제품 모델 저장")
print("=" * 60)

# ── 데이터 로딩 ──────────────────────────────────────────────────────
feat_path = OUT / "transition_features.csv"
if not feat_path.exists():
    raise FileNotFoundError(f"{feat_path} 없음 — run_transition_pipeline.py 먼저 실행")

feat = pd.read_csv(feat_path, encoding="utf-8-sig")
print(f"  피처 로드: {feat.shape}")
print(f"  컬럼: {list(feat.columns)}")

# 피처: 조기 수준(level_) + 조기 트렌드(delta_) + 인프라 추세(infra_)
# dep_20XX / dep_slope_annual 은 레이블 정의에 쓰이므로 제외
FEAT_PREFIXES = ["level_", "delta_", "infra_"]
FEAT_COLS = [c for c in feat.columns
             if any(c.startswith(p) for p in FEAT_PREFIXES)]
META_COLS = ["행정동코드", "자치구", "행정동"]

# label_entry: 전역 Q75 기준 기준연도 미달 → 최신연도 초과 (Q1 진입, 주 모델)
# label_slope: Dep slope 양수+상위50% (빠른 악화 방향)
LABEL_COL = "label_entry"

print(f"  사용 피처: {FEAT_COLS}")

sub = feat.dropna(subset=[LABEL_COL]).copy()
y   = sub[LABEL_COL].astype(int)
X   = sub[FEAT_COLS].copy()
meta = sub[META_COLS].copy()

print(f"  학습 데이터: N={len(sub)}, 양성={y.sum()} ({y.mean():.1%})")

# ── 모델 정의 ───────────────────────────────────────────────────────
GB_PARAMS = dict(
    n_estimators=100, max_depth=3, learning_rate=0.05,
    min_samples_leaf=10, subsample=0.8, random_state=42,
)
LR_PARAMS = dict(C=0.1, max_iter=2000, class_weight="balanced", random_state=42)

gb_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("clf", GradientBoostingClassifier(**GB_PARAMS)),
])
lr_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc",  StandardScaler()),
    ("clf", LogisticRegression(**LR_PARAMS)),
])

# ── CV 성능 측정 ─────────────────────────────────────────────────────
print("\n  교차검증 성능 측정...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_metrics = {}

for name, pipe in [("GradientBoosting", gb_pipe), ("LogisticRegression", lr_pipe)]:
    prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    auc  = roc_auc_score(y, prob)
    ap   = average_precision_score(y, prob)
    pred = (prob >= 0.5).astype(int)
    cm   = confusion_matrix(y, pred).tolist()
    report_str = classification_report(y, pred, target_names=["안정","전이"], zero_division=0)
    cv_metrics[name] = {
        "auc": round(auc, 4), "ap": round(ap, 4),
        "confusion_matrix": cm,
    }
    print(f"  [{name}] AUC={auc:.3f}  AP={ap:.3f}")
    print("  " + report_str.replace("\n", "\n  "))

# ── 전체 데이터로 최종 학습 (시제품 모델) ────────────────────────────
print("\n  전체 데이터로 최종 학습...")
gb_pipe.fit(X, y)
lr_pipe.fit(X, y)

# 피처 중요도
imp = pd.Series(
    gb_pipe["clf"].feature_importances_, index=FEAT_COLS
).sort_values(ascending=False)
print(f"  상위 5 피처: {list(imp.head(5).index)}")

# ── 저장 ─────────────────────────────────────────────────────────────
print("\n  모델 저장 중...")

# 1. 모델 피클
with open(MODEL_DIR / "gb_model.pkl", "wb") as f:
    pickle.dump(gb_pipe, f)
with open(MODEL_DIR / "lr_model.pkl", "wb") as f:
    pickle.dump(lr_pipe, f)

# 2. 피처 정보 저장 (predict_risk.py에서 사용)
model_meta = {
    "feat_cols":   FEAT_COLS,
    "meta_cols":   META_COLS,
    "label_col":   LABEL_COL,
    "cv_metrics":  cv_metrics,
    "feat_importance": imp.to_dict(),
    "trained_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "n_train":     int(len(sub)),
    "n_positive":  int(y.sum()),
    "positive_rate": round(float(y.mean()), 4),
    "gb_params":   GB_PARAMS,
    "lr_params":   LR_PARAMS,
    "threshold":   0.5,
    "grade_bins": {"저위험": [0, 0.30], "중위험": [0.30, 0.50],
                   "고위험": [0.50, 0.70], "최고위험": [0.70, 1.01]},
}
with open(MODEL_DIR / "model_meta.json", "w", encoding="utf-8") as f:
    json.dump(model_meta, f, ensure_ascii=False, indent=2)

# 3. 스케일러 / 피처 통계 저장 (새 데이터 입력 시 대조용)
feat_stats = X.describe().to_dict()
with open(MODEL_DIR / "feat_stats.json", "w", encoding="utf-8") as f:
    json.dump(feat_stats, f, ensure_ascii=False, indent=2)

# 4. 모델 카드
best = "GradientBoosting" if cv_metrics["GradientBoosting"]["auc"] >= cv_metrics["LogisticRegression"]["auc"] else "LogisticRegression"
gb_auc = cv_metrics["GradientBoosting"]["auc"]
lr_auc = cv_metrics["LogisticRegression"]["auc"]
gb_ap  = cv_metrics["GradientBoosting"]["ap"]
lr_ap  = cv_metrics["LogisticRegression"]["ap"]
top5 = "\n".join(f"  {i+1}. `{c}` ({v:.4f})" for i, (c, v) in enumerate(imp.head(5).items()))

card = f"""\
# 모델 카드 — 서울 5060 남성 1인가구 Q1 전이 예측 모델

## 버전
- 생성일: {datetime.now().strftime("%Y-%m-%d %H:%M")}
- 데이터: 서울시 통신정보 2022~2025 (4년)

## 문제 정의

서울 5060 남성 1인가구 행정동 단위에서,
현재 상대적으로 낮은 Dependency 수준에 있는 행정동이
**향후 Q1(최고위험군)으로 전이될 확률**을 예측.

## 레이블 정의

**label_entry** (주 레이블):
- 양성(1): 기준연도 Dep < 전역 Q75 이면서 최신연도 Dep >= 전역 Q75 → Q1으로 전이
- 음성(0): 그 외 (이미 Q1이었거나, 전이 없음)
- 양성 비율: {y.mean():.1%}
- 전역 정규화 기반 점수를 사용하므로 연도 간 비교 유효

## 모델 성능 (Stratified 5-fold CV)

| 모델 | AUC | Average Precision |
|------|-----|-------------------|
| **Gradient Boosting (권장)** | **{gb_auc:.3f}** | **{gb_ap:.3f}** |
| Logistic Regression (기준선) | {lr_auc:.3f} | {lr_ap:.3f} |

- **AUC {gb_auc:.3f}**: {'우수 (0.8+)' if gb_auc >= 0.8 else '양호 (0.7+)' if gb_auc >= 0.7 else '개선 필요 (0.6+)' if gb_auc >= 0.6 else '재검토 필요'}
- N={len(sub)}으로 작아 신뢰구간 넓음 — 방향성 참고용으로 해석

## 입력 피처 ({len(FEAT_COLS)}개)

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

{top5}

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
"""
(MODEL_DIR / "model_card.md").write_text(card, encoding="utf-8")

print(f"\n  저장 완료:")
for p in sorted(MODEL_DIR.iterdir()):
    print(f"    {p.name}  ({p.stat().st_size:,} bytes)")

print(f"\n  최우선 모델: {best} (AUC={cv_metrics[best]['auc']:.3f})")
print("\n  다음 단계: python code/predict_risk.py")
