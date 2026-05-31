"""
전체 파이프라인 오케스트레이터 — run_all.py
==========================================
단 하나의 명령으로 전이 예측 시제품 모델을 처음부터 끝까지 완성.

실행:
  C:/Users/vinvi/anaconda3/python.exe code/run_all.py

단계:
  1. run_transition_pipeline.py  — 데이터 로딩 + 피처 + 레이블 + 모델 학습
  2. save_model.py               — 시제품 모델 저장 (pkl + 메타 + 모델카드)
  3. predict_risk.py             — 최종 예측 실행 + 지도 시각화
"""
import sys, io, subprocess, time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PYTHON = sys.executable
ROOT   = Path(__file__).resolve().parent.parent
CODE   = ROOT / "code"

# (script, 필수여부)  필수=True이면 실패 시 파이프라인 중단
STEPS = [
    ("파이프라인",  CODE / "run_transition_pipeline.py", True),
    ("모델 저장",   CODE / "save_model.py",              True),
    ("예측 실행",   CODE / "predict_risk.py",            True),
    ("SHAP 분석",   CODE / "analyze_shap.py",            False),  # shap 미설치 시 skip
]

print("=" * 60)
print("전이 예측 시제품 — 전체 파이프라인")
print("=" * 60)

total_start = time.time()
all_ok = True

for step_name, script, required in STEPS:
    print(f"\n{'─'*60}")
    print(f"  [{step_name}] {script.name}")
    print(f"{'─'*60}")

    start = time.time()
    result = subprocess.run(
        [PYTHON, str(script)],
        capture_output=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        if required:
            print(f"\n  [ERROR] {step_name} 실패 (exit {result.returncode}, {elapsed:.1f}s)")
            all_ok = False
            break
        else:
            print(f"\n  [SKIP] {step_name} 실패 — 선택 단계이므로 계속 진행 ({elapsed:.1f}s)")
    else:
        print(f"\n  [OK] {step_name} 완료 ({elapsed:.1f}s)")

total = time.time() - total_start
print(f"\n{'=' * 60}")
if all_ok:
    print(f"  모든 단계 완료! 총 {total:.0f}초")
    print(f"  산출물: {ROOT / 'Outputs' / '전이예측'}")
    print()
    print("  주요 파일:")
    OUT = ROOT / "Outputs" / "전이예측"
    key_files = [
        "model/gb_model.pkl",
        "model/model_card.md",
        "transition_predictions.csv",
        "risk_predictions_final.csv",
        "risk_map_final.png",
        "feature_importance.png",
        "gu_transition_map.png",
        "shap_values.csv",
        "shap_top3.csv",
        "shap_bar.png",
        "shap_beeswarm.png",
    ]
    for f in key_files:
        p = OUT / f
        if p.exists():
            print(f"    ✓ {f}  ({p.stat().st_size:,} bytes)")
        else:
            print(f"    ✗ {f}  (없음)")
else:
    print(f"  [실패] {total:.0f}초 경과 — 위 에러 메시지 확인")
print("=" * 60)
