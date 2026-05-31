"""
산출물 자동검증 스크립트
========================
코드 수정 후 실행하면 아래 항목을 자동 체크:

1. CSV 정합성: dependency/avoidance/shadow_index.csv 간 수치 일치
2. 하드코딩 검증: 시각화 스크립트 안의 숫자가 실제 데이터와 일치하는지
3. 연도 일관성: 코드/문서에서 데이터 기간 표기가 맞는지
4. 캐시 신선도: 캐시 파일이 원본 CSV보다 오래됐으면 경고
5. 문서 교차검증: methodology/validation md 파일 속 수치

사용: python code/validate_outputs.py
"""

import re
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
OUT = ROOT / "Outputs"
DEP_DIR = OUT / "편의의 역설"
AVO_DIR = OUT / "복지의 역설"

PASS = "  PASS"
FAIL = "  FAIL"
WARN = "  WARN"

n_pass, n_fail, n_warn = 0, 0, 0


def check(condition, msg_pass, msg_fail, level="fail"):
    global n_pass, n_fail, n_warn
    if condition:
        print(f"{PASS}  {msg_pass}")
        n_pass += 1
    else:
        if level == "warn":
            print(f"{WARN}  {msg_fail}")
            n_warn += 1
        else:
            print(f"{FAIL}  {msg_fail}")
            n_fail += 1


# ══════════════════════════════════════════════════════════════════
# [1] CSV 정합성
# ══════════════════════════════════════════════════════════════════
def check_csv_consistency():
    print("\n== [1] CSV 정합성 ==")

    dep = pd.read_csv(DEP_DIR / "dependency_index.csv")
    avo = pd.read_csv(AVO_DIR / "avoidance_index.csv")
    shadow = pd.read_csv(OUT / "shadow_index.csv")

    # shadow_index가 최신 dep/avo를 반영하는지
    dep_gus = sorted(dep["자치구"].unique())
    shadow_gus = sorted(shadow["자치구"].unique())
    avo_gus = sorted(avo["자치구"].unique())

    check(set(shadow_gus) == set(avo_gus),
          f"shadow 자치구 = avoidance 자치구 ({len(shadow_gus)}개)",
          f"shadow 자치구 != avoidance 자치구")

    check(len(shadow) == 25,
          "shadow_index: 25개 자치구",
          f"shadow_index: {len(shadow)}개 자치구 (25 예상)")

    # Avoidance 수치 일치 확인
    for _, row in shadow.iterrows():
        avo_row = avo[avo["자치구"] == row["자치구"]]
        if len(avo_row) > 0:
            avo_val = avo_row["Avoidance"].values[0]
            if abs(row["Avoidance"] - avo_val) > 0.1:
                check(False, "",
                      f"shadow vs avoidance 불일치: {row['자치구']} "
                      f"(shadow={row['Avoidance']:.1f}, avo={avo_val:.1f})")
                return
    check(True, "shadow Avoidance = avoidance_index Avoidance 일치", "")

    # Dependency 행정동 수 확인
    n_dep = len(dep)
    check(n_dep > 400,
          f"dependency_index: {n_dep}개 행정동",
          f"dependency_index: {n_dep}개 행정동 (400+ 예상)")


# ══════════════════════════════════════════════════════════════════
# [2] 하드코딩 검증
# ══════════════════════════════════════════════════════════════════
def check_hardcoded_values():
    print("\n== [2] 하드코딩 검증 ==")

    dep = pd.read_csv(DEP_DIR / "dependency_index.csv")
    avo = pd.read_csv(AVO_DIR / "avoidance_index.csv")
    shadow = pd.read_csv(OUT / "shadow_index.csv")

    n_dep_dongs = len(dep)
    dep_med = shadow["Dependency"].median()
    avo_med = shadow["Avoidance"].median()

    # visualize_shadow.py 하드코딩 확인
    shadow_code = (CODE / "visualize_shadow.py").read_text(encoding="utf-8")

    # r 값 확인 (build_dependency_index 결과에서)
    dep_method = (DEP_DIR / "dependency_methodology.md").read_text(encoding="utf-8")
    r_match = re.search(r"Spearman r = ([0-9.]+)", dep_method)
    actual_r = r_match.group(1) if r_match else "?"

    partial_match = re.search(r"부분상관: r = ([0-9.]+)", dep_method)
    actual_partial = partial_match.group(1) if partial_match else "?"

    # shadow 코드 안의 r 값
    shadow_r_matches = re.findall(r'r = \+([0-9.]+)', shadow_code)
    if shadow_r_matches:
        code_r = shadow_r_matches[0]
        check(code_r == actual_r,
              f"visualize_shadow.py Spearman r={code_r} = 실제 {actual_r}",
              f"visualize_shadow.py r={code_r} != 실제 r={actual_r}")
    if len(shadow_r_matches) > 1:
        code_partial = shadow_r_matches[1]
        check(code_partial == actual_partial,
              f"visualize_shadow.py 부분상관 r={code_partial} = 실제 {actual_partial}",
              f"visualize_shadow.py 부분상관 r={code_partial} != 실제 r={actual_partial}")

    # 부정 갭 확인: "부정 갭" 근처의 +0.xxx 패턴만 매칭
    avo_valid = (AVO_DIR / "avoidance_validation.md").read_text(encoding="utf-8")
    gap_match = re.search(r'남_5060.*?([+-][0-9.]+)', avo_valid)
    actual_gap = gap_match.group(1) if gap_match else "?"

    # shadow 코드에서 "부정 갭" 라인의 숫자만 추출
    gap_line_match = re.search(r'부정 갭.*?\+([0-9.]+)', shadow_code)
    if gap_line_match:
        code_gap = gap_line_match.group(1)
        check(f"+{code_gap}" == actual_gap or code_gap == actual_gap.lstrip("+"),
              f"visualize_shadow.py 부정갭={code_gap} = 실제 {actual_gap}",
              f"visualize_shadow.py 부정갭={code_gap} != 실제 {actual_gap}")

    # 행정동 수 확인
    dong_matches = re.findall(r'행정동 (\d+)개', shadow_code)
    if dong_matches:
        code_dongs = int(dong_matches[0])
        check(code_dongs == n_dep_dongs,
              f"visualize_shadow.py 행정동={code_dongs} = 실제 {n_dep_dongs}",
              f"visualize_shadow.py 행정동={code_dongs} != 실제 {n_dep_dongs}")

    # visualize_dependency.py 연도 확인
    dep_code = (CODE / "visualize_dependency.py").read_text(encoding="utf-8")
    has_2025_store = "2025" in dep_code and "점포" in dep_code
    has_2025_telecom = "2025" in dep_code and "통신" in dep_code
    check(has_2025_store,
          "visualize_dependency.py: 상권 2025 포함",
          "visualize_dependency.py: 상권에 2025 누락", level="warn")
    check(has_2025_telecom,
          "visualize_dependency.py: 통신정보 2025 포함",
          "visualize_dependency.py: 통신정보에 2025 누락", level="warn")

    # visualize_avoidance.py 2025 가구주 확인
    avo_code = (CODE / "visualize_avoidance.py").read_text(encoding="utf-8")
    has_2025_hh = "2025" in avo_code and "가구주" in avo_code
    check(has_2025_hh,
          "visualize_avoidance.py: 가구주 2025 포함",
          "visualize_avoidance.py: 가구주에 2025 누락", level="warn")


# ══════════════════════════════════════════════════════════════════
# [3] 연도 일관성
# ══════════════════════════════════════════════════════════════════
def check_year_consistency():
    print("\n== [3] 연도 일관성 ==")

    # build_dependency_index.py
    dep_code = (CODE / "build_dependency_index.py").read_text(encoding="utf-8")
    dep_years_29 = re.findall(r'_29개_통신정보.*?\[([^\]]+)\]', dep_code)
    dep_years_10 = re.findall(r'_10개_관심집단.*?\[([^\]]+)\]', dep_code)
    dep_store = re.findall(r'점포.*?(\d{4})', dep_code)

    has_2025_in_dep = "2025" in dep_code
    check(has_2025_in_dep,
          "build_dependency_index.py: 2025 포함됨",
          "build_dependency_index.py: 2025 누락!")

    # build_avoidance_index.py
    avo_code = (CODE / "build_avoidance_index.py").read_text(encoding="utf-8")
    has_2025_hh = "2025" in avo_code and "가구주" in avo_code
    check(has_2025_hh,
          "build_avoidance_index.py: 2025 가구주 포함됨",
          "build_avoidance_index.py: 2025 가구주 누락!")

    # 시간안정성 라벨
    temporal_label = re.search(r'시간적 안정성 검증 \((\d{4}) vs (\d{4})\)', avo_code)
    if temporal_label:
        y1, y2 = temporal_label.group(1), temporal_label.group(2)
        check(y2 == "2025",
              f"시간안정성 라벨: {y1} vs {y2}",
              f"시간안정성 라벨이 낡음: {y1} vs {y2} (최신 연도 반영 필요)")

    # methodology 문서 연도
    dep_method = (DEP_DIR / "dependency_methodology.md").read_text(encoding="utf-8")
    check("2025" in dep_method,
          "dependency_methodology.md: 2025 언급됨",
          "dependency_methodology.md: 2025 언급 없음", level="warn")

    avo_method = (AVO_DIR / "avoidance_methodology.md").read_text(encoding="utf-8")
    check("2025" in avo_method,
          "avoidance_methodology.md: 2025 언급됨",
          "avoidance_methodology.md: 2025 언급 없음", level="warn")


# ══════════════════════════════════════════════════════════════════
# [4] 캐시 신선도
# ══════════════════════════════════════════════════════════════════
def check_cache_freshness():
    print("\n== [4] 캐시 신선도 ==")

    dep_csv = DEP_DIR / "dependency_index.csv"
    avo_csv = AVO_DIR / "avoidance_index.csv"
    shadow_csv = OUT / "shadow_index.csv"

    if not dep_csv.exists() or not avo_csv.exists():
        check(False, "", "dependency/avoidance CSV가 없음!")
        return

    dep_time = dep_csv.stat().st_mtime
    avo_time = avo_csv.stat().st_mtime

    # shadow_index.csv가 dep/avo보다 새로운지
    if shadow_csv.exists():
        shadow_time = shadow_csv.stat().st_mtime
        check(shadow_time >= dep_time - 60 and shadow_time >= avo_time - 60,
              f"shadow_index.csv가 dep/avo CSV보다 새로움 또는 동시 생성",
              f"shadow_index.csv가 dep/avo CSV보다 오래됨! 재생성 필요",
              level="warn")

    # 캐시 파일 확인
    for cache_path in DEP_DIR.glob("*.pkl"):
        cache_time = cache_path.stat().st_mtime
        if cache_time < dep_time - 60:
            check(False, "",
                  f"캐시 {cache_path.name}이 CSV보다 오래됨 — 삭제 후 재실행 권장",
                  level="warn")
        else:
            check(True,
                  f"캐시 {cache_path.name} 신선도 OK", "")


# ══════════════════════════════════════════════════════════════════
# [5] 구성요소 완전성
# ══════════════════════════════════════════════════════════════════
def check_component_completeness():
    print("\n== [5] 구성요소 완전성 ==")

    dep_code = (CODE / "visualize_dependency.py").read_text(encoding="utf-8")

    # Top10 프로파일에 5개 구성요소 다 있는지
    dep_components = re.findall(r'"(\w+_n)", "([^"]+)", ', dep_code)
    expected = {"conv_total_n", "absence_n", "delivery_n", "weekday_move_n", "solo_ratio_n"}
    found = {c[0] for c in dep_components}

    check(expected.issubset(found),
          f"Top10 프로파일: 5개 구성요소 전부 포함 ({len(found)}개)",
          f"Top10 프로파일: 구성요소 누락 — 있는 것: {found}, 없는 것: {expected - found}")

    # 가중치 합 = 1.0 확인
    weight_matches = re.findall(r'weights = \[([^\]]+)\]', dep_code)
    if weight_matches:
        weights = [float(w.strip()) for w in weight_matches[0].split(",")]
        wsum = sum(weights)
        check(abs(wsum - 1.0) < 0.01,
              f"Top10 프로파일 가중치 합 = {wsum:.2f}",
              f"Top10 프로파일 가중치 합 = {wsum:.2f} (1.0 아님!)")


# ══════════════════════════════════════════════════════════════════
# [6] 민감도 분석 해석 검증
# ══════════════════════════════════════════════════════════════════
def check_sensitivity_report():
    print("\n== [6] 민감도 분석 리포트 ==")

    report_path = OUT / "sensitivity_analysis.md"
    if not report_path.exists():
        check(False, "", "sensitivity_analysis.md 없음!")
        return

    report = report_path.read_text(encoding="utf-8")

    # "매우 안정"이 Avoidance 결론에 대해 쓰이면 경고
    # Avoidance 결론 섹션만 추출 (Avoidance Index ~ 해석 사이)
    parts = report.split("## ")
    avo_conclusion = ""
    for part in parts:
        if part.startswith("Avoidance"):
            avo_conclusion = part.split("**결론**:")[-1].split("\n")[0] if "**결론**" in part else ""
            break
    check("매우 안정" not in avo_conclusion,
          "민감도 리포트: Avoidance 결론에 '매우 안정' 미사용 (적절)",
          "민감도 리포트: Avoidance 결론에 '매우 안정' 사용됨 — 과장 우려",
          level="warn")

    # rho < 0.90 시나리오 언급 여부
    check("0.889" in report or "0.888" in report or "민감" in avo_section,
          "민감도 리포트: 복지불신 강화 시 rho<0.90 언급됨",
          "민감도 리포트: rho<0.90 시나리오 미언급 — 보수적 해석 필요",
          level="warn")


# ══════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("산출물 자동검증")
    print(f"검증 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    check_csv_consistency()
    check_hardcoded_values()
    check_year_consistency()
    check_cache_freshness()
    check_component_completeness()
    check_sensitivity_report()

    print("\n" + "=" * 60)
    print(f"결과: PASS={n_pass}  FAIL={n_fail}  WARN={n_warn}")
    if n_fail > 0:
        print(">>> FAIL 있음 — 발표 전 반드시 수정 필요!")
    elif n_warn > 0:
        print(">>> WARN 있음 — 확인 권장")
    else:
        print(">>> 전부 PASS — 발표 준비 완료")
    print("=" * 60)

    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
