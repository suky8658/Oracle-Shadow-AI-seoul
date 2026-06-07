"""
섀도우 AI — 실행 엔트리포인트
============================
처방 점수를 산출하고 CSV + 요약 MD를 저장한다.

실행:
  python code/run_shadow_ai.py
"""
import sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from build_shadow_ai import build_shadow_scores, build_gu_aggregate, W_DEP, W_AVOID

OUT_DIR = ROOT / "Outputs" / "shadow_ai"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRADE_DESC = {"최고위험": "즉시 개입", "고위험": "고위험 모니터링", "중위험": "예방 프로그램", "저위험": "정기 안내"}


def main():
    print("=" * 60)
    print("섀도우 AI — 처방 점수 산출")
    print("=" * 60)

    df = build_shadow_scores()

    # CSV 저장 (행정동 단위 — 동 근거)
    csv_path = OUT_DIR / "shadow_prescriptions.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  저장: {csv_path}")
    print(f"  행정동 수: {len(df)}")

    # 자치구 단위 집계 (화면·RAG 입력용)
    gu_df, thresholds = build_gu_aggregate(df)
    gu_path = OUT_DIR / "shadow_gu_aggregate.csv"
    gu_df.to_csv(gu_path, index=False, encoding="utf-8-sig")
    print(f"  저장: {gu_path}")
    print(f"  자치구 수: {len(gu_df)}  "
          f"(등급 재보정 임계값  최고위험≥{thresholds[0][1]:.1f} · "
          f"고위험≥{thresholds[1][1]:.1f} · 중위험≥{thresholds[2][1]:.1f})")

    # 위험등급 분포
    print("\n  위험등급 분포 (행정동):")
    for grade in ["최고위험", "고위험", "중위험", "저위험"]:
        n = (df["위험등급"] == grade).sum()
        print(f"    {grade} ({GRADE_DESC[grade]}): {n}개 행정동")
    print("\n  위험등급 분포 (자치구, 재보정):")
    for grade in ["최고위험", "고위험", "중위험", "저위험"]:
        n = (gu_df["위험등급"] == grade).sum()
        print(f"    {grade}: {n}개 자치구")

    # 상위 10 출력
    print("\n  상위 10 행정동:")
    for _, row in df.head(10).iterrows():
        print(
            f"    [{row['위험등급']}] {row['자치구']} {row['행정동']}"
            f"  Shadow={row['Shadow_Score']:.1f}"
            f"  전이확률={row['전이확률_GB']:.3f}"
            f"  Avoidance={row['Avoidance']:.1f}"
        )

    _write_summary(df, gu_df, thresholds)
    print("\n완료!")
    return df


def _write_gu_section(gu_df, thresholds):
    lines = [
        "",
        "## 자치구 단위 집계 (대표점수 = 소속 행정동 Shadow Score 평균)",
        "",
        f"등급 임계값(자치구 분포 재보정): "
        f"최고위험 ≥ {thresholds[0][1]:.1f} · 고위험 ≥ {thresholds[1][1]:.1f} · "
        f"중위험 ≥ {thresholds[2][1]:.1f}",
        "",
        "| 순위 | 자치구 | Shadow Score | 위험등급 | 고위험 행정동 | 대표 행정동 |",
        "|---|---|---|---|---|---|",
    ]
    for i, (_, r) in enumerate(gu_df.iterrows(), 1):
        lines.append(
            f"| {i} | {r['자치구']} | {r['Shadow_Score']:.1f} | {r['위험등급']} "
            f"| {r['고위험_행정동수']}/{r['행정동수']} ({r['고위험_비율']:.0f}%) "
            f"| {r['대표행정동']} ({r['대표행정동점수']:.1f}) |"
        )
    return lines


def _write_summary(df: "pd.DataFrame", gu_df=None, thresholds=None):
    top30 = df.head(30)
    lines = [
        "# 섀도우 AI 처방 요약",
        "",
        f"**생성일:** 2026-05-31  ",
        f"**대상:** 서울시 5060 남성 1인가구 행정동 {len(df)}개  ",
        f"**가중치:** 전이확률 {int(W_DEP * 100)}% + Avoidance {int(W_AVOID * 100)}%",
        "",
        "## 위험등급 분포",
        "",
        "| 등급 | 의미 | 행정동 수 |",
        "|---|---|---|",
    ]
    for grade in ["최고위험", "고위험", "중위험", "저위험"]:
        n = (df["위험등급"] == grade).sum()
        lines.append(f"| {grade} | {GRADE_DESC[grade]} | {n} |")

    lines += [
        "",
        "## 상위 30 행정동",
        "",
        "| 순위 | 자치구 | 행정동 | Shadow Score | 위험등급 | 전이확률_GB | Avoidance |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, (_, row) in enumerate(top30.iterrows(), 1):
        lines.append(
            f"| {i} | {row['자치구']} | {row['행정동']} | {row['Shadow_Score']:.1f} "
            f"| {row['위험등급']} | {row['전이확률_GB']:.3f} | {row['Avoidance']:.1f} |"
        )

    if gu_df is not None and thresholds is not None:
        lines += _write_gu_section(gu_df, thresholds)

    path = OUT_DIR / "shadow_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"  저장: {path}")


if __name__ == "__main__":
    main()
