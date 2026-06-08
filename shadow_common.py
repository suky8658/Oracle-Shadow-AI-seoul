# -*- coding: utf-8 -*-
"""
SHADOW 공통 모듈 — 자치구 단위 집계 · 데이터 로더 · 표시 상수
================================================================
모든 분석 단위를 '자치구'로 통일한다.
전이예측 모델은 행정동 단위(419개)로 학습/예측되어 있으므로, 그 결과를
자치구로 '집계'해서 쓴다. (재학습 없음 — 기존 산출물만 롤업)

  · Q1 전이확률(자치구) = 소속 행정동 전이확률_GB 평균
  · 최고위험 행정동 비율 = 보조 지표
  · SHAP 위험동인(자치구) = 행정동 SHAP 평균 후 상위 요인
"""
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Outputs"

# ── 표시 상수 ─────────────────────────────────────────────────────────
GRADE_COLOR = {"최고위험": "#e74c3c", "고위험": "#e67e22", "중위험": "#f1c40f", "저위험": "#2ecc71"}
GRADE_ORDER = ["최고위험", "고위험", "중위험", "저위험"]
Q_COLORS = {"Q1": "#e74c3c", "Q2": "#e67e22", "Q3": "#3498db", "Q4": "#9b59b6"}
Q_LABEL = {"Q1": "Q1 · 최고위험형", "Q2": "Q2 · 정서취약형",
           "Q3": "Q3 · 상대안전형", "Q4": "Q4 · 자기해결형"}
Q_STATE = {"Q1": "의존 ↑ · 회피 ↑", "Q2": "의존 ↑ · 회피 ↓",
           "Q3": "의존 ↓ · 회피 ↓", "Q4": "의존 ↓ · 회피 ↑"}

FEAT_LABELS = {
    "level_외출커뮤적은": "고립수준", "level_이동횟수": "이동수준",
    "level_배달식재료": "배달의존수준", "level_독거비율": "독거비율",
    "level_인프라": "인프라밀도", "level_인구": "인구규모",
    "delta_외출커뮤적은": "고립 변화속도", "delta_이동횟수": "이동 변화속도",
    "delta_배달식재료": "배달 변화속도", "delta_독거비율": "독거 변화속도",
    "infra_slope": "인프라 증가추세", "infra_delta": "인프라 변화량",
    "infra_recent": "최근 인프라량",
}


def grade_of(score):
    if score >= 80:
        return "최고위험"
    if score >= 65:
        return "고위험"
    if score >= 50:
        return "중위험"
    return "저위험"


# ── 데이터 로더 ───────────────────────────────────────────────────────
@st.cache_data
def load_presc():
    p = OUT / "shadow_ai" / "shadow_prescriptions.csv"
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None


@st.cache_data
def load_shap_values():
    p = OUT / "전이예측" / "shap_values.csv"
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None


@st.cache_data
def load_shadow_index():
    p = OUT / "shadow_index.csv"
    return pd.read_csv(p) if p.exists() else None


# ── 자치구 집계 ───────────────────────────────────────────────────────
@st.cache_data
def aggregate_gu():
    """행정동 전이예측 → 자치구 단위로 롤업한 표를 만든다."""
    presc = load_presc()
    idx = load_shadow_index()
    if presc is None:
        return None
    df = presc.dropna(subset=["전이확률_GB", "Avoidance", "Shadow_Score", "위험등급"]).copy()

    rows = []
    for gu, g in df.groupby("자치구"):
        n = len(g)
        n_top = int((g["위험등급"] == "최고위험").sum())
        q1p = float(g["전이확률_GB"].mean()) * 100          # 자치구 Q1 전이확률(%) = 행정동 평균
        sscore = float(g["Shadow_Score"].mean())
        rows.append({
            "자치구": gu,
            "Q1전이확률": q1p,
            "최고위험비율": n_top / n * 100 if n else 0.0,
            "n_행정동": n,
            "n_최고위험": n_top,
            "Avoidance": float(g["Avoidance"].mean()),
            "ShadowScore평균": sscore,
        })
    gu_df = pd.DataFrame(rows)

    # 등급 = Q1전이확률의 '서울 내 상대순위'(백분위)로 부여 → 랭킹과 색이 일치
    q1 = gu_df["Q1전이확률"]

    def _rel_grade(v):
        pr = float((q1 < v).mean())  # 0~1 백분위
        if pr >= 0.80:
            return "최고위험"
        if pr >= 0.55:
            return "고위험"
        if pr >= 0.30:
            return "중위험"
        return "저위험"

    gu_df["등급"] = gu_df["Q1전이확률"].apply(_rel_grade)

    # SHADOW Map 진단 Q유형 결합 (현재 진단)
    if idx is not None:
        gu_df = gu_df.merge(idx[["자치구", "Quadrant", "Dependency"]], on="자치구", how="left")
    gu_df = gu_df.sort_values("Q1전이확률", ascending=False).reset_index(drop=True)
    return gu_df


@st.cache_data
def gu_top_drivers(gu, k=3):
    """자치구의 SHAP 위험동인 상위 k개 (소속 행정동 SHAP 평균)."""
    shap = load_shap_values()
    if shap is None:
        return []
    g = shap[shap["자치구"] == gu]
    if g.empty:
        return []
    meta = {"행정동", "자치구", "행정동코드", "전이확률"}
    feats = [c for c in shap.columns if c not in meta]
    means = g[feats].mean().sort_values(ascending=False)
    out = []
    for c, v in means.items():
        if v <= 0:
            continue
        out.append((FEAT_LABELS.get(c, c), float(v)))
        if len(out) >= k:
            break
    return out


@st.cache_data
def gu_dong_detail(gu):
    """자치구 내 행정동별 전이예측 상세 (drill-down 근거)."""
    presc = load_presc()
    if presc is None:
        return None
    g = presc[presc["자치구"] == gu].copy()
    cols = ["행정동", "전이확률_GB", "Shadow_Score", "위험등급", "Avoidance"]
    g = g[[c for c in cols if c in g.columns]].sort_values("Shadow_Score", ascending=False)
    g = g.rename(columns={"전이확률_GB": "전이확률"})
    g["전이확률"] = (g["전이확률"] * 100).round(1)
    return g.reset_index(drop=True)
