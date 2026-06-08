"""
🗺️ SHADOW Map — 진단 맵 (Dependency × Avoidance 4분면)
================================================================
서울 자치구를 두 축으로 진단한다.
  · Dependency (편의 의존)  : 비대면 생활 구조에 얼마나 잠겨 있는가
  · Avoidance  (복지 회피)  : 복지를 회피·미연결하는 정도

두 축의 교차로 Q1~Q4 유형을 나누고, 유형별 심리기저·처방방향을 함께 보여준다.
RAG 처방은 SHADOW AI 섹션으로 분리됨(전이예측 결과를 행정동 단위로 받아 수행).
※ 기존 산출물(shadow_index.csv, 편의의 역설/dependency_index.csv)을 '읽기'만 한다.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from kb.prescriptions import KB, Q_COLORS  # noqa: E402

KO_FONT = "Malgun Gothic, Apple SD Gothic Neo, sans-serif"
Q_LABEL = {
    "Q1": "Q1 · 최고위험",
    "Q2": "Q2 · 정서취약",
    "Q3": "Q3 · 상대안전",
    "Q4": "Q4 · 자기해결",
}
Q_STATE = {
    "Q1": "의존 ↑ · 회피 ↑",
    "Q2": "의존 ↑ · 회피 ↓",
    "Q3": "의존 ↓ · 회피 ↓",
    "Q4": "의존 ↓ · 회피 ↑",
}


@st.cache_data
def load_shadow():
    p = ROOT / "Outputs" / "shadow_index.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_dep_dong():
    """행정동별 Dependency (편의의 역설) — 보조 디테일."""
    p = ROOT / "Outputs" / "편의의 역설" / "dependency_index.csv"
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None


shadow = load_shadow()
dep_dong = load_dep_dong()

# ── 헤더 ──────────────────────────────────────────────────────────────
st.markdown(
    "<div class='page-head'><h1>🗺️ SHADOW Map · 진단 맵</h1>"
    "<p>Dependency(편의 의존) × Avoidance(복지 회피) — 두 역설이 겹치는 자리에서 "
    "‘보이지 않는 고립’의 유형을 진단합니다.</p></div>",
    unsafe_allow_html=True,
)

if shadow is None:
    st.warning("shadow_index.csv 가 없어요.")
    st.stop()

sdf = shadow.dropna(subset=["Dependency", "Avoidance", "Quadrant"]).copy()


def split_point(low_qs, high_qs, col):
    lo = sdf[sdf["Quadrant"].isin(low_qs)][col]
    hi = sdf[sdf["Quadrant"].isin(high_qs)][col]
    if len(lo) and len(hi):
        return (lo.max() + hi.min()) / 2
    return sdf[col].median()


dep_split = split_point(["Q3", "Q4"], ["Q1", "Q2"], "Dependency")
avo_split = split_point(["Q2", "Q3"], ["Q1", "Q4"], "Avoidance")

# ── 두 축이란? (정보 보강) ────────────────────────────────────────────
ca, cb = st.columns(2)
ca.markdown(
    "<div class='card'><b>↔ Dependency · 편의 의존</b><br>"
    "<span style='color:#7b8794;font-size:.86rem'>배달·간편식·생활편의 인프라로 "
    "혼자서도 불편 없이 사는 정도. 높을수록 외출·대면의 필요가 줄어 "
    "<b>고립이 보이지 않게 유지</b>됩니다. (편리함의 역설)</span></div>",
    unsafe_allow_html=True,
)
cb.markdown(
    "<div class='card'><b>↕ Avoidance · 복지 회피</b><br>"
    "<span style='color:#7b8794;font-size:.86rem'>복지 수요가 있어도 낙인·불신 때문에 "
    "제도와 연결되지 않는 정도. 높을수록 <b>손 내밀수록 더 숨는</b> 경향입니다. "
    "(복지의 역설)</span></div>",
    unsafe_allow_html=True,
)
st.markdown("")

# ── 분면 카운트 카드 ──────────────────────────────────────────────────
qcounts = sdf["Quadrant"].value_counts()
cc = st.columns(4)
for col, q in zip(cc, ["Q1", "Q2", "Q3", "Q4"]):
    n = int(qcounts.get(q, 0))
    col.markdown(
        f"<div style='border-top:4px solid {Q_COLORS[q]};border-radius:12px;padding:12px;"
        f"background:#fff;box-shadow:0 2px 10px rgba(20,40,80,.05);text-align:center;"
        f"border:1px solid #e8edf3'>"
        f"<div style='font-size:1.8rem;font-weight:800;color:{Q_COLORS[q]}'>{n}</div>"
        f"<div style='font-size:.78rem;color:#1f2d3d;font-weight:700'>{Q_LABEL[q]}</div>"
        f"<div style='font-size:.72rem;color:#7b8794'>{Q_STATE[q]}</div></div>",
        unsafe_allow_html=True,
    )
st.markdown("")

# ── 지도 + 사이드 진단 ────────────────────────────────────────────────
col_map, col_side = st.columns([3, 1.15])
with col_map:
    fig = go.Figure()
    xr = [sdf["Dependency"].min() - 5, sdf["Dependency"].max() + 5]
    yr = [sdf["Avoidance"].min() - 5, sdf["Avoidance"].max() + 5]
    for x0, x1, y0, y1, q in [
        (dep_split, xr[1], avo_split, yr[1], "Q1"),
        (dep_split, xr[1], yr[0], avo_split, "Q2"),
        (xr[0], dep_split, yr[0], avo_split, "Q3"),
        (xr[0], dep_split, avo_split, yr[1], "Q4"),
    ]:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor=Q_COLORS[q], opacity=0.07, line_width=0, layer="below")
    fig.add_vline(x=dep_split, line_dash="dot", line_color="#aaa")
    fig.add_hline(y=avo_split, line_dash="dot", line_color="#aaa")
    for x, y, txt, q, ax, ay in [
        (xr[1], yr[1], "Q1 최고위험", "Q1", "right", "top"),
        (xr[1], yr[0], "Q2 정서취약", "Q2", "right", "bottom"),
        (xr[0], yr[0], "Q3 상대안전", "Q3", "left", "bottom"),
        (xr[0], yr[1], "Q4 자기해결", "Q4", "left", "top"),
    ]:
        fig.add_annotation(x=x, y=y, text=txt, showarrow=False,
                           font=dict(color=Q_COLORS[q], size=12, family=KO_FONT),
                           xanchor=ax, yanchor=ay, opacity=0.75)
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        sub = sdf[sdf["Quadrant"] == q]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["Dependency"], y=sub["Avoidance"],
            mode="markers+text", name=Q_LABEL[q],
            text=sub["자치구"], textposition="top center",
            textfont=dict(size=10, family=KO_FONT),
            marker=dict(size=16, color=Q_COLORS[q], opacity=0.88,
                        line=dict(color="white", width=1.5)),
            customdata=sub[["자치구", "Quadrant"]],
            hovertemplate="<b>%{customdata[0]}</b><br>Dependency %{x:.1f}<br>"
                          "Avoidance %{y:.1f}<br>%{customdata[1]}<extra></extra>",
        ))
    fig.update_layout(
        height=560, paper_bgcolor="white", plot_bgcolor="#fafbfd",
        margin={"t": 10, "b": 40, "l": 10, "r": 10},
        xaxis=dict(title="← Dependency (편의 의존) →", range=xr, gridcolor="#eef2f7"),
        yaxis=dict(title="← Avoidance (복지 회피) →", range=yr, gridcolor="#eef2f7"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                    font=dict(family=KO_FONT, size=10)),
        font=dict(family=KO_FONT),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("점선 = 사분면 경계 · 색 = 진단된 Q유형 · 위치 = 두 지수 좌표")

with col_side:
    st.markdown("#### 🔍 자치구 진단")
    gu_m = st.selectbox("자치구", sorted(sdf["자치구"].unique()), key="diag_gu")
    r = sdf[sdf["자치구"] == gu_m].iloc[0]
    q = str(r["Quadrant"])
    qcol = Q_COLORS.get(q, "#95a5a6")
    st.markdown(
        f"**{gu_m}** <span class='chip' style='background:{qcol}'>{q}</span>",
        unsafe_allow_html=True,
    )
    d1, d2 = st.columns(2)
    d1.metric("Dependency", f"{r['Dependency']:.1f}")
    d2.metric("Avoidance", f"{r['Avoidance']:.1f}")
    if q in KB:
        st.markdown("**심리기저**")
        st.caption(KB[q]["심리기저"])
        st.markdown("**처방 방향**")
        st.caption(KB[q]["처방방향"])
    if dep_dong is not None:
        sub_dd = dep_dong[dep_dong["자치구"] == gu_m].sort_values("Dependency", ascending=False)
        if not sub_dd.empty:
            with st.expander(f"🏘️ {gu_m} 행정동별 의존도 — {len(sub_dd)}개 동"):
                st.dataframe(sub_dd[["행정동", "Dependency"]].round(1).reset_index(drop=True),
                             width="stretch", height=220)
    st.info("💊 행정동 단위 구체 처방 → **SHADOW AI ▸ RAG 처방**")

# ── 4분면 유형 설명 (정보 보강) ───────────────────────────────────────
st.markdown("<div class='sec'>📐 네 가지 진단 유형</div>", unsafe_allow_html=True)
qc = st.columns(4)
for col, q in zip(qc, ["Q1", "Q2", "Q3", "Q4"]):
    kb = KB.get(q, {})
    col.markdown(
        f"<div class='qcard' style='border-top-color:{Q_COLORS[q]}'>"
        f"<div class='qname' style='color:{Q_COLORS[q]}'>{Q_LABEL[q]}</div>"
        f"<div class='qstate'>{Q_STATE[q]} · {qcounts.get(q, 0)}개 자치구</div>"
        f"<div class='qrx'><b>처방 방향</b><br>{kb.get('처방방향', '')}</div></div>",
        unsafe_allow_html=True,
    )

# ── 전체 진단 테이블 ──────────────────────────────────────────────────
with st.expander("📋 자치구별 진단 전체 (25개)"):
    show = sdf[["자치구", "Dependency", "Avoidance", "Quadrant"]].copy()
    show = show.sort_values(["Quadrant", "Avoidance"], ascending=[True, False]).reset_index(drop=True)
    st.dataframe(show.round(1), width="stretch", height=400)
