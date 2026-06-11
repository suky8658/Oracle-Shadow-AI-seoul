"""
🗺️ SHADOW Map - 진단 맵 (Dependency × Avoidance 4분면)
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
from theme import KO_FONT  # noqa: E402  (Pretendard 통일)

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
    """행정동별 Dependency (편의의 역설) - 보조 디테일."""
    p = ROOT / "Outputs" / "편의의 역설" / "dependency_index.csv"
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None


shadow = load_shadow()
dep_dong = load_dep_dong()

# ── 헤더 ──────────────────────────────────────────────────────────────
st.markdown(
    "<div class='page-head'><h1>SHADOW Map · 진단 맵</h1>"
    "<p>Dependency(편의 의존) × Avoidance(복지 회피) - 두 역설이 겹치는 자리에서 "
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
ca, cb = st.columns(2, gap="large")
ca.markdown(
    "<div class='card'>"
    "<div style='font-weight:700;font-size:.98rem;margin-bottom:8px'>Dependency · 편의 의존</div>"
    "<div style='color:#8B95A1;font-size:.9rem;line-height:1.6'>배달·간편식·생활편의 인프라로 "
    "혼자서도 불편 없이 사는 정도. 높을수록 외출·대면의 필요가 줄어 "
    "<b style='color:#191F28'>고립이 보이지 않게 유지</b>됩니다.</div></div>",
    unsafe_allow_html=True,
)
cb.markdown(
    "<div class='card'>"
    "<div style='font-weight:700;font-size:.98rem;margin-bottom:8px'>Avoidance · 복지 회피</div>"
    "<div style='color:#8B95A1;font-size:.9rem;line-height:1.6'>복지 수요가 있어도 낙인·불신 때문에 "
    "제도와 연결되지 않는 정도. 높을수록 <b style='color:#191F28'>손 내밀수록 더 숨는</b> "
    "경향입니다.</div></div>",
    unsafe_allow_html=True,
)
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── 분면 카운트 카드 ──────────────────────────────────────────────────
qcounts = sdf["Quadrant"].value_counts()
cc = st.columns(4, gap="medium")
for col, q in zip(cc, ["Q1", "Q2", "Q3", "Q4"]):
    n = int(qcounts.get(q, 0))
    col.markdown(
        f"<div class='grade-card'>"
        f"<div class='num' style='color:{Q_COLORS[q]}'>{n}</div>"
        f"<div class='lbl'>{Q_LABEL[q]}</div>"
        f"<div class='sub'>{Q_STATE[q]}</div></div>",
        unsafe_allow_html=True,
    )
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

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
    fig.add_vline(x=dep_split, line_dash="dot", line_color="#E5E8EB", line_width=1)
    fig.add_hline(y=avo_split, line_dash="dot", line_color="#E5E8EB", line_width=1)
    for x, y, txt, q, ax, ay in [
        (xr[1], yr[1], "Q1 최고위험", "Q1", "right", "top"),
        (xr[1], yr[0], "Q2 정서취약", "Q2", "right", "bottom"),
        (xr[0], yr[0], "Q3 상대안전", "Q3", "left", "bottom"),
        (xr[0], yr[1], "Q4 자기해결", "Q4", "left", "top"),
    ]:
        fig.add_annotation(x=x, y=y, text=txt, showarrow=False,
                           font=dict(color=Q_COLORS[q], size=12, family=KO_FONT),
                           xanchor=ax, yanchor=ay, opacity=0.55)
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        sub = sdf[sdf["Quadrant"] == q]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["Dependency"], y=sub["Avoidance"],
            mode="markers+text", name=Q_LABEL[q],
            text=sub["자치구"], textposition="top center",
            textfont=dict(size=9, family=KO_FONT, color="#8B95A1"),
            marker=dict(size=15, color=Q_COLORS[q], opacity=0.9,
                        line=dict(color="white", width=1.5)),
            customdata=sub[["자치구", "Quadrant"]],
            hovertemplate="<b>%{customdata[0]}</b><br>Dependency %{x:.1f}<br>"
                          "Avoidance %{y:.1f}<br>%{customdata[1]}<extra></extra>",
        ))
    fig.update_layout(
        height=600, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin={"t": 10, "b": 40, "l": 10, "r": 10}, showlegend=False,
        xaxis=dict(title="Dependency (편의 의존) →", range=xr, gridcolor="#E5E8EB",
                   zeroline=False, showline=False,
                   title_font=dict(color="#8B95A1", size=12), tickfont=dict(color="#8B95A1")),
        yaxis=dict(title="Avoidance (복지 회피) →", range=yr, gridcolor="#E5E8EB",
                   zeroline=False, showline=False,
                   title_font=dict(color="#8B95A1", size=12), tickfont=dict(color="#8B95A1")),
        font=dict(family=KO_FONT),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("색 = 진단된 Q유형 · 위치 = 두 지수 좌표")

with col_side:
    st.markdown("<div style='font-size:1.05rem;font-weight:700;margin-bottom:10px'>자치구 진단</div>",
                unsafe_allow_html=True)
    gu_m = st.selectbox("자치구", sorted(sdf["자치구"].unique()), key="diag_gu")
    r = sdf[sdf["자치구"] == gu_m].iloc[0]
    q = str(r["Quadrant"])
    qcol = Q_COLORS.get(q, "#95a5a6")
    st.markdown(
        f"<div style='margin:6px 0 4px'><b style='font-size:1.05rem'>{gu_m}</b> "
        f"<span class='chip' style='background:{qcol}'>{q}</span></div>",
        unsafe_allow_html=True,
    )
    d1, d2 = st.columns(2)
    d1.metric("Dependency", f"{r['Dependency']:.1f}")
    d2.metric("Avoidance", f"{r['Avoidance']:.1f}")
    if q in KB:
        st.markdown("<div style='font-weight:700;font-size:.9rem;margin:14px 0 4px'>심리기저</div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div style='color:#8B95A1;font-size:.86rem;line-height:1.6'>{KB[q]['심리기저']}</div>",
                    unsafe_allow_html=True)
        st.markdown("<div style='font-weight:700;font-size:.9rem;margin:14px 0 4px'>처방 방향</div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div style='color:#8B95A1;font-size:.86rem;line-height:1.6'>{KB[q]['처방방향']}</div>",
                    unsafe_allow_html=True)
    if dep_dong is not None:
        sub_dd = dep_dong[dep_dong["자치구"] == gu_m].sort_values("Dependency", ascending=False)
        if not sub_dd.empty:
            with st.expander(f"{gu_m} 행정동별 의존도 · {len(sub_dd)}개 동"):
                st.dataframe(sub_dd[["행정동", "Dependency"]].round(1).reset_index(drop=True),
                             use_container_width=True, height=220)

# ── 4분면 유형 설명 (정보 보강) ───────────────────────────────────────
st.markdown("<div class='sec'>네 가지 진단 유형</div>", unsafe_allow_html=True)
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
with st.expander("자치구별 진단 전체 (25개)"):
    show = sdf[["자치구", "Dependency", "Avoidance", "Quadrant"]].copy()
    show = show.sort_values(["Quadrant", "Avoidance"], ascending=[True, False]).reset_index(drop=True)
    st.dataframe(show.round(1), use_container_width=True, height=400)
