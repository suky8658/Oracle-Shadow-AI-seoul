"""
🔮 SHADOW AI ▸ 전이예측 (자치구 단위)
================================================================
행정동 단위로 학습된 전이예측 모델(Gradient Boosting)의 결과를 '자치구'로 집계해,
각 자치구가 Q1 고위험군으로 전이할 확률을 보여준다.
  · 자치구 Q1 전이확률 = 소속 행정동 전이확률 평균
  · 행정동 상세는 drill-down 으로 근거 제공 (단위는 자치구로 통일)
선택한 자치구는 RAG 처방 페이지로 그대로 이어진다.
"""
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shadow_common import (  # noqa: E402
    aggregate_gu, gu_top_drivers, gu_dong_detail,
    GRADE_COLOR, GRADE_ORDER, Q_COLORS, Q_LABEL,
)

KO_FONT = "Malgun Gothic, Apple SD Gothic Neo, sans-serif"

st.markdown("<div class='page-head'><h1>🔮 전이예측 · 자치구</h1>"
            "<p>행정동 전이예측 모델(GBM)을 자치구로 집계 - 각 자치구가 "
            "<b>Q1 고위험군으로 전이할 확률</b>을 선제적으로 봅니다.</p></div>",
            unsafe_allow_html=True)

gdf = aggregate_gu()
if gdf is None:
    st.error("전이예측 결과(Outputs/shadow_ai/shadow_prescriptions.csv)가 없어요.")
    st.stop()

# ── 등급 요약 카드 ────────────────────────────────────────────────────
cc = st.columns(4)
for col, grade in zip(cc, GRADE_ORDER):
    n = int((gdf["등급"] == grade).sum())
    color = GRADE_COLOR[grade]
    col.markdown(
        f"<div class='grade-card' style='border-top-color:{color}'>"
        f"<div class='num' style='color:{color}'>{n}</div>"
        f"<div class='lbl'>{grade} 자치구</div></div>",
        unsafe_allow_html=True,
    )
st.markdown("")

col_rank, col_pick = st.columns([2.1, 1])

# ── 자치구 Q1 전이확률 랭킹 ───────────────────────────────────────────
with col_rank:
    st.markdown("<div class='sec'>자치구별 Q1 전이확률</div>", unsafe_allow_html=True)
    r = gdf.sort_values("Q1전이확률", ascending=True)
    fig = go.Figure(go.Bar(
        x=r["Q1전이확률"], y=r["자치구"], orientation="h",
        marker_color=[GRADE_COLOR[g] for g in r["등급"]],
        text=[f"{v:.1f}%" for v in r["Q1전이확률"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Q1 전이확률 %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        height=max(420, len(r) * 23),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin={"t": 10, "b": 10, "l": 0, "r": 50},
        xaxis=dict(title="Q1 전이확률 (%)", gridcolor="#eef2f7"),
        font=dict(family=KO_FONT, size=11),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("색 = 전이위험 등급(서울 25개 자치구 내 Q1전이확률 상대순위) · 값 = 소속 행정동 전이확률 평균")

# ── 자치구 선택 → 상세 ────────────────────────────────────────────────
with col_pick:
    st.markdown("<div class='sec'>🔍 자치구 선택</div>", unsafe_allow_html=True)
    gu_list = gdf["자치구"].tolist()
    default = st.session_state.get("sel_gu", gu_list[0])
    if default not in gu_list:
        default = gu_list[0]
    gu = st.selectbox("자치구", gu_list, index=gu_list.index(default), key="predict_gu")
    st.session_state["sel_gu"] = gu          # RAG 페이지와 공유

    row = gdf[gdf["자치구"] == gu].iloc[0]
    grade = str(row["등급"])
    gcol = GRADE_COLOR[grade]
    q = str(row.get("Quadrant", "-"))
    qcol = Q_COLORS.get(q, "#95a5a6")

    st.markdown(
        f"**{gu}** <span class='chip' style='background:{gcol}'>{grade}</span> "
        f"<span class='chip' style='background:{qcol}'>{q}</span>",
        unsafe_allow_html=True,
    )
    st.metric("Q1 전이확률 (자치구 평균)", f"{row['Q1전이확률']:.1f}%")
    m1, m2 = st.columns(2)
    m1.metric("최고위험 행정동", f"{int(row['n_최고위험'])}/{int(row['n_행정동'])}")
    m2.metric("최고위험 비율", f"{row['최고위험비율']:.0f}%")
    st.caption(f"현재 진단(SHADOW Map): **{Q_LABEL.get(q, q)}** · "
               f"Avoidance {row['Avoidance']:.1f}")

# ── 위험동인(SHAP) ────────────────────────────────────────────────────
drivers = gu_top_drivers(gu, k=3)
if drivers:
    st.markdown(f"<div class='sec'>🔬 {gu} 위험동인 (SHAP · 행정동 평균)</div>", unsafe_allow_html=True)
    dcols = st.columns(len(drivers))
    for col, (nm, val) in zip(dcols, drivers):
        col.markdown(
            f"<div class='card' style='text-align:center'>"
            f"<div style='font-size:.78rem;color:#7b8794'>위험 기여</div>"
            f"<div style='font-size:1.05rem;font-weight:800'>{nm}</div>"
            f"<div style='color:#e74c3c;font-weight:700'>+{val:.3f}</div></div>",
            unsafe_allow_html=True,
        )
    st.caption("모델이 이 자치구의 전이확률을 끌어올린 상위 요인 (행정동 SHAP 평균).")

# ── 행정동 drill-down (근거) ──────────────────────────────────────────
detail = gu_dong_detail(gu)
if detail is not None and not detail.empty:
    with st.expander(f"🏘️ {gu} 행정동별 상세 (근거 · {len(detail)}개 동)"):
        st.dataframe(detail, use_container_width=True, hide_index=True, height=300)
        st.caption("전이확률(%) = 행정동 단위 모델 예측 · Shadow Score = 전이확률×0.75 + Avoidance×0.25")

# ── 모델 신뢰도 ───────────────────────────────────────────────────────
with st.expander("📈 모델 신뢰도 · 방법"):
    st.markdown(
        "- **모델**: Gradient Boosting (label_entry 기준 AUC ≈ 0.84)\n"
        "- **단위**: 행정동(419개) 학습·예측 → 자치구 평균으로 집계\n"
        "- **의미**: 개인 단위 예측이 아니라 **지역 단위 전이 위험 신호**\n"
        "- 상세: `Outputs/전이예측/model/model_card.md`"
    )

st.info("💊 이 자치구에 맞는 **처방**을 보려면 → 좌측 **RAG 처방** 페이지로 (선택 자치구 그대로 이어집니다).")
