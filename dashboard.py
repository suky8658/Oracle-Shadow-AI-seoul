"""
서울 5060 남성 1인가구 Q1 전이 위험 예측 대시보드
=================================================
실행:  C:/Users/vinvi/anaconda3/python.exe -m streamlit run dashboard.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

ROOT   = Path(__file__).resolve().parent
OUT    = ROOT / "Outputs" / "전이예측"
DATA   = ROOT / "Data"
PYTHON = "C:/Users/vinvi/anaconda3/python.exe"

GRADE_COLORS = {"최고위험": "#e74c3c", "고위험": "#e67e22",
                "중위험":   "#f1c40f", "저위험":  "#2ecc71"}
GRADE_ORDER  = ["최고위험", "고위험", "중위험", "저위험"]
KO_FONT      = "Malgun Gothic, Apple SD Gothic Neo, NanumGothic, sans-serif"

FEAT_LABELS = {
    "level_외출커뮤적은":"고립수준",     "level_이동횟수":"이동수준",
    "level_배달식재료":"배달의존수준",   "level_독거비율":"독거비율",
    "level_인프라":"인프라밀도",         "level_인구":"인구규모",
    "delta_외출커뮤적은":"고립 변화속도","delta_이동횟수":"이동 변화속도",
    "delta_배달식재료":"배달 변화속도",  "delta_독거비율":"독거 변화속도",
    "infra_slope":"인프라 증가추세",     "infra_delta":"인프라 변화량",
    "infra_recent":"최근 인프라량",
}

PAGES = ["🤖  AI 분석 실행", "🗺️  서울 전체 지도",
         "🔍  행정동 상세 분석", "🏛️  자치구 비교"]

# ── 세션 상태 초기화 ──────────────────────────────────────────────────
if "current_page" not in st.session_state:
    st.session_state.current_page = PAGES[0]
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# ── 페이지 설정 ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="서울 5060 전이 위험 예측",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background-color:#ffffff; }
section[data-testid="stSidebar"] { background-color:#f4f6f8; }
.risk-card {
    border-radius:14px; padding:18px 14px; text-align:center;
    border-top:5px solid; background:#fff;
    box-shadow:0 2px 8px rgba(0,0,0,.07);
}
.risk-card .num { font-size:2.4rem; font-weight:800; line-height:1; }
.risk-card .lbl { font-size:.82rem; color:#555; margin-top:5px; }
.risk-card .sub { font-size:.75rem; color:#999; margin-top:2px; }
.sec { font-size:1rem; font-weight:700; color:#2c3e50;
       border-left:4px solid #e74c3c; padding-left:10px; margin-bottom:10px; }
.insight {
    background:#f8f9fa; border-left:4px solid #3498db;
    border-radius:0 8px 8px 0; padding:12px 16px;
    margin-top:10px; font-size:.9rem; line-height:1.7; color:#2c3e50;
}
.insight-warn  { border-left-color:#e74c3c; background:#fff5f5; }
.insight-ok    { border-left-color:#2ecc71; background:#f0fff4; }
.insight-mid   { border-left-color:#e67e22; background:#fff8f0; }
.step-box {
    display:inline-block; background:#ecf0f1; border-radius:8px;
    padding:10px 18px; margin:4px 6px; font-size:.85rem;
    font-weight:600; color:#2c3e50; text-align:center;
}
</style>
""", unsafe_allow_html=True)


# ── 데이터 로딩 ───────────────────────────────────────────────────────
@st.cache_data
def load_pred():
    p = OUT / "transition_predictions.csv"
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None

@st.cache_data
def load_gu():
    p = OUT / "gu_transition_score.csv"
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None

@st.cache_data
def load_yearly():
    p = OUT / "yearly_dependency.csv"
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None

@st.cache_data
def load_shap_vals():
    p = OUT / "shap_values.csv"
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None

@st.cache_data
def load_geo():
    p = DATA / "seoul_dong.geojson"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        geo = json.load(f)
    for feat in geo["features"]:
        pr = feat["properties"]
        dong = pr.get("adm_nm","").split()[-1]
        pr["join_key"] = f"{pr.get('sggnm','')}_{dong}"
    return geo

@st.cache_data
def load_meta():
    p = OUT / "model" / "model_meta.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ── 인사이트 생성 ─────────────────────────────────────────────────────
def insight(text, cls=""):
    return f'<div class="insight {cls}">{text}</div>'

def gauge_insight(prob, grade, dong):
    if grade == "최고위험":
        return insight(
            f"🚨 <b>{dong}의 전이 확률은 {prob*100:.1f}%로 서울 내 최고위험 구간에 해당해요.</b> "
            f"고위험군 Q1 진입 가능성이 가장 높은 행정동 중 하나예요.", "insight-warn")
    elif grade == "고위험":
        return insight(
            f"⚠️ <b>{dong}의 전이 확률은 {prob*100:.1f}%로 고위험 구간이에요.</b> "
            f"Q1 진입 기준선에 근접해 있으며, 악화 추세가 지속될 경우 최고위험 구간으로 이동할 수 있어요.",
            "insight-mid")
    elif grade == "중위험":
        return insight(
            f"🔶 <b>{dong}의 전이 확률은 {prob*100:.1f}%로 중위험 구간이에요.</b> "
            f"현재 수준에서는 Q1 진입 가능성이 높지 않지만, 트렌드에 따라 변동 가능성이 있어요.")
    else:
        return insight(
            f"✅ <b>{dong}의 전이 확률은 {prob*100:.1f}%로 현재 안정적인 구간이에요.</b> "
            f"서울 전체 행정동 중 상대적으로 낮은 위험도를 유지하고 있어요.", "insight-ok")

def dep_insight(years, dong_vals, q75_val, dong):
    if len(dong_vals) < 2:
        return insight("데이터 부족으로 트렌드 분석이 어려워요.")
    first, last = dong_vals[0], dong_vals[-1]
    diff = last - first
    crossed  = first < q75_val <= last
    above_all = first >= q75_val
    if crossed:
        return insight(
            f"📈 <b>{dong}의 Dependency 점수가 기준선을 초과했어요.</b> "
            f"{first:.1f}점 → {last:.1f}점으로 <b>{diff:+.1f}점</b> 상승해 "
            f"Q1 진입 기준선({q75_val:.0f}점)을 넘어섰어요.", "insight-warn")
    elif above_all:
        return insight(
            f"🔴 <b>{dong}은 분석 기간 내내 기준선 이상을 유지하고 있어요.</b> "
            f"2022년부터 지속적으로 Q1 구간에 머물러 있으며, 현재 {last:.1f}점이에요.", "insight-warn")
    elif diff > 3:
        return insight(
            f"📊 <b>{dong}의 Dependency 점수가 빠르게 상승하고 있어요.</b> "
            f"아직 기준선({q75_val:.0f}점)을 넘지 않았지만, {diff:+.1f}점 올라 "
            f"기준선까지의 여유가 {q75_val - last:.1f}점밖에 남지 않았어요.", "insight-mid")
    elif diff < -3:
        return insight(
            f"✅ <b>{dong}의 Dependency 점수가 개선되고 있어요.</b> "
            f"{abs(diff):.1f}점 감소하며 안정세를 보이고 있고, 현재 {last:.1f}점이에요.", "insight-ok")
    else:
        return insight(
            f"➡️ <b>{dong}의 Dependency 점수는 큰 변화 없이 유지 중이에요.</b> "
            f"{first:.1f}점 → {last:.1f}점으로 변화폭이 작고, 현재 기준선({q75_val:.0f}점)과의 거리는 "
            f"{abs(q75_val - last):.1f}점이에요.")

def shap_insight(shap_data):
    if not shap_data:
        return insight("SHAP 분석 결과가 없어요.")
    sorted_items = sorted(shap_data.items(), key=lambda x: x[1], reverse=True)
    pos = [(k,v) for k,v in sorted_items if v > 0.001][:2]
    neg = [(k,v) for k,v in sorted_items[::-1] if v < -0.001][:1]
    txt = "🔍 <b>이 행정동의 전이 확률에 가장 크게 기여한 요인이에요.</b><br>"
    if pos:
        factors = ", ".join([f"<b>{k}</b> ({v:+.3f})" for k,v in pos])
        txt += f"위험 증가 요인: {factors}이(가) 전이 확률을 높이고 있어요. "
    if neg:
        factors = ", ".join([f"<b>{k}</b> ({v:+.3f})" for k,v in neg])
        txt += f"위험 감소 요인: {factors}이(가) 전이 확률을 낮추는 방향으로 작용하고 있어요."
    if not pos and not neg:
        txt += "모든 요인이 비슷한 수준으로 영향을 주고 있어요."
    return insight(txt)

def gu_bar_insight(gu_df):
    top3 = gu_df.nlargest(3, "전이위험점수")["자치구"].tolist()
    q1_gus = gu_df[gu_df.get("Quadrant","") == "Q1"]["자치구"].tolist() if "Quadrant" in gu_df.columns else []
    top1_score = float(gu_df.nlargest(1,"전이위험점수")["전이위험점수"].iloc[0])
    txt = (f"📊 <b>전이 위험점수 1위는 {top3[0]}({top1_score:.0f}점)이에요.</b> "
           f"상위 3개 자치구는 {', '.join(top3)}으로, Dependency 점수 상승 속도와 복지 회피 경향이 "
           f"복합적으로 나타나고 있어요.")
    if q1_gus:
        txt += (f" Q1 구간에 속한 {', '.join(q1_gus[:3])}은 고립도 심화와 복지 회피가 "
                f"동시에 확인된 지역이에요.")
    return insight(txt, "insight-warn")

def gu_scatter_insight(gu_df):
    if "dep_slope_gu" not in gu_df.columns:
        return insight("데이터 없음")
    q1_df = gu_df[gu_df.get("Quadrant","") == "Q1"] if "Quadrant" in gu_df.columns else pd.DataFrame()
    high_both = q1_df.nlargest(2, "전이위험점수")["자치구"].tolist() if not q1_df.empty else []
    txt = ("🗺️ <b>오른쪽 위(Q1) 구간은 Dep 악화 속도가 빠르면서 복지 회피 경향도 높은 지역이에요.</b> "
           "두 지표가 동시에 높다는 것은 고립 심화가 가속화되고 있다는 신호예요.")
    if high_both:
        avg_slope = float(q1_df["dep_slope_gu"].mean())
        txt += (f" Q1에 속한 {', '.join(high_both)} 등은 평균 slope {avg_slope:.2f}로 "
                f"서울 자치구 중 Dependency 악화 속도가 가장 빠른 지역이에요.")
    return insight(txt, "insight-warn")


# ── 사이드바 ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏙️ 서울 5060 전이 예측")
    st.caption("남성 50-60대 1인가구 고위험군 전이 예측 시스템")
    st.divider()

    # 데이터 상태
    pred_s = load_pred()
    meta_s = load_meta()
    if pred_s is not None and meta_s is not None:
        trained = meta_s.get("trained_at","N/A")[:10]
        n_total = len(pred_s)
        n_high  = int((pred_s["전이확률_GB"] >= 0.5).sum())
        st.markdown(f"📅 **마지막 분석** {trained}")
        st.markdown(f"🏘️ **행정동** {n_total}개  |  🚨 **고위험** {n_high}개")
        shap_ok = (OUT / "shap_values.csv").exists()
        st.caption(f"{'✅ SHAP 분석 완료' if shap_ok else '⬜ SHAP 미실행 (pip install shap)'}")
    else:
        st.warning("분석 결과 없음\n'AI 분석 실행' 탭에서 시작하세요.")

    st.divider()

    # 페이지 선택
    page = st.radio(
        "페이지",
        PAGES,
        index=PAGES.index(st.session_state.current_page),
        key="nav_radio",
        label_visibility="collapsed",
    )
    st.session_state.current_page = page

    # 모델 정보 (참고용, 맨 아래)
    st.divider()
    with st.expander("ℹ️ 모델 정보 (참고용)"):
        if meta_s:
            for name, m in meta_s.get("cv_metrics",{}).items():
                auc = m.get("auc",0)
                c = "#2ecc71" if auc>=0.8 else "#e67e22" if auc>=0.7 else "#e74c3c"
                st.markdown(f"**{name}**: "
                            f'<span style="color:{c}">AUC {auc:.3f}</span>  AP {m.get("ap",0):.3f}',
                            unsafe_allow_html=True)
            st.caption(f"N={meta_s.get('n_train','?')}  |  "
                       f"양성={meta_s.get('n_positive','?')} ({meta_s.get('positive_rate',0):.1%})")
            st.markdown("**Dependency 가중치**")
            st.caption("인프라 30% · 외출부재 25% · 배달 15% · 이동 15% · 독거 15%")
        else:
            st.info("분석 후 표시됩니다.")


# ════════════════════════════════════════════════════════════════════
# PAGE 0: AI 분석 실행
# ════════════════════════════════════════════════════════════════════
if "AI" in page:

    # 분석 완료 후 돌아왔을 때 메시지
    if st.session_state.analysis_done:
        st.session_state.analysis_done = False
        st.success("✅ AI 분석이 완료됐어요! 왼쪽 메뉴에서 **서울 전체 지도** 탭을 눌러 결과를 확인해 보세요.")
        st.balloons()

    st.markdown("## 🤖 AI 분석 실행")
    st.caption("통신 데이터를 기반으로 서울 419개 행정동의 고위험군 전이 확률을 분석합니다")
    st.divider()

    # 파이프라인 단계 표시
    st.markdown("**분석 파이프라인 (4단계)**")
    st.markdown("""
    <div style="margin:12px 0 20px">
      <span class="step-box">📡 1. 데이터 로딩</span>
      <span style="color:#bdc3c7;font-size:1.2rem">→</span>
      <span class="step-box">🔢 2. Dep 점수 산출</span>
      <span style="color:#bdc3c7;font-size:1.2rem">→</span>
      <span class="step-box">🤖 3. ML 모델 학습</span>
      <span style="color:#bdc3c7;font-size:1.2rem">→</span>
      <span class="step-box">💾 4. 결과 저장</span>
    </div>
    """, unsafe_allow_html=True)

    # 현재 상태 카드
    pred_exists = load_pred() is not None
    col_s, _ = st.columns([1, 2])
    with col_s:
        if pred_exists and meta_s:
            st.markdown(f"""
            <div style="background:#f0fff4;border:1px solid #2ecc71;border-radius:10px;
                        padding:14px 18px;margin-bottom:16px">
              <b style="color:#27ae60">✅ 이전 분석 결과 있음</b><br>
              <span style="color:#555;font-size:.88rem">마지막 실행: {meta_s.get('trained_at','N/A')}</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#fff5f5;border:1px solid #e74c3c;border-radius:10px;
                        padding:14px 18px;margin-bottom:16px">
              <b style="color:#c0392b">❌ 분석 결과 없음</b><br>
              <span style="color:#555;font-size:.88rem">아래 버튼을 눌러 첫 분석을 시작하세요</span>
            </div>""", unsafe_allow_html=True)

    # 분석 시작 버튼
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        start = st.button("▶  분석 시작", type="primary", use_container_width=True)

    if start:
        script = ROOT / "code" / "run_all.py"
        status_area = st.empty()

        with status_area:
            with st.spinner("⏳ 분석 중... (약 30~60초 소요)"):
                result = subprocess.run(
                    [PYTHON, str(script)],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=str(ROOT),
                )

        if result.returncode == 0:
            st.session_state.analysis_done = True
            st.session_state.current_page  = "🗺️  서울 전체 지도"
            st.cache_data.clear()
            st.rerun()
        else:
            st.error("❌ 분석 중 오류가 발생했어요.")
            with st.expander("오류 내용 보기"):
                st.code(result.stderr or result.stdout or "알 수 없는 오류", language=None)

    # 안내
    st.divider()
    st.markdown("""
    **💡 언제 다시 실행하나요?**
    - 새로운 통신 데이터가 추가됐을 때
    - 분기별 상권 데이터가 갱신됐을 때
    - 모델 파라미터를 변경했을 때
    """)


# ════════════════════════════════════════════════════════════════════
# PAGE 1: 서울 전체 지도
# ════════════════════════════════════════════════════════════════════
elif "지도" in page:
    pred = load_pred()
    if pred is None:
        st.error("분석 결과 없음 — 'AI 분석 실행' 탭에서 분석을 시작해 주세요.")
        st.stop()

    if st.session_state.analysis_done:
        st.session_state.analysis_done = False
        st.success("✅ 분석 완료! 아래에서 결과를 확인하세요.")

    st.markdown("## 🗺️ 서울 전체 Q1 전이 위험 지도")
    st.caption("서울 419개 행정동의 5060 남성 1인가구 고위험군 전이 확률 (2022→2025 기반)")

    # 위험등급 카드
    counts = pred["위험등급"].value_counts()
    c1,c2,c3,c4 = st.columns(4)
    for col, (grade,color,sub) in zip([c1,c2,c3,c4],[
        ("최고위험","#e74c3c","전이확률 70% 이상"),
        ("고위험",  "#e67e22","전이확률 50–70%"),
        ("중위험",  "#f1c40f","전이확률 30–50%"),
        ("저위험",  "#2ecc71","전이확률 30% 미만"),
    ]):
        cnt = int(counts.get(grade,0))
        col.markdown(f"""
        <div class="risk-card" style="border-top-color:{color}">
          <div class="num" style="color:{color}">{cnt}</div>
          <div class="lbl">{grade}</div>
          <div class="sub">{cnt/len(pred)*100:.0f}%  ·  {sub}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("")

    col_map, col_rank = st.columns([3,1])

    with col_map:
        geo = load_geo()
        pred["join_key"] = pred["자치구"] + "_" + pred["행정동"]
        if geo is not None:
            fig = px.choropleth_mapbox(
                pred, geojson=geo, locations="join_key",
                featureidkey="properties.join_key",
                color="전이확률_GB",
                color_continuous_scale=[
                    [0.0,"#d5f5e3"],[0.3,"#f9e79f"],
                    [0.5,"#f0a500"],[0.7,"#e74c3c"],[1.0,"#7b241c"],
                ],
                range_color=[0,1],
                hover_name="행정동",
                hover_data={"자치구":True,"전이확률_GB":":.3f","위험등급":True,"join_key":False},
                labels={"전이확률_GB":"전이확률"},
                mapbox_style="carto-positron",
                center={"lat":37.5665,"lon":126.978}, zoom=10,
                height=560, opacity=0.78,
            )
            fig.update_layout(
                margin={"r":0,"t":0,"l":0,"b":0},
                paper_bgcolor="white",
                coloraxis_colorbar=dict(
                    title="전이확률",
                    tickvals=[0,0.3,0.5,0.7,1.0],
                    ticktext=["0%","30%","50%","70%","100%"],
                    len=0.55, thickness=14,
                ),
                font=dict(family=KO_FONT),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            top30 = pred.nlargest(30,"전이확률_GB").copy()
            top30["label"] = top30["자치구"]+" "+top30["행정동"]
            fig_fb = px.bar(top30.iloc[::-1], x="전이확률_GB", y="label",
                            color="위험등급", color_discrete_map=GRADE_COLORS,
                            orientation="h", height=600,
                            labels={"전이확률_GB":"전이확률","label":""})
            fig_fb.update_layout(paper_bgcolor="white", font=dict(family=KO_FONT))
            st.plotly_chart(fig_fb, use_container_width=True)

    with col_rank:
        st.markdown('<div class="sec">🚨 고위험 상위 25</div>', unsafe_allow_html=True)
        top25 = (pred.nlargest(25,"전이확률_GB")
                 [["자치구","행정동","전이확률_GB","위험등급"]].copy())
        top25.columns = ["자치구","행정동","확률","등급"]
        top25["확률"] = top25["확률"].apply(lambda x: f"{x:.3f}")
        top25 = top25.reset_index(drop=True)
        top25.index += 1
        st.dataframe(top25, use_container_width=True, height=535)


# ════════════════════════════════════════════════════════════════════
# PAGE 2: 행정동 상세 분석  (세로 배치)
# ════════════════════════════════════════════════════════════════════
elif "행정동" in page:
    pred   = load_pred()
    yearly = load_yearly()
    shap_v = load_shap_vals()
    meta   = load_meta()

    if pred is None:
        st.error("분석 결과 없음 — 'AI 분석 실행' 탭에서 분석을 시작해 주세요.")
        st.stop()

    st.markdown("## 🔍 행정동 상세 분석")

    cs1, cs2 = st.columns([1,2])
    with cs1:
        gu_list  = sorted(pred["자치구"].dropna().unique())
        sel_gu   = st.selectbox("자치구", gu_list)
    with cs2:
        dong_df  = pred[pred["자치구"]==sel_gu].sort_values("전이확률_GB", ascending=False)
        sel_dong = st.selectbox("행정동  (전이확률 높은 순)", dong_df["행정동"].tolist())

    row = pred[(pred["자치구"]==sel_gu)&(pred["행정동"]==sel_dong)]
    if row.empty:
        st.warning("해당 행정동 데이터 없음")
        st.stop()
    row   = row.iloc[0]
    prob  = float(row["전이확률_GB"])
    grade = str(row["위험등급"])
    gcol  = GRADE_COLORS.get(grade,"#95a5a6")

    st.markdown(
        f"### {sel_gu} **{sel_dong}**  "
        f'<span style="background:{gcol};color:white;font-size:.9rem;'
        f'padding:5px 14px;border-radius:20px;font-weight:700">{grade}</span>',
        unsafe_allow_html=True)
    st.divider()

    # ━━━ ① 전이 확률 게이지 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown('<div class="sec">① Q1 전이 확률</div>', unsafe_allow_html=True)
    col_g, col_rank2 = st.columns([1.4, 1])

    with col_g:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            number={"suffix":"%","font":{"size":52,"family":KO_FONT},"valueformat":".1f"},
            gauge={
                "axis":{"range":[0,100],"tickwidth":1,"tickfont":{"size":11}},
                "bar":{"color":gcol,"thickness":0.28},
                "bgcolor":"white","borderwidth":0,
                "steps":[
                    {"range":[0, 30],"color":"#e8f8f0"},
                    {"range":[30,50],"color":"#fef9e7"},
                    {"range":[50,70],"color":"#fdebd0"},
                    {"range":[70,100],"color":"#fadbd8"},
                ],
                "threshold":{"line":{"color":gcol,"width":5},"thickness":0.75,"value":prob*100},
            },
        ))
        fig_g.update_layout(
            height=260, margin={"t":20,"b":0,"l":20,"r":20},
            paper_bgcolor="white", font=dict(family=KO_FONT),
        )
        st.plotly_chart(fig_g, use_container_width=True)

    with col_rank2:
        all_p = load_pred()["전이확률_GB"].dropna()
        rank  = int((all_p > prob).sum()) + 1
        total = len(all_p)
        st.markdown(f"""
        <div style="padding:20px 10px">
          <div style="font-size:2.5rem;font-weight:800;color:{gcol}">{rank}위</div>
          <div style="color:#666;font-size:.9rem">서울 {total}개 행정동 중</div>
          <div style="color:#888;font-size:.85rem;margin-top:4px">상위 {rank/total*100:.1f}%</div>
          <hr style="border-color:#eee;margin:12px 0">
          <div style="font-size:.85rem;color:#555">
            서울 평균: <b>{all_p.mean():.3f}</b><br>
            이 동: <b style="color:{gcol}">{prob:.3f}</b>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(gauge_insight(prob, grade, sel_dong), unsafe_allow_html=True)
    st.markdown("")

    # ━━━ ② 연도별 Dependency 궤적 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown('<div class="sec">② 연도별 Dependency 궤적</div>', unsafe_allow_html=True)
    dep_yr = ["dep_2022","dep_2023","dep_2024","dep_2025"]
    avail  = [c for c in dep_yr if c in row.index and pd.notna(row[c])]

    if avail and yearly is not None:
        years      = [int(c.split("_")[1]) for c in avail]
        dong_vals  = [float(row[c]) for c in avail]
        yr_avail   = [c for c in dep_yr if c in yearly.columns]
        all_dep    = yearly[yr_avail].values.flatten()
        q75_val    = float(np.nanpercentile(all_dep, 75))
        avg_vals   = [float(yearly[c].mean()) if c in yearly.columns else np.nan for c in avail]

        fig_dep = go.Figure()
        fig_dep.add_trace(go.Scatter(
            x=years, y=avg_vals, name="서울 평균",
            mode="lines+markers",
            line=dict(color="#bdc3c7",width=2,dash="dot"),
            marker=dict(size=6),
        ))
        fig_dep.add_trace(go.Scatter(
            x=years, y=dong_vals, name=sel_dong,
            mode="lines+markers+text",
            line=dict(color=gcol,width=3),
            marker=dict(size=10),
            text=[f"{v:.1f}" for v in dong_vals],
            textposition="top center",
            textfont=dict(size=12,color=gcol),
        ))
        fig_dep.add_hline(
            y=q75_val,
            line_color="#e74c3c", line_dash="dash", line_width=1.5,
            annotation_text=f"Q1 진입 기준선  {q75_val:.0f}점",
            annotation_position="top right",
            annotation_font=dict(color="#e74c3c",size=11,family=KO_FONT),
        )
        fig_dep.add_hrect(y0=q75_val,y1=100,fillcolor="#e74c3c",opacity=0.04,line_width=0)
        fig_dep.update_layout(
            height=320, paper_bgcolor="white", plot_bgcolor="white",
            margin={"t":20,"b":10,"l":0,"r":0},
            yaxis=dict(range=[0,108],title="Dependency 점수",gridcolor="#f0f0f0"),
            xaxis=dict(tickvals=years,gridcolor="#f0f0f0"),
            legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,
                        font=dict(family=KO_FONT,size=10)),
            font=dict(family=KO_FONT),
        )
        st.plotly_chart(fig_dep, use_container_width=True)
        st.markdown(dep_insight(years, dong_vals, q75_val, sel_dong), unsafe_allow_html=True)
    else:
        st.info("연도별 Dep 데이터 없음")
    st.markdown("")

    # ━━━ ③ SHAP 원인 분해 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown('<div class="sec">③ 전이 확률 원인 분해 (SHAP)</div>',
                unsafe_allow_html=True)

    shap_data = None
    if shap_v is not None:
        sr = shap_v[(shap_v.get("자치구","") == sel_gu) &
                    (shap_v.get("행정동","") == sel_dong)]
        if not sr.empty:
            meta_c = [c for c in ["행정동코드","자치구","행정동","전이확률"] if c in shap_v.columns]
            feat_c = [c for c in shap_v.columns if c not in meta_c]
            sr     = sr.iloc[0]
            shap_data = {FEAT_LABELS.get(c,c): float(sr[c]) for c in feat_c if c in sr.index}

    if shap_data:
        sorted_shap = dict(sorted(shap_data.items(), key=lambda x: x[1], reverse=True))
        names  = list(sorted_shap.keys())
        values = list(sorted_shap.values())
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in values]

        fig_shap = go.Figure(go.Bar(
            x=values[::-1], y=names[::-1],
            orientation="h",
            marker_color=colors[::-1],
            text=[f"{v:+.4f}" for v in values[::-1]],
            textposition="outside",
        ))
        fig_shap.add_vline(x=0, line_color="#aaa", line_width=1.5)
        x_max = max(abs(v) for v in values) if values else 0.1
        fig_shap.update_layout(
            height=max(280, len(names)*28),
            margin={"t":10,"b":30,"l":0,"r":80},
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(range=[-x_max*1.5,x_max*1.5],
                       title="← 위험 감소  |  위험 증가 →",
                       title_font=dict(size=10,family=KO_FONT)),
            font=dict(family=KO_FONT,size=11),
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        st.caption("🔴 빨강: 전이 위험을 높이는 요인  /  🔵 파랑: 위험을 낮추는 요인")
        st.markdown(shap_insight(shap_data), unsafe_allow_html=True)
    else:
        st.info("SHAP 분석 결과가 없어요.  \n"
                "`pip install shap` 후 'AI 분석 실행' 탭에서 분석을 다시 실행해 주세요.")
    st.markdown("")

    # 피처 상세 값
    with st.expander("📋 피처 상세 값 보기"):
        if meta:
            rows_feat = [{"피처":FEAT_LABELS.get(c,c),"코드":c,"값":round(float(row[c]),4)}
                         for c in meta.get("feat_cols",[])
                         if c in row.index and pd.notna(row[c])]
            if rows_feat:
                st.dataframe(pd.DataFrame(rows_feat), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════
# PAGE 3: 자치구 비교  (세로 배치)
# ════════════════════════════════════════════════════════════════════
elif "자치구" in page:
    gu_df = load_gu()
    if gu_df is None:
        st.error("분석 결과 없음 — 'AI 분석 실행' 탭에서 분석을 시작해 주세요.")
        st.stop()

    st.markdown("## 🏛️ 자치구별 Q1 전이 위험 비교")
    Q_COL = {"Q1":"#e74c3c","Q2":"#e67e22","Q3":"#3498db","Q4":"#9b59b6"}

    # ━━━ ① 전이 위험점수 순위 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown('<div class="sec">① 자치구 전이 위험점수 순위</div>', unsafe_allow_html=True)
    gu_s   = gu_df.sort_values("전이위험점수", ascending=True)
    bcolors = [Q_COL.get(q,"#95a5a6") for q in gu_s.get("Quadrant",[])]

    fig_bar = go.Figure(go.Bar(
        x=gu_s["전이위험점수"], y=gu_s["자치구"],
        orientation="h", marker_color=bcolors,
        text=gu_s["전이위험점수"].apply(lambda x: f"  {x:.0f}"),
        textposition="outside",
        hovertemplate="<b>%{y}</b>  위험점수 %{x:.1f}<extra></extra>",
    ))
    for q,c in Q_COL.items():
        lbl = {"Q1":"Q1 고의존·회피↑","Q2":"Q2 고의존·회피↓",
               "Q3":"Q3 저의존·회피↓","Q4":"Q4 저의존·회피↑"}.get(q,q)
        fig_bar.add_trace(go.Bar(x=[None],y=[None],marker_color=c,name=lbl,showlegend=True))
    fig_bar.update_layout(
        height=620, paper_bgcolor="white", plot_bgcolor="white",
        margin={"t":20,"b":10,"l":0,"r":60},
        xaxis=dict(range=[0,120],gridcolor="#f0f0f0"),
        legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="right",x=1,
                    font=dict(family=KO_FONT,size=10)),
        font=dict(family=KO_FONT,size=11), barmode="overlay",
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown(gu_bar_insight(gu_df), unsafe_allow_html=True)
    st.markdown("")

    # ━━━ ② Dep slope × Avoidance 산점도 ━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown('<div class="sec">② Dep 악화 추세 × 복지 회피도</div>', unsafe_allow_html=True)

    if "dep_slope_gu" in gu_df.columns and "Avoidance" in gu_df.columns:
        gs = gu_df.dropna(subset=["dep_slope_gu","Avoidance"]).copy()
        xm, ym = float(gs["dep_slope_gu"].median()), float(gs["Avoidance"].median())

        fig_sc = go.Figure()
        for _, r in gs.iterrows():
            c = Q_COL.get(r.get("Quadrant","Q3"),"#95a5a6")
            fig_sc.add_trace(go.Scatter(
                x=[r["dep_slope_gu"]], y=[r["Avoidance"]],
                mode="markers+text",
                marker=dict(size=r["전이위험점수"]/2.5+10,color=c,opacity=0.85,
                            line=dict(color="white",width=1.5)),
                text=[r["자치구"]], textposition="top center",
                textfont=dict(size=10,family=KO_FONT),
                hovertemplate=(f"<b>{r['자치구']}</b><br>"
                               f"Dep slope: {r['dep_slope_gu']:.3f}<br>"
                               f"Avoidance: {r['Avoidance']:.1f}<br>"
                               f"위험점수: {r['전이위험점수']:.1f}<extra></extra>"),
                showlegend=False,
            ))
        fig_sc.add_vline(x=xm,line_color="#ddd",line_dash="dot")
        fig_sc.add_hline(y=ym,line_color="#ddd",line_dash="dot")
        xr,yr = float(gs["dep_slope_gu"].max()),float(gs["Avoidance"].max())
        xl,yl = float(gs["dep_slope_gu"].min()),float(gs["Avoidance"].min())
        for tx,ty,txt,c in [
            (xr,yr,"⚠️ Q1  최우선 개입","#e74c3c"),
            (xl,yr,"Q4  관찰 필요","#9b59b6"),
            (xl,yl,"Q3  상대적 안전","#3498db"),
            (xr,yl,"Q2  인프라 주의","#e67e22"),
        ]:
            fig_sc.add_annotation(
                x=tx,y=ty,text=txt,showarrow=False,
                font=dict(color=c,size=10,family=KO_FONT),
                xanchor="right" if tx==xr else "left",
                yanchor="top" if ty==yr else "bottom",
            )
        fig_sc.update_layout(
            height=520, paper_bgcolor="white", plot_bgcolor="#fafafa",
            margin={"t":20,"b":30,"l":30,"r":10},
            xaxis=dict(title="Dep 연간 악화 속도 (slope →)",
                       title_font=dict(size=11,family=KO_FONT),gridcolor="#f0f0f0"),
            yaxis=dict(title="복지 회피도 (Avoidance →)",
                       title_font=dict(size=11,family=KO_FONT),gridcolor="#f0f0f0"),
            font=dict(family=KO_FONT),
        )
        st.plotly_chart(fig_sc, use_container_width=True)
        st.caption("버블 크기 = 전이 위험점수  ·  기준선 = 서울 중앙값")
        st.markdown(gu_scatter_insight(gu_df), unsafe_allow_html=True)
    else:
        st.info("자치구 비교 데이터 없음")

    with st.expander("📋 자치구 전체 데이터"):
        show_c = [c for c in ["자치구","전이위험점수","Quadrant","Avoidance",
                               "dep_slope_gu","dep_2022","dep_2025"] if c in gu_df.columns]
        st.dataframe(gu_df[show_c].sort_values("전이위험점수",ascending=False)
                     .reset_index(drop=True).round(2),
                     use_container_width=True, hide_index=True)
