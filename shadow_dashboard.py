"""
Shadow AI — Prescription Dashboard
====================================
실행: C:/Users/vinvi/anaconda3/python.exe -m streamlit run shadow_dashboard.py
"""
import json
import subprocess
import sys
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

ROOT   = Path(__file__).resolve().parent
OUT    = ROOT / "Outputs"
SHADOW = OUT / "shadow_ai"
DATA   = ROOT / "Data"
PYTHON = sys.executable

GRADE_COLOR = {
    "최고위험": "#e74c3c", "고위험": "#e67e22",
    "중위험":   "#f1c40f", "저위험":  "#2ecc71",
}
GRADE_RANGE = {
    "최고위험": "80점 이상", "고위험": "65–80점",
    "중위험":   "50–65점",  "저위험":  "50점 미만",
}
GRADE_ORDER = ["최고위험", "고위험", "중위험", "저위험"]
KO_FONT     = "Malgun Gothic, Apple SD Gothic Neo, NanumGothic, sans-serif"

# Avoidance 구성요소 (가중치·라벨·방향성)
AVO_COMP = {
    "A_도움부재":    ("주변 도움 부재율",    0.30),
    "B_외로움부정":  ("외로움 부정 경향",    0.30),
    "C_복지불신":    ("복지 서비스 불신",    0.25),
    "D_네트워크축소":("사회 네트워크 축소율", 0.15),
}

FEAT_LABELS = {
    "level_외출커뮤적은": "고립수준",      "level_이동횟수":   "이동수준",
    "level_배달식재료":  "배달의존수준",   "level_독거비율":   "독거비율",
    "level_인프라":      "인프라밀도",     "level_인구":       "인구규모",
    "delta_외출커뮤적은":"고립 변화속도",  "delta_이동횟수":   "이동 변화속도",
    "delta_배달식재료":  "배달 변화속도",  "delta_독거비율":   "독거 변화속도",
    "infra_slope":       "인프라 증가추세","infra_delta":      "인프라 변화량",
    "infra_recent":      "최근 인프라량",
}

PAGES = ["🤖  분석 실행", "🗺️  전이 위험 지도", "🔍  행정동 상세"]

if "current_page" not in st.session_state or st.session_state.current_page not in PAGES:
    st.session_state.current_page = PAGES[0]
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

st.set_page_config(
    page_title="Shadow AI — Q1 Transition Risk Dashboard",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background-color:#ffffff; }
section[data-testid="stSidebar"] { background-color:#f4f6f8; }
.grade-card {
    border-radius:14px; padding:18px 14px; text-align:center;
    border-top:5px solid; background:#fff;
    box-shadow:0 2px 8px rgba(0,0,0,.07);
}
.grade-card .num { font-size:2.4rem; font-weight:800; line-height:1; }
.grade-card .lbl { font-size:.82rem; color:#555; margin-top:5px; }
.grade-card .sub { font-size:.75rem; color:#999; margin-top:2px; }
.sec {
    font-size:1.25rem; font-weight:700; color:#2c3e50;
    border-left:4px solid #e74c3c; padding-left:10px; margin-bottom:10px;
}
.insight {
    background:#f8f9fa; border-left:4px solid #3498db;
    border-radius:0 8px 8px 0; padding:12px 16px;
    margin-top:10px; font-size:.9rem; line-height:1.7; color:#2c3e50;
}
.insight-warn { border-left-color:#e74c3c; background:#fff5f5; }
.insight-ok   { border-left-color:#2ecc71; background:#f0fff4; }
.insight-mid  { border-left-color:#e67e22; background:#fff8f0; }
.step-box {
    display:inline-block; background:#ecf0f1; border-radius:8px;
    padding:10px 18px; margin:4px 6px; font-size:.85rem;
    font-weight:600; color:#2c3e50; text-align:center;
}
</style>
""", unsafe_allow_html=True)


# ── 데이터 로딩 ───────────────────────────────────────────────────────
@st.cache_data
def load_shadow(_mtime=None):
    p = SHADOW / "shadow_prescriptions.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, encoding="utf-8-sig")
    # 구버전 컬럼명 호환
    if "처방등급" in df.columns and "위험등급" not in df.columns:
        df = df.rename(columns={"처방등급": "위험등급"})
    return df

@st.cache_data
def load_risk():
    p = OUT / "전이예측" / "risk_predictions_final.csv"
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None

@st.cache_data
def load_shap_vals():
    p = OUT / "전이예측" / "shap_values.csv"
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None

@st.cache_data
def load_avoidance():
    p = OUT / "복지의 역설" / "avoidance_index.csv"
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None

@st.cache_data
def load_geo():
    p = DATA / "seoul_dong.geojson"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        geo = json.load(f)
    for feat in geo["features"]:
        pr   = feat["properties"]
        dong = pr.get("adm_nm", "").split()[-1]
        pr["join_key"] = f"{pr.get('sggnm', '')}_{dong}"
    return geo


# ── 인사이트 함수 ─────────────────────────────────────────────────────
def insight(text, cls=""):
    return f'<div class="insight {cls}">{text}</div>'

def shap_insight(shap_data):
    if not shap_data:
        return insight("SHAP 분석 결과가 없어요.")
    items = sorted(shap_data.items(), key=lambda x: x[1], reverse=True)
    pos   = [(k, v) for k, v in items if v > 0.001][:2]
    neg   = [(k, v) for k, v in items[::-1] if v < -0.001][:1]
    txt   = "🔍 <b>전이 위험도에 가장 크게 기여한 요인이에요.</b><br>"
    if pos:
        factors = ", ".join([f"<b>{k}</b> ({v:+.3f})" for k, v in pos])
        txt += f"위험 증가 요인: {factors}이(가) 전이 위험을 높이고 있어요. "
    if neg:
        factors = ", ".join([f"<b>{k}</b> ({v:+.3f})" for k, v in neg])
        txt += f"위험 완화 요인: {factors}이(가) 위험을 낮추는 방향으로 작용하고 있어요."
    if not pos and not neg:
        txt += "모든 요인이 비슷한 수준으로 영향을 주고 있어요."
    return insight(txt)

def avoidance_insight(sel_gu, avo_v, comp_data):
    # 가중 편차 기여도 기준으로 주요 드라이버 식별
    drivers = sorted(comp_data.items(), key=lambda x: x[1]["diff"], reverse=True)
    top_label, top_d   = drivers[0]   # Avoidance를 가장 많이 올리는 요인
    bot_label, bot_d   = drivers[-1]  # 가장 많이 낮추는 요인
    total_w_diff = sum(d["diff"] for _, d in comp_data.items())

    if avo_v >= 70:
        base = (f"🚨 <b>{sel_gu}의 복지 회피 인덱스는 {avo_v:.1f}점으로 서울 내 매우 높은 수준이에요.</b> "
                f"복지 서비스를 스스로 기피하거나 도움 요청을 꺼리는 경향이 두드러지는 지역이에요.")
        cls = "insight-warn"
    elif avo_v >= 50:
        base = (f"⚠️ <b>{sel_gu}의 복지 회피 인덱스는 {avo_v:.1f}점으로 서울 평균보다 높아요.</b> "
                f"복지 접근에 심리적 장벽이 있는 인구가 상대적으로 많아요.")
        cls = "insight-mid"
    else:
        base = (f"✅ <b>{sel_gu}의 복지 회피 인덱스는 {avo_v:.1f}점으로 서울 평균 이하예요.</b> "
                f"복지 수용 의향이 상대적으로 높아 공식 채널 연계가 수월한 지역이에요.")
        cls = "insight-ok"

    # 가중 편차 기여도 기준 (최대 가중치 0.30 × 100 = 30점)
    if top_d["diff"] > 5:
        detail = (f" 가중 편차 기여도 분석 결과 '<b>{top_label}</b>'이 "
                  f"이 구의 Avoidance를 서울 평균보다 {top_d['diff']:+.2f}만큼 끌어올리는 "
                  f"주된 요인으로 확인돼요.")
    elif top_d["diff"] > 1:
        detail = (f" '<b>{top_label}</b>'이 소폭 상회하며 Avoidance를 높이는 방향으로 작용하고 있어요.")
    else:
        detail = " 4개 세부 항목의 가중 편차가 모두 작아 특정 요인에 의한 집중 현상은 없어요."

    if bot_d["diff"] < -3:
        detail += (f" 반면 '<b>{bot_label}</b>'은 "
                   f"{bot_d['diff']:+.2f}으로 Avoidance를 낮추는 방향으로 작용하고 있어요.")

    return insight(base + detail, cls)

def w_contrib_insight(sel_gu, avo_v, comp_data):
    sorted_comps = sorted(comp_data.items(), key=lambda x: x[1]["w_contrib"], reverse=True)
    top_label, top_d = sorted_comps[0]
    total_contrib    = sum(d["w_contrib"] for _, d in comp_data.items())
    top_pct          = top_d["w_contrib"] / total_contrib * 100 if total_contrib > 0 else 0
    top2_pct         = sum(d["w_contrib"] for _, d in sorted_comps[:2]) / total_contrib * 100 if total_contrib > 0 else 0
    avg_top_label    = max(comp_data.items(), key=lambda x: x[1]["w_contrib_avg"])[0]

    txt = (
        f"📊 <b>{sel_gu}의 Avoidance {avo_v:.1f}점 구성에서 "
        f"'{top_label}'이 {top_d['w_contrib']:.1f}점({top_pct:.0f}%)으로 가장 큰 비중을 차지해요.</b> "
    )
    if top_label != avg_top_label:
        txt += (
            f"서울 평균 1위 요인은 '{avg_top_label}'인 반면 "
            f"{sel_gu}는 '{top_label}'이 주도하는 특이 구조예요. "
        )
    else:
        txt += f"서울 평균과 동일한 요인이 1위로, 구조적으로 유사한 패턴이에요. "

    if top_pct >= 45:
        txt += f"단일 요인 집중 구조예요."
    elif top2_pct >= 65:
        txt += f"상위 2개 요인이 전체의 {top2_pct:.0f}%를 차지하는 이원 집중 구조예요."

    return insight(txt, "insight-mid")

def dep_trend_insight(sel_dong, d_vals, q75_val, slope):
    if d_vals[-1] >= q75_val:
        msg = (f"🔴 <b>{sel_dong}은 고위험군 진입 기준선({q75_val:.0f}점) 이상을 유지하고 있어요.</b> "
               f"현재 {d_vals[-1]:.1f}점으로, 연간 {abs(slope):.2f}점"
               f"{'씩 상승 중이에요.' if slope > 0 else '이지만 하락 추세예요.'}")
        return insight(msg, "insight-warn")
    elif slope > 1:
        msg = (f"📈 <b>{sel_dong}의 편의 의존도가 연간 {slope:.2f}점씩 상승하고 있어요.</b> "
               f"현재 {d_vals[-1]:.1f}점으로, 기준선({q75_val:.0f}점)까지 "
               f"{q75_val - d_vals[-1]:.1f}점 남았어요.")
        return insight(msg, "insight-mid")
    elif slope < -0.5:
        msg = (f"✅ <b>{sel_dong}의 편의 의존도가 개선되고 있어요.</b> "
               f"연간 {abs(slope):.2f}점 감소하며 안정세를 보이고 있어요.")
        return insight(msg, "insight-ok")
    return insight(f"➡️ <b>{sel_dong}의 편의 의존도는 큰 변화 없이 유지 중이에요.</b> "
                   f"현재 {d_vals[-1]:.1f}점이에요.")

def gu_comp_insight(sel_dong, sel_gu, score, gu_comp_df):
    total     = len(gu_comp_df)
    rank_list = gu_comp_df.sort_values("Shadow_Score", ascending=False)["행정동"].tolist()
    rank      = rank_list.index(sel_dong) + 1
    avg       = float(gu_comp_df["Shadow_Score"].mean())
    top_dong  = rank_list[0]
    top_sc    = float(gu_comp_df.nlargest(1, "Shadow_Score")["Shadow_Score"].iloc[0])
    if rank == 1:
        return insight(
            f"🚨 <b>{sel_dong}은 {sel_gu} 내에서 Shadow Score가 가장 높아요.</b> "
            f"{total}개 행정동 중 1위({score:.1f}점)로, 이 구에서 가장 시급한 관심이 필요한 지역이에요.",
            "insight-warn")
    elif score >= avg + 5:
        return insight(
            f"⚠️ <b>{sel_dong}의 Shadow Score는 {sel_gu} 평균({avg:.1f}점)보다 "
            f"{score - avg:.1f}점 높아요.</b> "
            f"{sel_gu} 내 {rank}위/{total}위이며, 1위 {top_dong}({top_sc:.1f}점)과의 격차는 "
            f"{top_sc - score:.1f}점이에요.", "insight-mid")
    elif score >= avg:
        return insight(
            f"📊 <b>{sel_dong}의 Shadow Score는 {sel_gu} 평균({avg:.1f}점)과 비슷한 수준이에요.</b> "
            f"{sel_gu} 내 {rank}위/{total}위이며, 추세 변화에 따라 등급이 변동될 수 있어요.")
    return insight(
        f"✅ <b>{sel_dong}의 Shadow Score는 {sel_gu} 평균({avg:.1f}점)보다 낮아요.</b> "
        f"{sel_gu} 내 {rank}위/{total}위로, 현재 상대적으로 안정적인 수준이에요.",
        "insight-ok")


def _mini_gauge(value, title, subtitle, color, steps):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": f"{title}<br><span style='font-size:0.75em;color:#888'>{subtitle}</span>",
               "font": {"size": 13, "family": KO_FONT}},
        number={"font": {"size": 30, "family": KO_FONT}, "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"size": 8}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "white", "borderwidth": 0,
            "steps": steps,
            "threshold": {"line": {"color": color, "width": 4},
                          "thickness": 0.75, "value": value},
        },
    ))
    fig.update_layout(
        height=200, margin={"t": 40, "b": 0, "l": 10, "r": 10},
        paper_bgcolor="white", font=dict(family=KO_FONT),
    )
    return fig


# ── 사이드바 ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🕵️ Shadow AI")
    st.caption("서울시 5060 남성 1인가구 Q1 고위험군 전이 위험도 예측 시스템")
    st.divider()

    _p = SHADOW / "shadow_prescriptions.csv"
    df_s = load_shadow(_mtime=_p.stat().st_mtime if _p.exists() else None)
    if df_s is not None:
        n_top = int((df_s["위험등급"] == "최고위험").sum())
        n_hi  = int((df_s["위험등급"] == "고위험").sum())
        ts    = SHADOW / "shadow_prescriptions.csv"
        mtime = datetime.datetime.fromtimestamp(ts.stat().st_mtime).strftime("%Y-%m-%d")
        st.markdown(f"📅 **마지막 분석** {mtime}")
        st.markdown(f"🏘️ **행정동** {len(df_s)}개  |  🚨 **최고위험 전이** {n_top}개")
        st.caption(f"⚠️ 고위험 {n_hi}개  ·  서울 평균 Shadow Score {df_s['Shadow_Score'].mean():.1f}점")
    else:
        st.warning("분석 결과 없음\n'분석 실행' 탭에서 시작하세요.")

    st.divider()

    page = st.radio(
        "페이지", PAGES,
        index=PAGES.index(st.session_state.current_page),
        key="nav_radio", label_visibility="collapsed",
    )
    st.session_state.current_page = page

    st.divider()

    with st.expander("📊 Shadow Score 상세"):
        st.markdown("""
**계산 구조**
```
Shadow Score
  = Dependency × 75%
  + Avoidance  × 25%
```
- **Dependency** — 편의 의존도 기반 위험도
  행정동 단위, 0~100 정규화
- **Avoidance** — 복지 회피 인덱스
  자치구 단위, 0~100 정규화

---
**위험등급 기준**

| 등급 | Shadow Score |
|---|---|
| 🔴 **최고위험** | 80점 이상 |
| 🟠 **고위험** | 65–80점 |
| 🟡 **중위험** | 50–65점 |
| 🟢 **저위험** | 50점 미만 |
""")
        if df_s is not None:
            st.markdown("---\n**현황 통계**")
            st.markdown(f"- 서울 평균 Shadow Score: **{df_s['Shadow_Score'].mean():.1f}점**")
            st.markdown(f"- 평균 편의 의존도: **{df_s['전이확률_정규화'].mean():.1f}점**")
            st.markdown(f"- 평균 복지 회피도: **{df_s['Avoidance'].mean():.1f}점**")
            top1 = df_s.nlargest(1, "Shadow_Score").iloc[0]
            st.markdown(
                f"- 최고위험 1위: **{top1['자치구']} {top1['행정동']}** ({top1['Shadow_Score']:.1f}점)"
            )


# ════════════════════════════════════════════════════════════════════
# PAGE 0: 분석 실행
# ════════════════════════════════════════════════════════════════════
if "분석" in page:
    if st.session_state.analysis_done:
        st.session_state.analysis_done = False
        st.success("✅ 분석이 완료됐어요! '처방 지도' 탭에서 결과를 확인해 보세요.")
        st.balloons()

    st.markdown("## 🤖 분석 실행")
    st.caption("편의 의존도와 복지 회피 인덱스를 결합해 서울 419개 행정동의 Q1 고위험군 전이 위험도를 분석합니다")
    st.divider()

    st.markdown("**Shadow 전이위험도 산출 파이프라인 (4단계)**")
    st.markdown("""
    <div style="margin:12px 0 24px">
      <span class="step-box">📡 1. 편의 의존도 분석</span>
      <span style="color:#bdc3c7;font-size:1.2rem">→</span>
      <span class="step-box">🔗 2. 복지 회피 인덱스 결합</span>
      <span style="color:#bdc3c7;font-size:1.2rem">→</span>
      <span class="step-box">⚖️ 3. 가중 합산</span>
      <span style="color:#bdc3c7;font-size:1.2rem">→</span>
      <span class="step-box">🏷️ 4. 위험등급 부여</span>
    </div>
    """, unsafe_allow_html=True)

    col_stat, _ = st.columns([1, 2])
    with col_stat:
        if df_s is not None:
            ts    = SHADOW / "shadow_prescriptions.csv"
            mtime = datetime.datetime.fromtimestamp(ts.stat().st_mtime).strftime("%Y-%m-%d")
            st.markdown(f"""
            <div style="background:#f0fff4;border:1px solid #2ecc71;border-radius:10px;
                        padding:14px 18px;margin-bottom:16px">
              <b style="color:#27ae60">✅ 분석 결과 있음</b><br>
              <span style="color:#555;font-size:.88rem">마지막 분석: {mtime}</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#fff5f5;border:1px solid #e74c3c;border-radius:10px;
                        padding:14px 18px;margin-bottom:16px">
              <b style="color:#c0392b">❌ 분석 결과 없음</b><br>
              <span style="color:#555;font-size:.88rem">아래 버튼을 눌러 분석을 시작하세요</span>
            </div>""", unsafe_allow_html=True)

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        start = st.button("▶  분석 시작", type="primary", use_container_width=True)

    if start:
        with st.spinner("⏳ 분석 중... (잠시만 기다려 주세요)"):
            result = subprocess.run(
                [PYTHON, str(ROOT / "code" / "run_shadow_ai.py")],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace", cwd=str(ROOT),
            )
        if result.returncode == 0:
            st.session_state.analysis_done = True
            st.session_state.current_page  = "🗺️  처방 지도"
            st.cache_data.clear()
            st.rerun()
        else:
            st.error("❌ 분석 중 오류가 발생했어요.")
            with st.expander("오류 내용 보기"):
                st.code(result.stderr or result.stdout or "알 수 없는 오류", language=None)

    st.divider()
    st.markdown("""
**💡 언제 다시 실행하나요?**
- 새로운 통신·상권 데이터가 추가됐을 때
- 서울서베이 설문 데이터가 갱신됐을 때
- 위험도 예측 모델을 새로 학습했을 때
- 두 축의 가중치를 조정했을 때
""")


# ════════════════════════════════════════════════════════════════════
# PAGE 1: 처방 지도
# ════════════════════════════════════════════════════════════════════
elif "지도" in page:
    _p2 = SHADOW / "shadow_prescriptions.csv"
    df = load_shadow(_mtime=_p2.stat().st_mtime if _p2.exists() else None)
    if df is None:
        st.error("분석 결과 없음 — '분석 실행' 탭에서 먼저 분석을 시작해 주세요.")
        st.stop()

    if st.session_state.analysis_done:
        st.session_state.analysis_done = False
        st.success("✅ 분석 완료! 아래에서 결과를 확인하세요.")

    st.markdown("## 🗺️ Shadow AI Q1 전이 위험 지도")
    st.caption("전이확률(75%) + 복지 회피 인덱스(25%) 결합 — 서울 419개 행정동 Q1 고위험군 전이 위험도")

    c1, c2, c3, c4 = st.columns(4)
    for col, grade in zip([c1, c2, c3, c4], GRADE_ORDER):
        cnt   = int((df["위험등급"] == grade).sum())
        color = GRADE_COLOR[grade]
        col.markdown(f"""
        <div class="grade-card" style="border-top-color:{color}">
          <div class="num" style="color:{color}">{cnt}</div>
          <div class="lbl">{grade}</div>
          <div class="sub">{GRADE_RANGE[grade]}  ·  {cnt}개 행정동</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("")

    col_map, col_rank = st.columns([3, 1])

    with col_map:
        geo = load_geo()
        df["join_key"] = df["자치구"] + "_" + df["행정동"]

        if geo is not None:
            # 빨주노초 계단형 색상 — 등급 경계(50/65/80)에서 정확히 전환
            fig = px.choropleth_mapbox(
                df, geojson=geo, locations="join_key",
                featureidkey="properties.join_key",
                color="Shadow_Score",
                color_continuous_scale=[
                    [0.00, "#2ecc71"],  # 저위험 (초록)
                    [0.50, "#2ecc71"],
                    [0.50, "#f1c40f"],  # 중위험 (노랑)
                    [0.65, "#f1c40f"],
                    [0.65, "#e67e22"],  # 고위험 (주황)
                    [0.80, "#e67e22"],
                    [0.80, "#e74c3c"],  # 최고위험 (빨강)
                    [1.00, "#e74c3c"],
                ],
                range_color=[0, 100],
                hover_name="행정동",
                hover_data={
                    "자치구": True, "Shadow_Score": ":.1f",
                    "위험등급": True, "join_key": False,
                },
                labels={"Shadow_Score": "Shadow Score"},
                mapbox_style="carto-positron",
                center={"lat": 37.5665, "lon": 126.978}, zoom=10,
                height=560, opacity=0.82,
            )
            fig.update_layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                paper_bgcolor="white",
                coloraxis_colorbar=dict(
                    title="Shadow Score",
                    tickvals=[0, 50, 65, 80, 100],
                    ticktext=["0 (저위험)", "50 (중위험↑)", "65 (고위험↑)",
                              "80 (최고위험↑)", "100"],
                    len=0.55, thickness=14,
                ),
                font=dict(family=KO_FONT),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            top50 = df.nlargest(50, "Shadow_Score").copy()
            top50["label"] = top50["자치구"] + " " + top50["행정동"]
            fig_fb = go.Figure()
            for g in GRADE_ORDER:
                sub = top50[top50["위험등급"] == g]
                if sub.empty:
                    continue
                fig_fb.add_trace(go.Bar(
                    x=sub["Shadow_Score"], y=sub["label"],
                    orientation="h", name=g,
                    marker_color=GRADE_COLOR[g],
                    hovertemplate="<b>%{y}</b>  Shadow Score %{x:.1f}<extra></extra>",
                ))
            fig_fb.update_layout(
                height=900, barmode="overlay",
                paper_bgcolor="white", plot_bgcolor="white",
                margin={"t": 20, "b": 10, "l": 0, "r": 60},
                xaxis=dict(range=[0, 105], gridcolor="#f0f0f0", title="Shadow Score"),
                legend=dict(orientation="h", yanchor="bottom", y=1.01,
                            xanchor="right", x=1, font=dict(family=KO_FONT, size=10)),
                font=dict(family=KO_FONT, size=10),
            )
            st.plotly_chart(fig_fb, use_container_width=True)

    with col_rank:
        st.markdown('<div class="sec">🚨 최고위험 행정동</div>', unsafe_allow_html=True)
        top_df = df[df["위험등급"] == "최고위험"].sort_values("Shadow_Score", ascending=False)
        with st.container(height=560, border=False):
            for i, (_, row) in enumerate(top_df.iterrows(), 1):
                dep_c = row["전이확률_정규화"] * 0.75
                avo_c = row["Avoidance"] * 0.25
                st.markdown(f"""
                <div style="border:1px solid #f5b7b1;border-radius:10px;
                            padding:10px 12px;margin-bottom:8px;background:#fff">
                  <div style="font-weight:700;font-size:.92rem;color:#111">
                    {i}. {row['자치구']} {row['행정동']}
                  </div>
                  <div style="font-size:1.6rem;font-weight:800;
                              color:#e74c3c;line-height:1.2">
                    {row['Shadow_Score']:.1f}
                  </div>
                  <div style="margin-top:5px">
                    <span style="background:#fdf2f0;border-radius:4px;padding:2px 6px;
                                 font-size:.75rem;color:#e74c3c;font-weight:600">
                      Dep {dep_c:.1f}
                    </span>
                    &nbsp;
                    <span style="background:#fef6ee;border-radius:4px;padding:2px 6px;
                                 font-size:.75rem;color:#e67e22;font-weight:600">
                      Avo {avo_c:.1f}
                    </span>
                  </div>
                </div>""", unsafe_allow_html=True)



# ════════════════════════════════════════════════════════════════════
# PAGE 2: 행정동 상세
# ════════════════════════════════════════════════════════════════════
elif "행정동" in page:
    df     = load_shadow()
    risk   = load_risk()
    shap_v = load_shap_vals()
    avo_df = load_avoidance()

    if df is None:
        st.error("분석 결과 없음 — '분석 실행' 탭에서 먼저 분석을 시작해 주세요.")
        st.stop()

    st.markdown("## 🔍 행정동 상세 전이위험 분석")

    cs1, cs2 = st.columns([1, 2])
    with cs1:
        gu_list = sorted(df["자치구"].dropna().unique())
        sel_gu  = st.selectbox("자치구", gu_list)
    with cs2:
        dong_df  = df[df["자치구"] == sel_gu].sort_values("Shadow_Score", ascending=False)
        sel_dong = st.selectbox("행정동  (Shadow Score 높은 순)", dong_df["행정동"].tolist())

    row = df[(df["자치구"] == sel_gu) & (df["행정동"] == sel_dong)]
    if row.empty:
        st.warning("해당 행정동 데이터 없음")
        st.stop()
    row   = row.iloc[0]
    grade = str(row["위험등급"])
    gcol  = GRADE_COLOR[grade]
    score = float(row["Shadow_Score"])
    dep_v = float(row["전이확률_정규화"])
    avo_v = float(row["Avoidance"])
    dep_c = dep_v * 0.75
    avo_c = avo_v * 0.25

    st.markdown(
        f"### {sel_gu} **{sel_dong}**  "
        f'<span style="background:{gcol};color:{"white" if grade != "중위험" else "#333"};'
        f'font-size:.9rem;padding:5px 14px;border-radius:20px;font-weight:700">{grade}</span>',
        unsafe_allow_html=True)
    st.divider()

    # ━━━ ① Shadow Score ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown('<div class="sec">① Shadow Score</div>', unsafe_allow_html=True)

    col_g, col_rank2 = st.columns([1.5, 1])
    with col_g:
        fig_main = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"size": 56, "family": KO_FONT}, "valueformat": ".1f"},
            gauge={
                "axis": {"range": [0, 100], "tickfont": {"size": 11}},
                "bar": {"color": gcol, "thickness": 0.28},
                "bgcolor": "white", "borderwidth": 0,
                "steps": [
                    {"range": [0,  50], "color": "#eafaf1"},
                    {"range": [50, 65], "color": "#fef9e7"},
                    {"range": [65, 80], "color": "#fdebd0"},
                    {"range": [80, 100], "color": "#fadbd8"},
                ],
                "threshold": {"line": {"color": gcol, "width": 5},
                              "thickness": 0.75, "value": score},
            },
        ))
        fig_main.update_layout(
            height=260, margin={"t": 20, "b": 0, "l": 20, "r": 20},
            paper_bgcolor="white", font=dict(family=KO_FONT),
        )
        st.plotly_chart(fig_main, use_container_width=True)

    with col_rank2:
        all_scores = df["Shadow_Score"]
        rank    = int((all_scores > score).sum()) + 1
        total   = len(df)
        gu_rank = int(dong_df["행정동"].tolist().index(sel_dong)) + 1
        gu_tot  = len(dong_df)
        st.markdown(f"""
        <div style="padding:24px 10px">
          <div style="font-size:2.6rem;font-weight:800;color:{gcol}">{rank}위</div>
          <div style="color:#666;font-size:.9rem">서울 {total}개 행정동 중</div>
          <div style="color:#888;font-size:.83rem;margin-top:2px">상위 {rank/total*100:.1f}%</div>
          <hr style="border-color:#eee;margin:14px 0">
          <div style="font-size:.85rem;color:#555">
            {sel_gu} 내: <b style="color:{gcol}">{gu_rank}/{gu_tot}위</b>
          </div>
          <div style="font-size:.82rem;color:#888;margin-top:8px">
            서울 평균: <b>{all_scores.mean():.1f}점</b>
          </div>
          <div style="font-size:.82rem;color:#888;margin-top:4px">
            위험등급: <b style="color:{gcol}">{grade}</b>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("**두 축 세부 점수**")
    col_dep, col_avo = st.columns(2)
    with col_dep:
        dep_rank = int((df["전이확률_정규화"] > dep_v).sum()) + 1
        st.plotly_chart(_mini_gauge(
            dep_v, "Dependency", "편의 의존도", "#e74c3c",
            [{"range": [0,  50], "color": "#fef9e7"},
             {"range": [50, 75], "color": "#fdebd0"},
             {"range": [75, 100], "color": "#fadbd8"}],
        ), use_container_width=True)
        st.markdown(f"""
        <div style="text-align:center;margin-top:-10px;font-size:.82rem;color:#555">
          Shadow Score 기여 &nbsp;<b style="color:#e74c3c">{dep_c:.1f}점</b> (×75%)<br>
          서울 내 <b style="color:#e74c3c">{dep_rank}위</b> / {total}개 행정동
        </div>""", unsafe_allow_html=True)
    with col_avo:
        st.plotly_chart(_mini_gauge(
            avo_v, "Avoidance", "복지 회피 인덱스", "#e67e22",
            [{"range": [0,  40], "color": "#fef9e7"},
             {"range": [40, 70], "color": "#fdebd0"},
             {"range": [70, 100], "color": "#fde8c8"}],
        ), use_container_width=True)
        st.markdown(f"""
        <div style="text-align:center;margin-top:-10px;font-size:.82rem;color:#555">
          Shadow Score 기여 &nbsp;<b style="color:#e67e22">{avo_c:.1f}점</b> (×25%)<br>
          자치구 복지 회피도 <b style="color:#e67e22">{avo_v:.1f}점</b>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    if grade == "최고위험":
        msg = (f"🚨 <b>{sel_dong}은 Shadow Score {score:.1f}점으로 즉시 개입이 필요한 최고위험 지역이에요.</b> "
               f"편의 의존도({dep_v:.1f}점)와 복지 회피도({avo_v:.1f}점) 모두 높아 두 위험 요인이 동시에 작용하고 있어요.")
        cls = "insight-warn"
    elif grade == "고위험":
        msg = (f"⚠️ <b>{sel_dong}은 Shadow Score {score:.1f}점으로 고위험 구간이에요.</b> "
               f"편의 의존도({dep_v:.1f}점)가 주된 위험 요인으로, 선제적 모니터링과 접촉이 필요해요.")
        cls = "insight-mid"
    elif grade == "중위험":
        msg = (f"🔶 <b>{sel_dong}은 Shadow Score {score:.1f}점으로 예방 프로그램 연결이 권고되는 구간이에요.</b> "
               f"현재 즉각적 위험은 낮지만 편의 의존도 추이를 지속 관찰할 필요가 있어요.")
        cls = ""
    else:
        msg = (f"✅ <b>{sel_dong}은 Shadow Score {score:.1f}점으로 현재 안정적인 수준이에요.</b> "
               f"서울 전체 평균({all_scores.mean():.1f}점)에 비해 낮은 수준을 유지하고 있어요.")
        cls = "insight-ok"
    st.markdown(insight(msg, cls), unsafe_allow_html=True)
    st.markdown("")

    # ━━━ ② 편의 의존도 연도별 추이 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown('<div class="sec">② 편의 의존도 연도별 추이</div>', unsafe_allow_html=True)

    if risk is not None:
        risk_row = risk[(risk["자치구"] == sel_gu) & (risk["행정동"] == sel_dong)]
        dep_cols = ["dep_2022", "dep_2023", "dep_2024", "dep_2025"]
        avail    = [c for c in dep_cols
                    if not risk_row.empty and c in risk_row.columns
                    and pd.notna(risk_row.iloc[0][c])]
        if avail:
            rr       = risk_row.iloc[0]
            years    = [int(c.split("_")[1]) for c in avail]
            d_vals   = [float(rr[c]) for c in avail]
            all_dep  = risk[avail].values.flatten()
            q75_val  = float(np.nanpercentile(all_dep, 75))
            avg_vals = [float(risk[c].mean()) for c in avail]

            fig_dep = go.Figure()
            fig_dep.add_trace(go.Scatter(
                x=years, y=avg_vals, name="서울 평균",
                mode="lines+markers",
                line=dict(color="#bdc3c7", width=2, dash="dot"),
                marker=dict(size=6),
            ))
            fig_dep.add_trace(go.Scatter(
                x=years, y=d_vals, name=sel_dong,
                mode="lines+markers+text",
                line=dict(color=gcol, width=3), marker=dict(size=10),
                text=[f"{v:.1f}" for v in d_vals],
                textposition="top center",
                textfont=dict(size=12, color=gcol),
            ))
            fig_dep.add_hline(
                y=q75_val, line_color="#e74c3c", line_dash="dash", line_width=1.5,
                annotation_text=f"고위험군 진입 기준선  {q75_val:.0f}점",
                annotation_position="top right",
                annotation_font=dict(color="#e74c3c", size=11, family=KO_FONT),
            )
            fig_dep.add_hrect(y0=q75_val, y1=110, fillcolor="#e74c3c", opacity=0.04, line_width=0)
            fig_dep.update_layout(
                height=300, paper_bgcolor="white", plot_bgcolor="white",
                margin={"t": 20, "b": 10, "l": 0, "r": 0},
                yaxis=dict(range=[0, 110], title="편의 의존도 점수", gridcolor="#f0f0f0"),
                xaxis=dict(tickvals=years, gridcolor="#f0f0f0"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1, font=dict(family=KO_FONT, size=10)),
                font=dict(family=KO_FONT),
            )
            st.plotly_chart(fig_dep, use_container_width=True)
            st.markdown(dep_trend_insight(sel_dong, d_vals, q75_val,
                                          float(row["dep_slope_annual"])),
                        unsafe_allow_html=True)
        else:
            st.info("연도별 편의 의존도 데이터 없음")
    else:
        st.info("위험도 예측 결과가 없어요. '분석 실행' 탭에서 분석을 먼저 실행해 주세요.")
    st.markdown("")

    # ━━━ ③ 편의 의존도 위험 요인 분해 (SHAP) ━━━━━━━━━━━━━━━━━━━━━
    st.markdown('<div class="sec">③ 전이 위험 요인 분해 (SHAP)</div>',
                unsafe_allow_html=True)

    shap_data = None
    if shap_v is not None:
        sr = shap_v[(shap_v.get("자치구", "") == sel_gu) &
                    (shap_v.get("행정동", "") == sel_dong)]
        if not sr.empty:
            meta_c = [c for c in ["행정동코드", "자치구", "행정동", "전이확률"]
                      if c in shap_v.columns]
            feat_c = [c for c in shap_v.columns if c not in meta_c]
            sr     = sr.iloc[0]
            shap_data = {FEAT_LABELS.get(c, c): float(sr[c])
                         for c in feat_c if c in sr.index}

    if shap_data:
        sorted_shap = dict(sorted(shap_data.items(), key=lambda x: x[1], reverse=True))
        names  = list(sorted_shap.keys())
        values = list(sorted_shap.values())
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in values]
        fig_shap = go.Figure(go.Bar(
            x=values[::-1], y=names[::-1],
            orientation="h", marker_color=colors[::-1],
            text=[f"{v:+.4f}" for v in values[::-1]],
            textposition="outside",
        ))
        fig_shap.add_vline(x=0, line_color="#aaa", line_width=1.5)
        x_max = max(abs(v) for v in values) if values else 0.1
        fig_shap.update_layout(
            height=max(280, len(names) * 28),
            margin={"t": 10, "b": 30, "l": 0, "r": 80},
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(range=[-x_max * 1.5, x_max * 1.5],
                       title="← 위험 감소  |  위험 증가 →",
                       title_font=dict(size=10, family=KO_FONT)),
            font=dict(family=KO_FONT, size=11),
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        st.caption("🔴 빨강: 전이 위험을 높이는 요인  /  🔵 파랑: 위험을 낮추는 요인")
        st.markdown(shap_insight(shap_data), unsafe_allow_html=True)
    else:
        st.info("SHAP 분석 결과가 없어요.  \n"
                "`pip install shap` 후 '분석 실행' 탭에서 분석을 다시 실행해 주세요.")
    st.markdown("")

    # ━━━ ④ 복지 회피 인덱스 세부 분석 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown('<div class="sec">④ 복지 회피 인덱스 세부 분석</div>', unsafe_allow_html=True)

    if avo_df is not None:
        gu_avo = avo_df[avo_df["자치구"] == sel_gu]
        avail_comp = {k: v for k, v in AVO_COMP.items() if k in avo_df.columns}

        if not gu_avo.empty and avail_comp:
            gu_row = gu_avo.iloc[0]
            comp_data = {}

            for col, (label, weight) in avail_comp.items():
                col_vals  = avo_df[col].dropna()
                cmin, cmax = col_vals.min(), col_vals.max()
                gu_val    = float(gu_row[col])
                avg_val   = float(col_vals.mean())
                norm_gu   = (gu_val - cmin) / (cmax - cmin) * 100 if cmax > cmin else 50.0
                norm_avg  = (avg_val - cmin) / (cmax - cmin) * 100 if cmax > cmin else 50.0
                # 가중 편차 기여도: SHAP 등가 — 이 요인이 Avoidance를 평균 대비 얼마나 올렸는가
                w_diff    = weight * (norm_gu - norm_avg)
                comp_data[label] = {
                    "raw_gu":      gu_val,
                    "raw_avg":     avg_val,
                    "norm_gu":     norm_gu,
                    "norm_avg":    norm_avg,
                    "w_contrib":   weight * norm_gu,      # 이 자치구의 절대 가중 기여도
                    "w_contrib_avg": weight * norm_avg,   # 서울 평균의 절대 가중 기여도
                    "diff":        w_diff,                # 가중 편차 기여도 (핵심)
                    "weight":      weight,
                }

            labels   = list(comp_data.keys())
            w_diffs  = [comp_data[l]["diff"] for l in labels]
            ngu      = [comp_data[l]["norm_gu"] for l in labels]
            navg     = [comp_data[l]["norm_avg"] for l in labels]
            colors_d = ["#e74c3c" if d > 0 else "#3498db" for d in w_diffs]

            # ── ① 절대 가중 기여도 (자치구 vs 서울 평균 구성 비교) ───────────
            st.markdown(
                f"**절대 가중 기여도** — {sel_gu}의 Avoidance는 어떤 요인으로 구성되어 있는가"
                f"<span style='color:#aaa;font-size:.76rem;font-weight:normal'>"
                f"&nbsp;&nbsp;·&nbsp;&nbsp;값 = 가중치 × 정규화 점수  ·  "
                f"막대 합계 = Avoidance 원시 점수 (최종 정규화 전)</span>",
                unsafe_allow_html=True,
            )
            _COMP_COLORS = {
                "주변 도움 부재율":    "#92400e",
                "외로움 부정 경향":    "#d97706",
                "복지 서비스 불신":    "#fb923c",
                "사회 네트워크 축소율": "#fed7aa",
            }
            fig_contrib = go.Figure()
            for lbl in labels:
                d     = comp_data[lbl]
                color = _COMP_COLORS.get(lbl, "#95a5a6")
                txt_color = "white" if color in ("#92400e", "#d97706", "#fb923c") else "#7c2d12"
                fig_contrib.add_trace(go.Bar(
                    name=lbl,
                    x=[d["w_contrib"], d["w_contrib_avg"]],
                    y=[sel_gu, "서울 평균"],
                    orientation="h",
                    marker_color=color,
                    text=[f"{d['w_contrib']:.1f}", f"{d['w_contrib_avg']:.1f}"],
                    textposition="inside",
                    insidetextanchor="middle",
                    textfont=dict(color=txt_color, size=10),
                    hovertemplate=(
                        f"<b>{lbl}</b><br>"
                        "절대 가중 기여도: %{x:.2f}점<br>"
                        f"가중치: {d['weight']:.0%}"
                        "<extra></extra>"
                    ),
                ))
            fig_contrib.update_layout(
                barmode="stack",
                height=160,
                margin={"t": 10, "b": 10, "l": 0, "r": 10},
                paper_bgcolor="white", plot_bgcolor="white",
                xaxis=dict(
                    title="절대 가중 기여도 합계",
                    title_font=dict(size=10, family=KO_FONT),
                    gridcolor="#f0f0f0",
                ),
                legend=dict(
                    orientation="h", yanchor="top", y=-0.25,
                    xanchor="left", x=0,
                    font=dict(family=KO_FONT, size=10),
                ),
                font=dict(family=KO_FONT, size=11),
            )
            st.plotly_chart(fig_contrib, use_container_width=True)
            st.markdown(w_contrib_insight(sel_gu, avo_v, comp_data), unsafe_allow_html=True)

            # ── ② 가중 편차 기여도 (서울 평균 대비 편차 분해) ──────────────
            st.markdown(
                f"**가중 편차 기여도** — 각 요인이 {sel_gu}의 Avoidance를 서울 평균 대비 얼마나 높이거나 낮추는가"
                f"<span style='color:#aaa;font-size:.76rem;font-weight:normal'>"
                f"&nbsp;&nbsp;·&nbsp;&nbsp;값 = 가중치 × (정규화 점수 − 서울 평균 정규화 점수)</span>",
                unsafe_allow_html=True,
            )
            fig_wdiff = go.Figure()
            fig_wdiff.add_trace(go.Bar(
                x=w_diffs[::-1], y=labels[::-1],
                orientation="h",
                marker_color=colors_d[::-1],
                text=[f"{d:+.2f}" for d in w_diffs[::-1]],
                textposition="outside",
                name=sel_gu,
                customdata=list(zip(
                    ngu[::-1], navg[::-1],
                    [comp_data[l]["weight"] for l in labels[::-1]],
                    w_diffs[::-1],
                )),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    f"{sel_gu} 정규화: %{{customdata[0]:.1f}}점<br>"
                    "서울 평균: %{customdata[1]:.1f}점<br>"
                    "가중치: %{customdata[2]:.0%}<br>"
                    "가중 편차 기여도: %{customdata[3]:+.2f}<extra></extra>"
                ),
            ))
            fig_wdiff.add_vline(x=0, line_color="#aaa", line_width=1.5)
            x_abs = max(abs(d) for d in w_diffs) if w_diffs else 5
            fig_wdiff.update_layout(
                height=220,
                margin={"t": 10, "b": 30, "l": 0, "r": 80},
                paper_bgcolor="white", plot_bgcolor="white",
                xaxis=dict(
                    range=[-(x_abs * 1.8), x_abs * 1.8],
                    title="← Avoidance를 낮추는 방향  |  Avoidance를 높이는 방향 →",
                    title_font=dict(size=10, family=KO_FONT),
                ),
                font=dict(family=KO_FONT, size=11),
            )
            st.plotly_chart(fig_wdiff, use_container_width=True)
            st.caption("🔴 빨강: Avoidance를 끌어올리는 요인  /  🔵 파랑: 낮추는 요인")
            st.markdown(avoidance_insight(sel_gu, avo_v, comp_data), unsafe_allow_html=True)

            with st.expander("📋 세부 항목 수치"):
                rows_avo = []
                for lbl, d in comp_data.items():
                    rows_avo.append({
                        "항목":           lbl,
                        "가중치":         f"{d['weight']:.0%}",
                        f"{sel_gu} 정규화": f"{d['norm_gu']:.1f}",
                        "서울 평균 정규화": f"{d['norm_avg']:.1f}",
                        "가중 기여도":    f"{d['w_contrib']:.2f}",
                        "가중 편차 기여도": f"{d['diff']:+.2f}",
                    })
                st.dataframe(pd.DataFrame(rows_avo), use_container_width=True, hide_index=True)
                st.caption("가중 편차 기여도 = 가중치 × (정규화 점수 − 서울 평균 정규화 점수) · 모든 값의 합 ≈ 이 자치구와 서울 평균의 Avoidance 원시 차이")
        else:
            st.info(f"{sel_gu}의 복지 회피 세부 데이터를 찾을 수 없어요.")
    else:
        st.info("복지 회피 인덱스 데이터를 불러올 수 없어요.")
    st.markdown("")

    # ━━━ ⑤ 같은 자치구 내 행정동 비교 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown('<div class="sec">⑤ 같은 자치구 내 행정동 비교</div>', unsafe_allow_html=True)

    gu_comp = df[df["자치구"] == sel_gu].sort_values("Shadow_Score", ascending=True).copy()
    fig_comp = go.Figure()
    for _, r in gu_comp.iterrows():
        is_sel = r["행정동"] == sel_dong
        fig_comp.add_trace(go.Bar(
            x=[r["Shadow_Score"]], y=[r["행정동"]],
            orientation="h",
            marker_color=GRADE_COLOR[r["위험등급"]],
            marker_opacity=1.0 if is_sel else 0.38,
            showlegend=False,
            hovertemplate=(
                f"<b>{r['행정동']}</b>  Shadow {r['Shadow_Score']:.1f}"
                f"  ({r['위험등급']})<extra></extra>"
            ),
        ))
        fig_comp.add_annotation(
            x=r["Shadow_Score"] + 0.5, y=r["행정동"],
            text=f"{r['Shadow_Score']:.1f}",
            showarrow=False, xanchor="left",
            font=dict(size=9, color="#555", family=KO_FONT),
        )
    fig_comp.update_layout(
        height=max(280, len(gu_comp) * 28),
        paper_bgcolor="white", plot_bgcolor="white",
        margin={"t": 10, "b": 10, "l": 0, "r": 60},
        xaxis=dict(range=[0, 105], gridcolor="#f0f0f0", title="Shadow Score"),
        font=dict(family=KO_FONT, size=10), barmode="overlay",
    )
    st.plotly_chart(fig_comp, use_container_width=True)
    st.caption(f"진한 색 = {sel_dong}  ·  연한 색 = 같은 {sel_gu} 내 다른 행정동")
    st.markdown(gu_comp_insight(sel_dong, sel_gu, score, gu_comp), unsafe_allow_html=True)

    with st.expander("📋 피처 상세 값"):
        if risk is not None:
            risk_row = risk[(risk["자치구"] == sel_gu) & (risk["행정동"] == sel_dong)]
            if not risk_row.empty:
                feat_cols = [c for c in risk_row.columns
                             if any(c.startswith(p) for p in ["level_", "delta_", "infra_"])]
                rows_feat = [{"피처": FEAT_LABELS.get(c, c), "코드": c,
                              "값": round(float(risk_row.iloc[0][c]), 4)}
                             for c in feat_cols if pd.notna(risk_row.iloc[0][c])]
                if rows_feat:
                    st.dataframe(pd.DataFrame(rows_feat),
                                 use_container_width=True, hide_index=True)
