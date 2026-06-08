"""
SHADOW 서비스 — 통합 셸 (진입점)
================================================
실행:  .venv\\Scripts\\python.exe -m streamlit run shadow_service.py

구성 (분석 단위 = 자치구로 통일)
  1. SHADOW Map
       └─ 🗺️ 진단 맵 (Dependency × Avoidance 4분면)   (views/shadow_map.py)
  2. SHADOW AI
       ├─ 🔮 전이예측   (views/shadow_predict.py — 자치구별 Q1 전이확률)
       └─ 💊 RAG 처방   (views/shadow_rag.py — 진단+전이예측 → 그래프 + 외부사례 RAG)

세 페이지가 하나의 공통 테마(흰 배경·카드 시스템)를 공유한다.
"""
import streamlit as st

st.set_page_config(
    page_title="SHADOW 서비스",
    page_icon="🌃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 공통 테마 (Map · 전이예측 · RAG 동일 룩) ──────────────────────────────
st.markdown("""
<style>
:root{
  --ink:#1f2d3d; --muted:#7b8794; --line:#e8edf3; --accent:#5b6ee1;
}
.stApp{ background:#ffffff; }
section[data-testid="stSidebar"]{ background:#f4f6f8; }
section[data-testid="stSidebar"] *{ font-family:'Malgun Gothic',sans-serif; }
.block-container{ padding-top:2.2rem; max-width:1280px; }
h1,h2,h3{ color:var(--ink); }

/* 페이지 헤더 */
.page-head{
  background:#fff; border:1px solid var(--line); border-radius:16px;
  padding:18px 24px; margin-bottom:18px;
  border-bottom:4px solid var(--accent);
  box-shadow:0 2px 12px rgba(20,40,80,.05);
}
.page-head h1{ font-size:1.6rem; font-weight:800; margin:0 0 4px; }
.page-head p{ margin:0; font-size:.9rem; color:var(--muted); }

/* 섹션 제목 */
.sec{
  font-size:1.15rem; font-weight:800; color:var(--ink);
  border-left:5px solid var(--accent); padding-left:11px; margin:20px 0 12px;
}

/* 카드 */
.card{
  background:#fff; border-radius:14px; padding:15px 17px;
  box-shadow:0 2px 10px rgba(20,40,80,.05); border:1px solid var(--line);
}

/* 등급/카운트 카드 */
.grade-card{
  border-radius:14px; padding:16px 12px; text-align:center;
  border-top:5px solid #ccc; background:#fff; border:1px solid var(--line);
  box-shadow:0 2px 10px rgba(20,40,80,.05);
}
.grade-card .num{ font-size:2.2rem; font-weight:800; line-height:1; }
.grade-card .lbl{ font-size:.8rem; color:var(--muted); margin-top:5px; }

/* 분면 설명 카드 */
.qcard{
  background:#fff; border-radius:14px; padding:14px 16px; height:100%;
  border:1px solid var(--line); border-top:5px solid #ccc;
  box-shadow:0 2px 10px rgba(20,40,80,.05);
}
.qcard .qname{ font-weight:800; font-size:1rem; }
.qcard .qstate{ font-size:.78rem; color:var(--muted); margin:2px 0 8px; }
.qcard .qrx{ font-size:.84rem; color:#34404e; line-height:1.55; }

/* 칩 */
.chip{ display:inline-block; padding:2px 11px; border-radius:12px;
  font-size:.76rem; font-weight:700; color:#fff; }

/* 인사이트 박스 */
.insight, .note{
  background:#f4f7ff; border-left:4px solid var(--accent);
  border-radius:0 10px 10px 0; padding:11px 15px; margin:8px 0;
  font-size:.88rem; color:#34404e; line-height:1.65;
}
.insight-warn{ background:#fff4f3; border-left-color:#e74c3c; }
.insight-ok{ background:#eefaf2; border-left-color:#27ae60; }
.insight-mid{ background:#fff8ef; border-left-color:#e67e22; }
div[data-testid="stMetricValue"]{ font-weight:800; }
</style>
""", unsafe_allow_html=True)

# ── 네비게이션 ────────────────────────────────────────────────────────
map_page   = st.Page("views/shadow_map.py",     title="진단 맵 (4분면)", icon="🗺️", default=True)
ai_predict = st.Page("views/shadow_predict.py", title="전이예측",        icon="🔮")
ai_rag     = st.Page("views/shadow_rag.py",     title="RAG 처방",        icon="💊")

pg = st.navigation({
    "1. SHADOW Map": [map_page],
    "2. SHADOW AI":  [ai_predict, ai_rag],
})
pg.run()
