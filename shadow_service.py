"""
SHADOW 서비스 — 통합 셸 (진입점)
================================================
실행:  .venv\\Scripts\\python.exe -m streamlit run shadow_service.py

구성
  1. SHADOW Map
       └─ 🗺️ 진단 맵 (Dependency × Avoidance 4분면)   (views/shadow_map.py)
  2. SHADOW AI
       ├─ 🔮 전이예측   (shadow_dashboard.py — 분석실행/전이위험지도/행정동상세, ★원본 그대로 호출★)
       └─ 💊 RAG 처방   (views/shadow_rag.py — 전이예측 결과를 자치구로 받아 그래프+외부사례 처방)

원칙: shadow_dashboard.py 및 기존 산출물은 일절 수정하지 않는다.
세 페이지가 하나의 공통 테마(흰 배경·카드 시스템)를 공유한다.
"""
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="SHADOW 서비스",
    page_icon="🌃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 공통 테마 (Map · RAG 공통 룩; 전이예측은 자체 스타일 위에 얹힘) ──────────
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
.page-head{
  background:#fff; border:1px solid var(--line); border-radius:16px;
  padding:18px 24px; margin-bottom:18px; border-bottom:4px solid var(--accent);
  box-shadow:0 2px 12px rgba(20,40,80,.05);
}
.page-head h1{ font-size:1.6rem; font-weight:800; margin:0 0 4px; }
.page-head p{ margin:0; font-size:.9rem; color:var(--muted); }
.sec{
  font-size:1.15rem; font-weight:800; color:var(--ink);
  border-left:5px solid var(--accent); padding-left:11px; margin:20px 0 12px;
}
.card{
  background:#fff; border-radius:14px; padding:15px 17px;
  box-shadow:0 2px 10px rgba(20,40,80,.05); border:1px solid var(--line);
}
.grade-card{
  border-radius:14px; padding:16px 12px; text-align:center;
  border-top:5px solid #ccc; background:#fff; border:1px solid var(--line);
  box-shadow:0 2px 10px rgba(20,40,80,.05);
}
.grade-card .num{ font-size:2.2rem; font-weight:800; line-height:1; }
.grade-card .lbl{ font-size:.8rem; color:var(--muted); margin-top:5px; }
.qcard{
  background:#fff; border-radius:14px; padding:14px 16px; height:100%;
  border:1px solid var(--line); border-top:5px solid #ccc;
  box-shadow:0 2px 10px rgba(20,40,80,.05);
}
.qcard .qname{ font-weight:800; font-size:1rem; }
.qcard .qstate{ font-size:.78rem; color:var(--muted); margin:2px 0 8px; }
.qcard .qrx{ font-size:.84rem; color:#34404e; line-height:1.55; }
.chip{ display:inline-block; padding:2px 11px; border-radius:12px;
  font-size:.76rem; font-weight:700; color:#fff; }
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


# ── SHADOW AI · 전이예측: 원본 shadow_dashboard.py 를 '수정 없이' 그대로 실행 ──
def shadow_ai_page():
    """원본 전이예측 대시보드(shadow_dashboard.py)를 한 글자도 안 건드리고 호출한다.

    - shadow_dashboard.py 의 st.set_page_config 는 셸에서 이미 호출했으므로 충돌 →
      실행 중에만 no-op 으로 무력화하고 끝나면 원복.
    - __file__ 을 주입해 그 안의 ROOT/Outputs 경로가 그대로 작동하게 한다.
    """
    import streamlit as _st
    _orig = _st.set_page_config
    _st.set_page_config = lambda *a, **k: None
    try:
        p = ROOT / "shadow_dashboard.py"
        exec(compile(p.read_text(encoding="utf-8"), str(p), "exec"),
             {"__file__": str(p), "__name__": "__main__"})
    finally:
        _st.set_page_config = _orig


# ── 네비게이션 ────────────────────────────────────────────────────────
map_page   = st.Page("views/shadow_map.py", title="진단 맵 (4분면)", icon="🗺️", default=True)
ai_predict = st.Page(shadow_ai_page,        title="전이예측",        icon="🔮")
ai_rag     = st.Page("views/shadow_rag.py", title="RAG 처방",        icon="💊")

pg = st.navigation({
    "1. SHADOW Map": [map_page],
    "2. SHADOW AI":  [ai_predict, ai_rag],
})
pg.run()
