"""
SHADOW 서비스 - 통합 셸 (진입점)
================================================
실행:  .venv\\Scripts\\python.exe -m streamlit run shadow_service.py

구성
  1. SHADOW Map
       └─ 🗺️ 진단 맵 (Dependency × Avoidance 4분면)   (views/shadow_map.py)
  2. SHADOW AI
       ├─ 🔮 전이예측   (shadow_dashboard.py - 분석실행/전이위험지도/행정동상세, ★원본 그대로 호출★)
       └─ 🕸️ 그래프 기반 RAG 처방 (views/shadow_rag.py - 3탭: 지식그래프 → 5단계 파이프라인 → 처방)

원칙: shadow_dashboard.py 및 기존 산출물은 일절 수정하지 않는다.
세 페이지가 하나의 공통 테마(흰 배경·카드 시스템)를 공유한다.
"""
from pathlib import Path
import streamlit as st

import theme

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="SHADOW 서비스",
    page_icon="🌃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 공통 테마 (단일 토큰 소스 theme.py · Map/AI/RAG 전 페이지 공유) ──────────
theme.inject()


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
ai_rag     = st.Page("views/shadow_rag.py", title="그래프 기반 RAG 처방", icon="🕸️")

pg = st.navigation({
    "1. SHADOW Map": [map_page],
    "2. SHADOW AI":  [ai_predict, ai_rag],
})
pg.run()
