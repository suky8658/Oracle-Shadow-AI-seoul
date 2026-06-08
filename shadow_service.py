"""
SHADOW 서비스 — 통합 셸 (진입점)
================================================
실행:  .venv\\Scripts\\python.exe -m streamlit run shadow_service.py

구성
  1. SHADOW Map
       ├─ 🗺️ 서울 위험 지도   (views/shadow_map.py)
       └─ 💊 RAG 처방        (views/shadow_rag.py)
  2. SHADOW AI
       └─ 🤖 전이예측 대시보드  (shadow_dashboard.py — ★수정하지 않고 그대로 호출★)

원칙: shadow_dashboard.py 및 기존 산출물은 일절 수정하지 않는다.
      이 셸은 기존 코드를 '감싸서' 호출하고, Outputs/*.csv 를 읽어 화면을 구성한다.
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


# ── SHADOW AI: 기존 dashboard.py 를 '수정 없이' 그대로 실행 ──────────────
def shadow_ai_page():
    """업데이트된 shadow_dashboard.py 를 한 글자도 건드리지 않고 호출한다.

    - shadow_dashboard.py 가 부르는 st.set_page_config 는 통합 셸에서 이미 호출했으므로
      충돌한다. → 실행 중에만 잠깐 no-op 으로 무력화하고, 끝나면 원복.
    - __file__ 을 shadow_dashboard.py 경로로 주입해야 그 안의 `ROOT/Outputs` 데이터
      로딩 경로가 원래대로 작동한다.
    """
    import streamlit as _st
    _orig_cfg = _st.set_page_config
    _st.set_page_config = lambda *a, **k: None
    try:
        p = ROOT / "shadow_dashboard.py"
        code = compile(p.read_text(encoding="utf-8"), str(p), "exec")
        exec(code, {"__file__": str(p), "__name__": "__main__"})
    finally:
        _st.set_page_config = _orig_cfg


# ── 네비게이션 (2단 그룹) ────────────────────────────────────────────────
# SHADOW Map 페이지 안에 [진단 맵] + [RAG 처방] 탭이 함께 들어있다.
map_page = st.Page("views/shadow_map.py", title="진단 맵 + RAG 처방", icon="🗺️", default=True)
ai_page  = st.Page(shadow_ai_page,        title="전이예측 대시보드",   icon="🤖")

pg = st.navigation({
    "1. SHADOW Map (진단 + 처방)": [map_page],
    "2. SHADOW AI (미래 전이예측)": [ai_page],
})
pg.run()
