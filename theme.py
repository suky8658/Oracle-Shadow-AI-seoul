# -*- coding: utf-8 -*-
"""
SHADOW 디자인 시스템 - 단일 토큰 소스 (Toss 스타일)
==================================================================
색·폰트·radius·그림자·간격을 '이 파일 한 곳'에서만 정의한다.
  · CSS 변수(:root)  → 주입 CSS 전역
  · Python 색 상수    → Plotly 차트 / 인라인 HTML
  · inject()          → st.markdown 으로 전역 스타일 1회 주입

다른 파일(shadow_service / shadow_dashboard / shadow_common / kb / views)은
색을 직접 쓰지 말고 여기서 import 한다.  (단일 소스 원칙)
"""
import streamlit as st

# ════════════════════════════════════════════════════════════════════
# 1. 디자인 토큰 (Toss)  - 값은 '여기서만' 바꾼다
# ════════════════════════════════════════════════════════════════════
COLOR = {
    "bg":        "#F2F4F6",   # 앱 배경 (오프화이트)
    "card":      "#FFFFFF",   # 카드/컨테이너
    "ink":       "#191F28",   # 메인 텍스트
    "muted":     "#8B95A1",   # 보조 텍스트·라벨
    "line":      "#E5E8EB",   # 구분선·그리드
    "blue":      "#3182F6",   # 포인트 (액션·강조·차트)
    "blue_dark": "#1B64DA",   # hover
    "blue_soft": "#E8F1FE",   # 옅은 파랑 배경
    "up":        "#00B26B",   # 상승·긍정
    "up_soft":   "#E7F9F1",
    "down":      "#F04452",   # 하락·부정·위험
    "down_soft": "#FDECEE",
    "amber":     "#FF9500",   # 중간 경고
    "amber_soft":"#FFF4E5",
}

FONT = ("Pretendard, -apple-system, BlinkMacSystemFont, system-ui, "
        "'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif")
# 기존 코드가 KO_FONT 이름으로 참조 → 동일 폰트로 통일
KO_FONT = FONT

RADIUS_CARD = 20
RADIUS_SM = 12
PAD_CARD = 24
GAP = 16
SHADOW = "0 1px 3px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.04)"
SHADOW_HOVER = "0 1px 3px rgba(0,0,0,0.06), 0 8px 24px rgba(0,0,0,0.08)"

# ── 의미 색상 (위험등급·사분면·적합도) - Toss 팔레트로 절제 ──────────────
# 위험등급: 긍정(초록)→부정(빨강) 4단계 시퀀셜. 무지개 대신 의미축에 정렬.
GRADE_COLOR = {
    "최고위험": COLOR["down"],    # 빨강
    "고위험":   COLOR["amber"],   # 주황
    "중위험":   "#FFC542",        # 옅은 앰버
    "저위험":   COLOR["up"],      # 초록
}
GRADE_SOFT = {
    "최고위험": COLOR["down_soft"],
    "고위험":   COLOR["amber_soft"],
    "중위험":   "#FFF8E1",
    "저위험":   COLOR["up_soft"],
}
GRADE_ORDER = ["최고위험", "고위험", "중위험", "저위험"]

# 사분면(범주형): 파랑 중심으로 절제. Q1=위험(빨강), Q3=안전(파랑).
Q_COLORS = {
    "Q1": COLOR["down"],   # 최고위험
    "Q2": COLOR["amber"],  # 정서취약
    "Q3": COLOR["blue"],   # 상대안전
    "Q4": "#5E6B7B",       # 자기해결 (중성 슬레이트)
}

# 적합도 배지
FIT_COLOR = {"적합": COLOR["up"], "부분 적합": COLOR["amber"], "주의 필요": COLOR["down"]}


# ════════════════════════════════════════════════════════════════════
# 2. Plotly 공통 - 차트 색/선 정돈 헬퍼
# ════════════════════════════════════════════════════════════════════
def fig_base(fig, height=None):
    """모든 Plotly 그림에 Toss 톤 공통 레이아웃 적용.
    배경 흰색·옅은 그리드(line)·회색 축라벨·Pretendard."""
    upd = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=COLOR["ink"], size=13),
        margin=dict(t=20, b=20, l=10, r=10),
        xaxis=dict(gridcolor=COLOR["line"], zerolinecolor=COLOR["line"],
                   linecolor=COLOR["line"], tickfont=dict(color=COLOR["muted"])),
        yaxis=dict(gridcolor=COLOR["line"], zerolinecolor=COLOR["line"],
                   linecolor=COLOR["line"], tickfont=dict(color=COLOR["muted"])),
    )
    if height is not None:
        upd["height"] = height
    fig.update_layout(**upd)
    return fig


# ════════════════════════════════════════════════════════════════════
# 3. 전역 CSS - :root 변수 + 컴포넌트
# ════════════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');

:root{
  --bg:#F2F4F6; --card:#FFFFFF; --ink:#191F28; --muted:#8B95A1; --line:#E5E8EB;
  --blue:#3182F6; --blue-dark:#1B64DA; --blue-soft:#E8F1FE;
  --up:#00B26B; --up-soft:#E7F9F1; --down:#F04452; --down-soft:#FDECEE;
  --amber:#FF9500; --amber-soft:#FFF4E5;
  --r-card:20px; --r-sm:12px; --pad:24px; --gap:16px;
  --shadow:0 1px 3px rgba(0,0,0,.04), 0 4px 16px rgba(0,0,0,.04);
  --shadow-hover:0 1px 3px rgba(0,0,0,.06), 0 8px 24px rgba(0,0,0,.08);
  --font:'Pretendard',-apple-system,BlinkMacSystemFont,system-ui,'Malgun Gothic',sans-serif;
}

/* ── 기본 ───────────────────────────────────────────────── */
html, body, .stApp, [class*="css"]{ font-family:var(--font); }
.stApp{ background:var(--bg); color:var(--ink); }
.block-container{ padding-top:2.4rem; padding-bottom:3rem; max-width:1240px; }
h1,h2,h3,h4{ color:var(--ink); letter-spacing:-.01em; }
p, span, div, label, li{ color:var(--ink); }
.stApp [data-testid="stMarkdownContainer"]{ line-height:1.6; }
/* 숫자는 tabular-nums 로 정렬 */
[data-testid="stMetricValue"], .num, .grade-card .num, .qcard *{
  font-feature-settings:"tnum"; font-variant-numeric:tabular-nums;
}

/* ── 사이드바 ───────────────────────────────────────────── */
section[data-testid="stSidebar"]{ background:var(--card); border-right:1px solid var(--line); }
section[data-testid="stSidebar"] *{ font-family:var(--font); }
section[data-testid="stSidebar"] .stRadio label{ font-size:.92rem; }
/* Material 아이콘은 폰트 덮어쓰기 예외 — 안 그러면 ligature가 'keyboard_double_arrow_left' 글자로 깨짐 */
[data-testid="stIconMaterial"], .material-icons,
span[class*="material-symbols"], span[class*="material-icons"]{
  font-family:'Material Symbols Rounded','Material Symbols Outlined','Material Symbols Sharp','Material Icons' !important;
}

/* ── 페이지 헤더 ────────────────────────────────────────── */
.page-head{
  background:var(--card); border:1px solid var(--line); border-radius:var(--r-card);
  padding:20px 26px; margin-bottom:28px; box-shadow:var(--shadow); text-align:left;
}
.page-head h1{ font-size:1.85rem; font-weight:800; margin:0 0 8px; letter-spacing:-.02em; text-align:left; }
.page-head p{ margin:0; font-size:.92rem; color:var(--muted); line-height:1.6; text-align:left; }

/* ── 섹션 제목 (장식선 제거, 위계로만) ──────────────────── */
.sec{
  font-size:1.28rem; font-weight:700; color:var(--ink);
  border-left:none; padding-left:0; margin:38px 0 18px; letter-spacing:-.01em;
}

/* ── 카드 ───────────────────────────────────────────────── */
.card{
  background:var(--card); border:1px solid var(--line); border-radius:var(--r-card);
  padding:var(--pad); box-shadow:var(--shadow); line-height:1.6;
  transition:box-shadow .18s ease;
}
.card:hover{ box-shadow:var(--shadow-hover); }

/* 지표(숫자) 카드 */
.grade-card{
  background:var(--card); border:1px solid var(--line); border-radius:var(--r-card);
  padding:22px 18px; text-align:center; border-top:none;
  box-shadow:var(--shadow); transition:box-shadow .18s ease;
}
.grade-card:hover{ box-shadow:var(--shadow-hover); }
.grade-card .num{ font-size:2.4rem; font-weight:700; line-height:1.1; letter-spacing:-.02em; }
.grade-card .lbl{ font-size:.9rem; color:var(--ink); font-weight:600; margin-top:10px; }
.grade-card .sub{ font-size:.8rem; color:var(--muted); margin-top:7px; }

/* 사분면 카드 */
.qcard{
  background:var(--card); border:1px solid var(--line); border-radius:var(--r-card);
  padding:20px; height:100%; border-top:none; box-shadow:var(--shadow);
  transition:box-shadow .18s ease;
}
.qcard:hover{ box-shadow:var(--shadow-hover); }
.qcard .qname{ font-weight:700; font-size:1.12rem; }
.qcard .qstate{ font-size:.8rem; color:var(--muted); margin:3px 0 12px; }
.qcard .qrx{ font-size:.86rem; color:var(--ink); line-height:1.6; }

/* 배지/칩 - 옅은 배경 + 진한 글자 */
.chip{
  display:inline-block; padding:3px 12px; border-radius:999px;
  font-size:.76rem; font-weight:700; color:#fff;
}

/* 인사이트 박스 - 옅은 배경, 가는 좌측선 */
.insight, .note{
  background:var(--blue-soft); border-left:3px solid var(--blue);
  border-radius:0 var(--r-sm) var(--r-sm) 0; padding:15px 18px; margin:10px 0;
  font-size:1rem; color:var(--ink); line-height:1.75;
}
.insight-warn{ background:var(--down-soft); border-left-color:var(--down); }
.insight-ok  { background:var(--up-soft);   border-left-color:var(--up); }
.insight-mid { background:var(--amber-soft); border-left-color:var(--amber); }

/* 파이프라인 스텝 박스 (구) */
.step-box{
  display:inline-block; background:var(--card); border:1px solid var(--line);
  border-radius:var(--r-sm); padding:12px 18px; margin:4px 6px;
  font-size:.86rem; font-weight:600; color:var(--ink);
  box-shadow:var(--shadow);
}

/* ── 파이프라인 (번호 배지 + 화살표 흐름) ──────────────── */
.pipeline{ display:flex; align-items:stretch; gap:10px; margin:6px 0 24px; }
.pstep{
  flex:1; min-width:0; display:flex; flex-direction:column; align-items:center;
  gap:10px; text-align:center; background:var(--card); border:1px solid var(--line);
  border-radius:var(--r-sm); padding:18px 10px; box-shadow:var(--shadow);
}
.pstep.done{ background:var(--blue-soft); border-color:#CFE0FD; }
.pnum{
  width:28px; height:28px; border-radius:50%; background:var(--blue); color:#fff;
  font-weight:700; font-size:.9rem; display:flex; align-items:center; justify-content:center;
}
.plabel{ font-size:.84rem; font-weight:600; color:var(--ink); line-height:1.45; }
.parrow{ display:flex; align-items:center; color:#CBD5E0; font-size:1.2rem; }

/* ── 랭킹 패널 (흰색~회색 중간 톤 · 지도와 높이 맞춤) ──── */
.rank-panel{
  background:#F7F8FA;                 /* 흰 카드(#FFF) ↔ 페이지 배경(#F2F4F6) 중간 톤 */
  border-radius:var(--r-card); padding:18px 16px 4px;
  height:560px; overflow-y:auto;       /* 지도 높이(560)에 정렬 */
}
.rank-panel::-webkit-scrollbar{ width:6px; }
.rank-panel::-webkit-scrollbar-thumb{ background:#D8DEE5; border-radius:3px; }
.rank-panel-title{
  font-size:1.2rem; font-weight:700; color:var(--ink); margin:2px 4px 14px;
}

/* ── 랭킹 카드 (외곽선 없이 그림자로 띄움) ─────────────── */
.rank-card{
  background:var(--card); border:none; border-radius:14px;
  padding:14px 16px; margin-bottom:12px; box-shadow:var(--shadow);
}
.rank-badge{
  display:inline-flex; align-items:center; justify-content:center; flex:none;
  width:22px; height:22px; border-radius:50%; background:var(--bg);
  color:var(--muted); font-size:.74rem; font-weight:700;
}
.mini-badge{
  display:inline-block; background:var(--bg); color:var(--muted);
  border-radius:6px; padding:2px 8px; font-size:.76rem; font-weight:600;
}

/* ── 상태 배지 (옅은 배경 + 진한 글자) ─────────────────── */
.status-badge{
  display:inline-flex; align-items:center; gap:6px;
  font-size:.84rem; font-weight:600; padding:7px 14px; border-radius:999px;
}
.status-badge.ok{ background:var(--up-soft); color:var(--up); }
.status-badge.none{ background:var(--bg); color:var(--muted); border:1px solid var(--line); }

/* ── 지표(st.metric) ────────────────────────────────────── */
[data-testid="stMetric"]{
  background:var(--card); border:1px solid var(--line); border-radius:var(--r-card);
  padding:18px 20px; box-shadow:var(--shadow);
}
[data-testid="stMetricLabel"]{ color:var(--muted); font-size:.84rem; font-weight:500; }
[data-testid="stMetricValue"]{ color:var(--ink); font-weight:700; letter-spacing:-.02em; }

/* ── 버튼 ───────────────────────────────────────────────── */
.stButton > button{
  border-radius:var(--r-sm); font-weight:600; border:1px solid var(--line);
  padding:.55rem 1.1rem; transition:all .15s ease; box-shadow:none;
}
.stButton > button:hover{ box-shadow:var(--shadow); }
.stButton > button[kind="primary"]{
  background:var(--blue); border-color:var(--blue); color:#fff;
}
.stButton > button[kind="primary"]:hover{
  background:var(--blue-dark); border-color:var(--blue-dark);
}

/* ── 섹션 카드 = 흰색 살짝-둥근 패널 ─────────────────────
   Streamlit 1.57 보더 컨테이너는 stVerticalBlock 에 off-white(lightenedBg05)
   배경이 박혀 있어, 카드 안에 심은 <span.sc-card> 마커를 :has() 로 집어
   '그 컨테이너만' 흰색으로 덮어쓴다. (버전 무관·정확) */
.sc-card{ display:none; }
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] span.sc-card){
  background:#FFFFFF !important;
  border:1px solid var(--line) !important;
  border-radius:14px !important;          /* 살짝만 둥글게 */
  padding:18px 22px !important;
  box-shadow:var(--shadow) !important;
  margin-bottom:16px;
}
/* 테두리 없는 흰 카드 변형 (border=False 컨테이너 + 이 마커 → 흰 배경+그림자만, 선 없음).
   ※ 바깥 보더 래퍼를 :has 로 잡으면 상위 컨테이너까지 번지므로, 안쪽 블록만 정밀 타깃. */
.sc-card-nobd{ display:none; }
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] span.sc-card-nobd){
  background:#FFFFFF !important;
  border:none !important;
  border-radius:14px !important;
  padding:18px 22px !important;
  box-shadow:var(--shadow) !important;
  margin-bottom:16px;
}

/* ── 챗봇 위젯 (레퍼런스 스타일) ──────────────────────────── */
.chat-wrap-mark, .chat-body-mark{ display:none; }
/* 위젯 전체 폭 제한 (입력창 포함) */
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] span.chat-wrap-mark){
  max-width:900px;
}
/* 회색 말풍선 영역 (헤더 카드 바로 아래에 이어붙음) */
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] span.chat-body-mark){
  background:#F7F8FA !important; border:1px solid var(--line) !important;
  border-radius:0 0 14px 14px !important; padding:14px 16px !important;
}
.chat-head{
  display:flex; align-items:center; gap:11px; background:var(--card);
  border:1px solid var(--line); border-bottom:none;
  border-radius:14px 14px 0 0; padding:13px 16px;
}
.chat-ava{
  width:36px; height:36px; border-radius:50%; background:var(--blue); color:#fff;
  display:flex; align-items:center; justify-content:center; font-size:1.15rem; flex:none;
}
.chat-name{ font-weight:800; font-size:1rem; color:var(--ink); line-height:1.2; }
.chat-status{ font-size:.74rem; color:var(--up); display:flex; align-items:center; gap:5px; margin-top:2px; }
.chat-status i{ width:7px; height:7px; border-radius:50%; background:var(--up); display:inline-block; }
.bubble-row{ display:flex; margin-bottom:9px; }
.bubble-row:last-child{ margin-bottom:0; }
.bubble-row.user{ justify-content:flex-end; }
.bubble-row.bot{ justify-content:flex-start; }
.bubble{ max-width:82%; padding:10px 14px; border-radius:16px; font-size:.9rem; line-height:1.55; }
.bubble.bot{ background:#fff; color:var(--ink); border:1px solid var(--line); border-bottom-left-radius:5px; }
.bubble.user{ background:var(--blue); color:#fff; border-bottom-right-radius:5px; }
.bubble.user b{ color:#fff; }

/* ── ⑤ 이식 제안 카드 (LabsAdvisor 스타일 · 대시보드 파랑) ── */
.rx-head{
  display:flex; align-items:center; justify-content:space-between;
  background:var(--blue); border-radius:10px; padding:11px 16px; margin-bottom:14px;
}
.rx-title{ color:#fff; font-weight:800; font-size:1.04rem; }
.rx-badge{ background:#fff; font-size:.72rem; font-weight:800; padding:3px 12px; border-radius:999px; }
.rx-meta{ font-size:.86rem; color:#4E5968; }
.rx-attrs{ display:flex; flex-wrap:wrap; gap:7px; margin-top:10px; }
.rx-attr{
  background:var(--blue-soft); color:var(--blue-dark); font-size:.76rem; font-weight:600;
  padding:4px 11px; border-radius:999px;
}
.rx-box{ background:#F7F8FA; border-radius:12px; padding:13px 15px; margin-bottom:10px; }
.rx-box-row{
  display:flex; justify-content:space-between; align-items:center;
  font-size:.82rem; color:var(--muted); padding:3px 0;
}
.rx-box-row b{ color:var(--ink); font-weight:700; }

/* ── 처방전 문서 (℞ 레터헤드 양식) ────────────────────────── */
.rx-doc-head{ display:flex; align-items:flex-start; gap:12px; }
.rx-rx{ font-family:Georgia,'Times New Roman',serif; font-size:2rem; font-weight:700;
  color:var(--blue); line-height:1; }
.rx-doc-title-wrap{ flex:1; }
.rx-doc-title{ font-size:1.3rem; font-weight:800; color:var(--ink); line-height:1.1; }
.rx-doc-sub{ font-size:.7rem; color:var(--muted); letter-spacing:.22em; }
.rx-issuer{ text-align:right; font-size:.82rem; color:var(--muted); line-height:1.45; }
.rx-issuer b{ color:var(--ink); font-size:.9rem; }
.rx-rule{ border:none; border-top:1px solid var(--line); margin:8px 0; }
.rx-rule-strong{ border:none; border-top:2px solid var(--blue); margin:10px 0 14px; }
.rx-meta-grid{ display:flex; flex-wrap:wrap; gap:26px; font-size:.85rem; color:var(--ink); }
.rx-meta-grid .k{ color:var(--muted); font-weight:600; margin-right:8px; }
.rx-foot{ display:flex; justify-content:space-between; align-items:center; gap:12px;
  margin-top:14px; font-size:.76rem; color:var(--muted); }
.rx-sign{ font-weight:700; color:var(--ink); }
/* 처방전 폼 필드(대상/지역/유형/등급) — 양식처럼 라벨+값+밑줄 */
.rx-form{ display:grid; grid-template-columns:1fr 1fr; gap:0 26px; margin:2px 0 0; }
.rx-field{ display:flex; align-items:baseline; gap:12px; padding:9px 2px;
  border-bottom:1px solid var(--line); }
.rx-flabel{ font-size:.7rem; color:var(--muted); font-weight:700; letter-spacing:.06em;
  min-width:54px; }
.rx-fval{ font-size:.94rem; color:var(--ink); font-weight:700; }

/* 차트는 배경 투명(theme 기본) → 흰 카드 위에서 자동으로 흰색으로 보임.
   .main-svg 에 배경을 칠하면 Plotly 상단 투명 레이어가 불투명해져 차트를
   가리므로, 차트 배경은 건드리지 않는다. */
.streamlit-expanderHeader, [data-testid="stExpander"] summary{
  border-radius:var(--r-sm); font-weight:600;
}
[data-testid="stExpander"]{ border:1px solid var(--line); border-radius:var(--r-card); }
[data-testid="stDataFrame"]{ border-radius:var(--r-sm); overflow:hidden; }

/* ── 입력 위젯 ──────────────────────────────────────────── */
[data-baseweb="select"] > div{ border-radius:var(--r-sm); border-color:var(--line); }
.stTextInput input, .stNumberInput input{ border-radius:var(--r-sm); }

/* 캡션 색 통일 */
[data-testid="stCaptionContainer"]{ color:var(--muted); }
hr{ border-color:var(--line); }
</style>
"""


def inject():
    """전역 Toss 테마를 1회 주입한다. (중복 주입은 무해)"""
    st.markdown(CSS, unsafe_allow_html=True)
