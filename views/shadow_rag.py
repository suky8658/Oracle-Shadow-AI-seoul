"""
🕸️ SHADOW AI ▸ 그래프 기반 RAG 처방 (자치구 단위) - ADB 엔진 연동판
================================================================
이 페이지는 더 이상 로컬 목업(kb/)으로 그리지 않는다.
팀원이 구현한 ★Oracle ADB 그래프 엔진★(shadow_rag_llm.py)이 진단·한계·근거논문·
이식후보를 '결정적으로' 계산해 주고, 화면은 그 결과(payload)를 그대로 보여준다.
마지막 ⑤ 처방문만 OpenAI가 실시간 스트리밍으로 생성한다. (무환각 뉴로-심볼릭)

데이터 흐름:
  자치구 선택
   → [ADB] get_prescription(gu, generate_text=False)  ← OpenAI 호출 없음 = 무료
        · 지역_프로파일 (Q·회피/의존점수·주동인·고위험 행정동)
        · 현행_제도 / 진단(한계+need+근거논문) / 이식후보
   → 화면: ①현행 → ②한계 → ③방향 → ④외부레퍼런스 (그래프 결과 그대로)
   → [OpenAI] stream_prescription(payload)  ← ⑤ 처방문만 실시간 생성(과금)

사이드바 라디오 서브내비:
  🕸️ 지식그래프      - 그래프 추론이 실제로 무엇을 찾았는지 요약(+ 시각화 자리)
  ⚙️ 5단계 파이프라인 - ①현행 → ②한계⭐ → ③방향 → ④외부레퍼런스 (ADB 그래프 결과)
  💊 처방            - 버튼 하나로 ⑤ 이식 제안을 OpenAI가 실시간 생성

자치구 선택은 session_state['sel_gu']로 전이예측 페이지와 공유한다.
원본 목업은 views/shadow_rag.py.mockup_bak 에 보관.
"""
import re
import sys
from datetime import date
from pathlib import Path

import streamlit as st


def _md_bold_to_html(s: str) -> str:
    """처방문은 HTML <span>과 마크다운 **굵게**가 섞여 있다. Streamlit이 HTML 섞인
    문서에서는 마크다운 **를 처리 안 해 '**'가 그대로 보이므로, 렌더 직전에 직접 <strong>으로 변환."""
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s or "")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shadow_common import aggregate_gu, Q_LABEL, GRADE_COLOR, Q_COLORS  # noqa: E402
from shadow_rag_llm import get_prescription, stream_prescription  # noqa: E402
from shadow_chat import get_chat_context, stream_chat  # noqa: E402
from rag_graph import render_rag_graph  # noqa: E402
from theme import COLOR  # noqa: E402

# 그래프 노드 타입 → 한글 라벨 (클릭 패널용)
NODE_TYPE_KO = {
    "Q": "진단 유형", "prog": "현행 제도", "conf": "낙인·의존 충돌",
    "limit": "구조적 한계", "need": "필요 방향", "cand": "이식후보", "gap": "구조적 빈틈",
}

# 전이예측 산출물이 없을 때를 위한 자치구 폴백 목록
SEOUL_GU = [
    "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구",
    "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구",
    "구로구", "금천구", "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구",
]

# 외부 레퍼런스 출처 위계 색 (엔진 reference_priority 값과 1:1)
SRC_COLOR = {"해외": COLOR["blue"], "국내타지역": COLOR["amber"], "서울타자치구": "#5E6B7B"}
SRC_ORDER = {"해외": 0, "국내타지역": 1, "서울타자치구": 2}

# RAG 5단계 다이어그램 (논증 흐름)
PIPELINE = [
    ("현행 제도", "이미 이런 제도들이 있긴 함"),
    ("한계 진단", "낙인 충돌·귀속·중복 배제"),
    ("필요 방향", "그래서 이런 방향이 필요"),
    ("외부 레퍼런스", "해외·타지역·타자치구서 검증"),
    ("이식 제안", "이 지역에도 마련돼야"),
]

# 생성 모델 선택지 (mini=저렴/개발용, 4o=고품질/시연용)
MODEL_OPTS = {
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o",
}


@st.cache_data(show_spinner=False)
def load_facts(gu):
    """ADB 그래프에서 진단·한계·이식후보를 추출 (OpenAI 호출 없음 = 무료). 자치구별 캐시.
    반환은 get_prescription 결과 dict: {자치구, 처방문(None), 사실, 토큰, 에러}."""
    return get_prescription(gu, generate_text=False)


def _f(v, fmt="{:.1f}", dash="-"):
    """None/비수치 안전 포맷."""
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return dash


# ════════════════════════════════════════════════════════════════════
# 헤더 + 공유 자치구 선택
# ════════════════════════════════════════════════════════════════════
st.markdown("<div class='page-head'><h1>그래프 기반 RAG 처방 · 자치구</h1>"
            "<p>Oracle ADB 지식그래프가 기존 제도의 한계를 진단하고, "
            "타지역·해외 사례를 그래프 추론으로 검색해 이식을 제안합니다.</p></div>",
            unsafe_allow_html=True)

# 자치구 목록 - 전이예측 집계가 있으면 그걸, 없으면 폴백
gdf = aggregate_gu()
gu_list = gdf["자치구"].tolist() if gdf is not None else SEOUL_GU
default = st.session_state.get("sel_gu", gu_list[0])
if default not in gu_list:
    default = gu_list[0]
gu = st.selectbox("자치구", gu_list, index=gu_list.index(default), key="rag_gu")
st.session_state["sel_gu"] = gu

# ★ ADB 그래프에서 이 자치구의 사실(진단·한계·이식후보) 추출 (무료·캐시)
with st.spinner(f"{gu} - ADB 그래프에서 진단 추출 중..."):
    res = load_facts(gu)
if res["에러"]:
    st.error(f"ADB에서 **{gu}** 분석 데이터를 불러오지 못했어요.\n\n{res['에러']}")
    st.info("`.env`(DB 접속정보·OPENAI_API_KEY)와 `wallet/` 폴더가 프로젝트 루트에 있는지 확인하세요.")
    st.stop()

P = res["사실"]
rp = P["지역_프로파일"]
q = str(rp.get("quadrant", "-"))

# 헤더 칩 - 전이확률·등급은 전이예측 집계에서(있으면), Q는 ADB 그래프에서
q1p = grade = None
if gdf is not None and gu in set(gdf["자치구"]):
    hrow = gdf[gdf["자치구"] == gu].iloc[0]
    q1p = float(hrow["Q1전이확률"])
    grade = str(hrow["등급"])

qcol = Q_COLORS.get(q, "#95a5a6")
chips = f"<span class='chip' style='background:{qcol}'>{Q_LABEL.get(q, q)}</span> "
if grade is not None:
    chips += f"<span class='chip' style='background:{GRADE_COLOR.get(grade, '#95a5a6')}'>{grade}</span> "
if q1p is not None:
    chips += f"<span style='color:#8B95A1;font-size:.9rem'>· Q1 전이확률 {q1p:.1f}%</span>"
st.markdown(f"#### 📍 {gu} {chips}", unsafe_allow_html=True)


# ── 진단·이식후보 인덱싱 (서브페이지 공통) ────────────────────────────
진단 = P.get("진단", [])
이식후보 = {t["need_id"]: t.get("candidates", []) for t in P.get("이식후보", [])}
# need_id → need 이름/정의 (③ 방향용, 진단에서 dedupe)
needs = []
seen_need = set()
for d in 진단:
    nd = d.get("need", {})
    nid = nd.get("id")
    if nid and nid not in seen_need:
        seen_need.add(nid)
        needs.append(nd)


# ── 사이드바 서브내비 ──────────────────────────────────────────────
SUBPAGES = ["🕸️  지식그래프", "💊  처방", "🤖  처방전·챗봇"]
with st.sidebar:
    st.markdown("<div style='font-size:.82rem;color:#8B95A1;font-weight:600;"
                "margin:6px 0 4px'>RAG 처방 단계</div>", unsafe_allow_html=True)
    sub = st.radio("RAG 처방 단계", SUBPAGES, key="rag_sub", label_visibility="collapsed")


# ════════════════════════════════════════════════════════════════════
# 서브페이지 0: 지식그래프 (그래프 추론이 실제로 무엇을 찾았는지)
# ════════════════════════════════════════════════════════════════════
if "지식그래프" in sub:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("현행 제도", f"{len(P.get('현행_제도', []))}건",
              help="이 자치구(또는 서울전체)에서 5060 남성 당사자에게 닿는 제도")
    c2.metric("진단된 한계", f"{len(진단)}개",
              help="제도↔낙인/의존 요소를 그래프로 조인해 '논리적으로' 도출한 구조적 한계")
    cand_total = sum(len(v) for v in 이식후보.values())
    c3.metric("이식후보 (그래프 추론)", f"{cand_total}건",
              help="이 자치구엔 없고, 이 유형의 민감/취약 요소와 충돌하지 않는 외부 제도")
    ev_total = len({e['evidence_id'] for d in 진단 for e in d.get('근거_논문', [])})
    c4.metric("매칭된 근거논문", f"{ev_total}편",
              help="각 한계를 뒷받침하는 근거(EVIDENCE 논문) 매칭 수")

    sens = rp.get("quadrant_민감낙인", [])
    vuln = rp.get("quadrant_취약의존", [])

    def _chip(x, bg, fg):
        return (f"<span style='background:{bg};color:{fg};padding:3px 11px;border-radius:999px;"
                f"font-size:.8rem;font-weight:700;margin:2px 4px 2px 0;display:inline-block'>{x}</span>")
    none_html = "<span style='color:#8B95A1'>없음</span>"
    sens_html = "".join(_chip(s, COLOR["down_soft"], COLOR["down"]) for s in sens) or none_html
    vuln_html = "".join(_chip(v, COLOR["amber_soft"], COLOR["amber"]) for v in vuln) or none_html
    st.markdown(
        "<div class='card' style='margin-top:14px'>"
        f"<div style='font-weight:700;margin-bottom:10px'>이 유형({q})이 민감해하는 요소</div>"
        f"<div style='line-height:2.1'><b>민감 낙인</b>&nbsp;&nbsp;{sens_html}<br>"
        f"<b>취약 의존</b>&nbsp;&nbsp;{vuln_html}</div>"
        "<div style='color:#8B95A1;font-size:.82rem;margin-top:10px'>"
        "아래 그래프에서, 제도가 이 요소를 건드리는 곳이 빨간색으로 표시됩니다.</div>"
        "</div>", unsafe_allow_html=True)

    # ── 인터랙티브 지식그래프 (payload → 계층형 Plotly) ───────────────
    st.markdown(f"<div class='sec' style='margin-top:18px'>지식그래프 - "
                f"{gu} 그래프 추론 경로</div>", unsafe_allow_html=True)
    fig, node_index = render_rag_graph(P)
    ev = st.plotly_chart(fig, use_container_width=True, on_select="rerun",
                         key=f"rag_graph_{gu}")

    # 선택된 노드 → 경로(trace) 설명 패널 (Map 사이드진단 / 예측 SHAP 자리에 대응)
    sel_id = None
    try:
        pts = ev.selection["points"] if (ev and ev.selection) else []
        if pts:
            sel_id = pts[-1]["customdata"][0]
    except (KeyError, IndexError, TypeError):
        sel_id = None

    if sel_id and sel_id in node_index:
        nd = node_index[sel_id]
        st.markdown(
            f"<div class='card' style='border-left:4px solid {nd['color']};margin-top:6px'>"
            f"<div style='font-weight:800;margin-bottom:6px'>{nd['label']} "
            f"<span style='color:#8B95A1;font-size:.82rem;font-weight:600'>· "
            f"{NODE_TYPE_KO.get(nd['type'], nd['type'])}</span></div>"
            f"<div style='font-size:.9rem;line-height:1.7'>{nd['detail']}</div></div>",
            unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# 서브페이지 2: 🤖 처방전·챗봇 (LLM 문장형 처방 생성 + 챗봇 자리)
# ════════════════════════════════════════════════════════════════════
elif "챗봇" in sub:
    st.markdown(f"<div class='sec'>처방전 - GPT-4o가 {gu} 처방전을 실시간으로 작성합니다.</div>",
                unsafe_allow_html=True)

    model_label = st.selectbox("생성 모델", list(MODEL_OPTS.keys()), index=0,
                               help="개발/테스트는 mini(저렴), 시연은 gpt-4o 권장")
    model = MODEL_OPTS[model_label]

    text_key = f"rx_text_{gu}"
    pressed = st.button("▶ AI 처방 생성", type="primary", use_container_width=True, key=f"rx_{gu}")

    def _rx_doc_header():
        # 처방전 ℞ 레터헤드 + 폼 필드(대상/지역/유형/위험등급)
        gradetxt = grade if grade else "-"
        gradecol = GRADE_COLOR.get(grade, COLOR["muted"])
        st.markdown(
            "<div class='rx-doc-head'>"
            "<span class='rx-rx'>℞</span>"
            "<div class='rx-doc-title-wrap'>"
            "<div class='rx-doc-title'>처방전</div>"
            "<div class='rx-doc-sub'>PRESCRIPTION</div></div>"
            f"<div class='rx-issuer'><b>SHADOW-AI</b><br>{date.today().isoformat()}"
            f"<br><span style='font-size:.7rem'>{model}</span></div></div>"
            "<hr class='rx-rule'>"
            "<div class='rx-form'>"
            f"<div class='rx-field'><span class='rx-flabel'>대상</span>"
            f"<span class='rx-fval'>{P.get('대상', '')}</span></div>"
            f"<div class='rx-field'><span class='rx-flabel'>지역</span>"
            f"<span class='rx-fval'>{gu}</span></div>"
            f"<div class='rx-field'><span class='rx-flabel'>유형</span>"
            f"<span class='rx-fval' style='color:{Q_COLORS.get(q, COLOR['ink'])}'>"
            f"{Q_LABEL.get(q, q)}</span></div>"
            f"<div class='rx-field'><span class='rx-flabel'>위험등급</span>"
            f"<span class='rx-fval' style='color:{gradecol}'>{gradetxt}</span></div>"
            "</div><hr class='rx-rule-strong'>",
            unsafe_allow_html=True)

    def _rx_doc_footer():
        st.markdown(
            "<hr class='rx-rule'>"
            "<div class='rx-foot' style='justify-content:flex-end'>"
            "<span class='rx-sign'>SHADOW-AI 정책분석 엔진</span></div>",
            unsafe_allow_html=True)

    if pressed:
        # 처방전 문서: ℞ 레터헤드 + 본문 실시간 스트리밍 + 서명 푸터
        # (HTML 파란 강조를 그리려면 unsafe_allow_html 필요 → placeholder 직접 렌더)
        with st.container(border=False):
            st.markdown("<span class='sc-card-nobd'></span>", unsafe_allow_html=True)
            _rx_doc_header()
            ph = st.empty()
            acc = ""
            try:
                for delta in stream_prescription(P, model):
                    acc += delta
                    ph.markdown(_md_bold_to_html(acc), unsafe_allow_html=True)
                st.session_state[text_key] = acc
                _rx_doc_footer()
            except Exception as e:
                st.error(f"처방 생성(OpenAI) 오류: {e}")
    elif st.session_state.get(text_key):
        st.caption("✓ 생성된 처방 (재생성하려면 ▶ 버튼)")
        with st.container(border=False):
            st.markdown("<span class='sc-card-nobd'></span>", unsafe_allow_html=True)
            _rx_doc_header()
            st.markdown(_md_bold_to_html(st.session_state[text_key]), unsafe_allow_html=True)
            _rx_doc_footer()

    with st.expander("이 처방의 근거 보기 (논문 · 이식 제도)"):
        st.markdown("**근거 논문**")
        seen = set()
        any_ev = False
        for d in 진단:
            for e in d.get("근거_논문", []):
                if e["evidence_id"] in seen:
                    continue
                seen.add(e["evidence_id"])
                any_ev = True
                st.markdown(f"- ({e.get('authors', '')}, {e.get('year', '')}) {e.get('title', '')}")
        if not any_ev:
            st.caption("매칭된 근거 논문 없음")
        st.markdown("**이식 후보 제도**")
        any_cand = False
        for nid, cands in 이식후보.items():
            for c in cands:
                any_cand = True
                st.markdown(f"- [{c.get('reference_priority', '')}] {c.get('name', '')}  ·  "
                            f"{', '.join(c.get('regions', []) or [])}")
        if not any_cand:
            st.caption("이식 가능한 외부 제도 없음 = 구조적 빈틈 → 서울 전역 확대 등 신규 구현 필요")
        st.caption("※ 본 처방은 자치구 단위 정책 설계 신호이며, 개인 단위 처방이 아닙니다.")

    # ── 챗봇 (shadow_chat 엔진 연동) ─────────────────────────────────
    st.divider()   # 처방전(위) ↔ 챗봇(아래) 시각적 구분
    st.markdown("<div class='sec'>🤖 챗봇 · 무엇이든 물어보기</div>", unsafe_allow_html=True)
    st.caption(f"{gu}의 분석 데이터(진단·처방 · 회피/의존 축분해 · 행정동 SHAP·전이확률 · "
               "논문 71편 · 25개구 순위)를 근거로 답합니다. 데이터에 없으면 솔직히 알려주고, "
               "무관한 질문은 거절해요.")

    # 자치구가 바뀌면 챗봇 컨텍스트 새로 로드 + 대화 초기화 (탭 전환만으론 유지)
    if st.session_state.get("rag_chat_gu") != gu:
        with st.spinner(f"{gu} 챗봇 데이터 불러오는 중..."):
            cr = get_chat_context(gu)
        if cr["에러"]:
            st.error(f"챗봇 데이터를 불러오지 못했어요: {cr['에러']}")
            st.session_state["rag_chat_ctx"] = None
        else:
            st.session_state["rag_chat_ctx"] = cr["context"]
        st.session_state["rag_chat_gu"] = gu
        st.session_state["rag_chat_hist"] = []

    if st.session_state.get("rag_chat_ctx"):
        hist = st.session_state["rag_chat_hist"]
        WELCOME = (f"안녕하세요! {gu}의 진단·처방 결과를 두고 무엇이든 물어보세요. "
                   "데이터에 있는 사실만 근거로 답해 드려요.")
        SUGGEST = [f"{gu}가 왜 취약한가요?", "회피점수를 분해하면?",
                   "고위험 행정동은 어디인가요?"]

        def _bubble(role, text):
            cls = "user" if role == "user" else "bot"
            body = (text or "").replace(chr(10), "<br>")
            return (f"<div class='bubble-row {cls}'>"
                    f"<div class='bubble {cls}'>{body}</div></div>")

        with st.container():
            st.markdown("<span class='chat-wrap-mark'></span>", unsafe_allow_html=True)

            # 헤더 바
            st.markdown(
                "<div class='chat-head'><div class='chat-ava'>🤖</div>"
                f"<div><div class='chat-name'>SHADOW 챗봇 · {gu}</div>"
                "<div class='chat-status'><i></i>온라인</div></div></div>",
                unsafe_allow_html=True)

            # 말풍선 영역 (봇=왼쪽 회색, 사용자=오른쪽 파랑)
            with st.container():
                st.markdown("<span class='chat-body-mark'></span>", unsafe_allow_html=True)
                st.markdown(_bubble("bot", WELCOME), unsafe_allow_html=True)
                for m in hist:
                    st.markdown(_bubble(m["role"], m["content"]), unsafe_allow_html=True)
                # 미응답 질문이 있으면 답변 실시간 생성
                if hist and hist[-1]["role"] == "user":
                    ph = st.empty()
                    acc = ""
                    try:
                        # 챗봇 컨텍스트가 커서(논문 71편 등) gpt-4o TPM(30k) 초과 →
                        # TPM 여유 큰 gpt-4o-mini 로 고정 (근거 기반 QA엔 충분)
                        for delta in stream_chat(st.session_state["rag_chat_ctx"],
                                                 hist[:-1], hist[-1]["content"],
                                                 model="gpt-4o-mini"):
                            acc += delta
                            ph.markdown(_bubble("bot", acc), unsafe_allow_html=True)
                        hist.append({"role": "assistant", "content": acc})
                    except Exception as e:
                        ph.error(f"답변 생성 오류: {e}")

            # 추천 질문 (대화 시작 전에만) + 입력창
            if not hist:
                sc = st.columns(len(SUGGEST))
                for i, sug in enumerate(SUGGEST):
                    if sc[i].button(sug, key=f"sug_{i}_{gu}", use_container_width=True):
                        st.session_state["rag_chat_hist"].append({"role": "user", "content": sug})
                        st.rerun()
            if prompt := st.chat_input(f"{gu}에 대해 물어보세요"):
                st.session_state["rag_chat_hist"].append({"role": "user", "content": prompt})
                st.rerun()


# ════════════════════════════════════════════════════════════════════
# 서브페이지 1: 💊 처방 (파이프라인 ①~⑤ · 모두 그래프 결과)
# ════════════════════════════════════════════════════════════════════
else:
    parts = []
    for i, (label, desc) in enumerate(PIPELINE, 1):
        cls = "pstep done"
        parts.append(
            f"<div class='{cls}'><div class='pnum'>{i}</div>"
            f"<div class='plabel'>{label}<br>"
            f"<span style='font-size:.72rem;color:#8B95A1;font-weight:400'>{desc}</span>"
            f"</div></div>")
        if i < len(PIPELINE):
            parts.append("<div class='parrow'>→</div>")
    st.markdown("<div class='pipeline'>" + "".join(parts) + "</div>", unsafe_allow_html=True)

    # ── 진단 요약 ────────────────────────────────────────────────────
    고위험동 = rp.get("고위험_행정동", [])
    st.markdown(
        f"<div style='font-size:1.34rem;font-weight:800;color:#191F28;"
        f"margin:20px 0 12px;letter-spacing:-.01em'>진단 요약"
        f"<span style='font-size:.92rem;font-weight:500;color:#8B95A1'>"
        f" · {gu}의 현재 상황</span></div>", unsafe_allow_html=True)
    동명 = ", ".join(d.get("행정동", "") for d in 고위험동)

    # 전이예측(가중편차기여도) 부호별 칩 — 빨강=위험 가중(+), 초록=위험 완충(−)
    def _chips(items, color, bg):
        return "".join(
            f"<span style='display:inline-block;background:{bg};color:{color};"
            f"border-radius:6px;padding:1px 7px;margin:3px 4px 0 0;font-size:.72rem;"
            f"font-weight:600'>{x}</span>" for x in items)

    악화 = rp.get("회피_악화요인", []) or []
    완충 = rp.get("회피_완충요인", []) or []
    sub_parts = []
    if 악화:
        sub_parts.append("<div style='margin-top:6px'>"
                         "<span style='color:#E03131;font-weight:700;font-size:.72rem'>▲ 위험 가중</span> "
                         + _chips(악화, "#E03131", "#FFF0F0") + "</div>")
    if 완충:
        sub_parts.append("<div style='margin-top:3px'>"
                         "<span style='color:#2F9E44;font-weight:700;font-size:.72rem'>▼ 위험 완충</span> "
                         + _chips(완충, "#2F9E44", "#EBFBEE") + "</div>")
    회피_sub = "".join(sub_parts) or (
        "주동인: " + " · ".join(rp.get("회피_주동인", []) or []))

    def _stat(label, value, sub):
        sub_html = (f"<div style='font-size:.78rem;color:#8B95A1;margin-top:7px;"
                    f"line-height:1.5'>{sub}</div>") if sub else ""
        return (f"<div style='flex:1;min-width:0'>"
                f"<div style='font-size:.84rem;color:#8B95A1;font-weight:500'>{label}</div>"
                f"<div style='font-size:2rem;font-weight:700;color:#191F28;"
                f"letter-spacing:-.02em;margin-top:4px'>{value}</div>"
                f"{sub_html}</div>")

    st.markdown(
        "<div class='card'><div style='display:flex;gap:28px'>"
        + _stat("회피점수 (복지 회피)", _f(rp.get("회피점수")), 회피_sub)
        + _stat("의존점수 (편의 의존)", _f(rp.get("의존점수")), "")
        + _stat("고위험 행정동", f"{len(고위험동)}곳", 동명)
        + "</div></div>", unsafe_allow_html=True)

    # 고위험 행정동 상세 — 전이확률 + 주요악화요인(SHAP 위험↑)
    if 고위험동:
        rows_html = ""
        for d in 고위험동:
            prob = d.get("전이확률")
            try:
                prob_txt = f"{float(prob) * 100:.1f}%"
            except (TypeError, ValueError):
                prob_txt = "-"
            악화f = " · ".join(d.get("주요악화요인", []) or []) or "-"
            rows_html += (
                f"<div style='display:flex;align-items:center;gap:12px;padding:9px 0;"
                f"border-top:1px solid #F1F3F5'>"
                f"<div style='font-weight:700;color:#191F28;min-width:84px'>{d.get('행정동','-')}</div>"
                f"<div style='color:#E03131;font-weight:700;min-width:62px'>{prob_txt}</div>"
                f"<div style='color:#495057;font-size:.82rem'>위험요인: {악화f}</div>"
                f"<div style='margin-left:auto;font-size:.74rem;color:#8B95A1'>{d.get('위험등급','')}</div>"
                f"</div>")
        st.markdown(
            "<div class='card' style='margin-top:10px'>"
            "<div style='font-size:.9rem;font-weight:700;color:#191F28;margin-bottom:2px'>"
            "고위험 행정동 · 전이확률 &amp; 주요 위험요인(SHAP)</div>"
            + rows_html + "</div>", unsafe_allow_html=True)

    # ── ① 현행 제도 ──────────────────────────────────────────────────
    st.markdown("<div class='sec'>① 현행 제도 · 이미 시행 중인 제도</div>",
                unsafe_allow_html=True)
    현행 = P.get("현행_제도", [])
    st.caption(f"이 자치구(또는 서울 전체)에서 시행 중이고, 50·60대 남성 당사자에게 닿는 제도 "
               f"- 총 **{len(현행)}건**")

    def _prog_card(p, pad="13px 16px"):
        return (f"<div class='card' style='padding:{pad};margin-bottom:10px'>"
                f"<b>{p.get('name', '-')}</b></div>")

    @st.dialog(f"{gu} 현행 제도 전체 ({len(현행)}건)", width="large")
    def _all_programs():
        kw = st.text_input("제도 검색", placeholder="제도명으로 검색",
                           key=f"prog_search_{gu}")
        rows = [p for p in 현행 if not kw or kw.lower() in (p.get("name") or "").lower()]
        st.caption(f"{len(rows)} / {len(현행)}건")
        for p in rows:
            st.markdown(_prog_card(p, pad="10px 14px"), unsafe_allow_html=True)

    if 현행:
        # fulfills_needs 가 많은(연관도 높은) 제도 우선 4건
        top = sorted(현행, key=lambda p: -len(p.get("fulfills_needs") or []))[:4]
        cols = st.columns(2)
        for i, p in enumerate(top):
            with cols[i % 2]:
                st.markdown(_prog_card(p), unsafe_allow_html=True)
        if len(현행) > len(top):
            if st.button(f"전체 {len(현행)}건 보기 · 검색", use_container_width=True,
                         key=f"all_prog_btn_{gu}"):
                _all_programs()
    else:
        st.info("이 자치구에 매칭된 현행 제도가 없습니다.")

    # ── ② 한계 진단 ────────────────────────────────────────────────────
    st.markdown("<div class='sec'>② 한계 진단 · 기존 제도가 놓치는 지점</div>",
                unsafe_allow_html=True)
    st.caption("제도가 어떤 낙인·의존 요소를 건드리는지 따져서 찾아낸 한계입니다. "
               "각 한계에는 근거 논문이 함께 붙습니다.")
    if 진단:
        @st.dialog("근거 논문", width="large")
        def _ev_dialog(한계, 근거):
            st.markdown(f"**{한계}** · 근거 논문 {len(근거)}편")
            for e in 근거:
                kf = e.get("key_finding", "") or ""
                meta = " · ".join(x for x in [
                    e.get("authors", ""), str(e.get("year", "") or ""),
                    e.get("scope", "")] if x)
                kf_html = (f"<div style='font-size:.86rem;color:#4E5968;line-height:1.65;"
                           f"margin-top:6px'>“{kf}”</div>") if kf else ""
                st.markdown(
                    "<div style='background:#F7F8FA;border-left:3px solid #C7CDD4;"
                    "border-radius:8px;padding:12px 15px;margin-bottom:10px'>"
                    f"<div style='font-weight:700;font-size:.92rem;color:#191F28'>"
                    f"{e.get('title', '')}</div>"
                    f"<div style='font-size:.76rem;color:#8B95A1;margin-top:3px'>{meta}</div>"
                    f"{kf_html}</div>", unsafe_allow_html=True)

        for i in range(0, len(진단), 2):
            cols = st.columns(2)
            for col, d in zip(cols, 진단[i:i + 2]):
                nd = d.get("need", {})
                근거 = d.get("근거_논문", [])
                with col, st.container(border=False):
                    st.markdown("<span class='sc-card-nobd'></span>", unsafe_allow_html=True)
                    def_html = (f"<div style='font-size:.8rem;color:#8B95A1;margin-top:4px'>"
                                f"{nd.get('def', '')}</div>") if nd.get("def") else ""
                    st.markdown(
                        "<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px'>"
                        f"<span style='background:{COLOR['down']};color:#fff;font-weight:800;"
                        "font-size:.72rem;padding:3px 10px;border-radius:8px'>한계</span>"
                        f"<span style='font-weight:800;font-size:1rem;color:{COLOR['down']}'>"
                        f"{d.get('한계')}</span></div>"
                        "<div style='font-size:.86rem;color:#4E5968;line-height:1.55'>"
                        f"원인 제도 <b>{d.get('원인_제도_총수', 0)}건</b> → 필요 방향 "
                        f"<b style='color:{COLOR['blue']}'>{nd.get('name', '')}</b></div>"
                        f"{def_html}",
                        unsafe_allow_html=True)
                    if 근거:
                        if st.button(f"근거 논문 {len(근거)}편 보기",
                                     key=f"ev_{gu}_{d.get('한계')}",
                                     use_container_width=True):
                            _ev_dialog(d.get("한계"), 근거)
    else:
        st.info("그래프가 진단한 구조적 한계가 없습니다(상대적으로 안전).")

    # ── ③ 필요한 처방 방향 ────────────────────────────────────────────
    st.markdown("<div class='sec'>③ 처방 방향 · 한계가 가리키는 방향</div>",
                unsafe_allow_html=True)
    if needs:
        ncols = st.columns(len(needs))
        for col, nd in zip(ncols, needs):
            col.markdown(
                f"<div class='card'><b>{nd.get('name', '')}</b><br>"
                f"<span style='color:#8B95A1;font-size:.85rem'>{nd.get('def', '')}</span></div>",
                unsafe_allow_html=True)
    else:
        st.caption("도출된 필요 방향이 없습니다.")

    # ── ④ 외부 레퍼런스 (이식후보) ────────────────────────────────────
    st.markdown("<div class='sec'>④ 외부 레퍼런스 · 다른 지역의 검증된 사례</div>",
                unsafe_allow_html=True)
    st.caption("③에서 필요하다고 본 방향을 충족하면서, 이 자치구엔 아직 없고, 이 유형과 부딪히지 않는 "
               "다른 지역의 실제 제도입니다. 해외 · 국내 타지역 · 서울 타자치구 순으로 보여줍니다.")
    # 외부 사례(이식후보)가 실제로 있는 처방 방향만 보여준다 (없는 방향은 숨김)
    shown = 0
    for nd in needs:
        cands = sorted(이식후보.get(nd.get("id"), []),
                       key=lambda c: SRC_ORDER.get(c.get("reference_priority"), 9))
        if not cands:
            continue
        shown += 1
        st.markdown(f"<div style='font-weight:700;margin:16px 0 6px'>{nd.get('name', '')} "
                    f"<span style='color:#8B95A1;font-size:.85rem'>({len(cands)}건)</span></div>",
                    unsafe_allow_html=True)
        cc = st.columns(min(3, len(cands)))
        for col, c in zip(cc, cands[:3]):
            src = c.get("reference_priority", "서울타자치구")
            scol = SRC_COLOR.get(src, "#5E6B7B")
            regions = ", ".join(c.get("regions", []) or [])
            rationale = c.get("rationale", "") or ""
            col.markdown(
                f"<div class='card' style='height:100%'>"
                f"<span class='chip' style='background:{scol};font-size:.68rem'>{src}</span>"
                f"<div style='font-weight:800;margin-top:8px'>{c.get('name', '-')}</div>"
                f"<div style='font-size:.76rem;color:#3182F6;font-weight:700'>{regions}</div>"
                f"<div style='font-size:.82rem;color:#191F28;margin-top:6px;line-height:1.55'>"
                f"{rationale}</div></div>",
                unsafe_allow_html=True)
    if shown == 0:
        st.caption("표시할 외부 레퍼런스가 없습니다.")

    # ── ⑤ 이식 제안 (그래프 기반 · 이 자치구에 도입 제안) ──────────────
    st.markdown("<div class='sec'>⑤ 이식 제안</div>", unsafe_allow_html=True)
    st.caption(f"④에서 검증된 외부 사례 가운데 {gu}에 우선 도입을 제안합니다.")
    제안수 = 0
    for nd in needs:
        cands = sorted(이식후보.get(nd.get("id"), []),
                       key=lambda c: SRC_ORDER.get(c.get("reference_priority"), 9))
        if not cands:
            continue
        제안수 += 1
        top = cands[0]
        src = top.get("reference_priority", "서울타자치구")
        scol = SRC_COLOR.get(src, "#5E6B7B")
        regions = ", ".join(top.get("regions", []) or [])
        rationale_html = (f"<div style='font-size:.86rem;color:#4E5968;margin-top:6px;"
                          f"line-height:1.55'>{top.get('rationale', '')}</div>"
                          ) if top.get("rationale") else ""
        st.markdown(
            f"<div class='card' style='margin-bottom:12px'>"
            f"<div style='font-size:.82rem;color:#8B95A1;font-weight:600'>"
            f"‘{nd.get('name', '')}’ 방향 · 도입 제안</div>"
            f"<div style='margin-top:7px'>"
            f"<span class='chip' style='background:{scol};font-size:.68rem'>{src}</span> "
            f"<b style='font-size:1.02rem'>{top.get('name', '-')}</b> "
            f"<span style='color:#3182F6;font-size:.8rem;font-weight:700'>{regions}</span></div>"
            f"{rationale_html}"
            f"<div style='font-size:.88rem;color:#191F28;margin-top:9px'>"
            f"→ 이 방식을 <b>{gu}</b>에 도입할 것을 제안합니다.</div>"
            f"</div>", unsafe_allow_html=True)
    if 제안수 == 0:
        st.info("이식할 외부 사례가 없어, 서울 전역 확대 등 신규 제도 구현이 필요합니다.")
