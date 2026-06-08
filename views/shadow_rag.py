"""
💊 SHADOW AI ▸ RAG 처방 (자치구 단위)
================================================================
SHADOW Map 진단(현재 Q) + 전이예측(Q1 전이확률)을 함께 입력으로 받아,
자치구 단위로 처방을 생성한다.

흐름 (자치구 1개의 하나의 논증):
  ① 현재 진단(Q) + 전이예측 현황 (Q1 전이확률)
  ② 이 Q지역에 필요한 제도적 특성 (낙인회피 / 복지의존)
  ③ retrieve — 그래프 1홉으로 '이런 제도들이 있습니다'
  ④ 한계 — 기존 제도의 트레이드오프·구조적 한계 (그래프 충돌 + 구조 한계)
  ⑤ 처방 — 외국/타지역 사례를 검색(RAG)해 '이런 제도가 마련되어야 한다' 생성

앞단(②③④)=그래프(정확·결정적), 뒷단(⑤)=외부사례 RAG(생성). 하이브리드.
"""
import os
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shadow_common import aggregate_gu, gu_top_drivers, GRADE_COLOR, Q_COLORS, Q_LABEL  # noqa: E402
from kb.prescriptions import KB  # noqa: E402
from kb.graph import build_graph, retrieve as graph_retrieve  # noqa: E402
from kb.cases import retrieve_cases  # noqa: E402

FIT_COLOR = {"적합": "#27ae60", "부분 적합": "#e67e22", "주의 필요": "#c0392b"}

# Q유형별 '필요 제도특성' (검색 needs)
NEEDS = {
    "Q1": ["낙인회피", "복지의존", "간접발굴"],
    "Q2": ["복지의존"],
    "Q3": ["경량관찰"],
    "Q4": ["낙인회피", "간접발굴"],
}
NEED_DESC = {
    "낙인회피": "복지 라벨·노출·신청을 요구하지 않는 무낙인 설계",
    "복지의존": "비대면 자족을 외출·관계로 되돌리는 설계",
    "간접발굴": "당사자가 안 나서도 생활권 지역망으로 포착",
    "경량관찰": "저비용 주기 모니터링으로 변화 조기 감지",
}


@st.cache_resource
def get_graph():
    return build_graph()


st.markdown("<div class='page-head'><h1>💊 RAG 처방 · 자치구</h1>"
            "<p>현재 진단(Q)과 전이예측(Q1 전이확률)을 함께 받아, 그래프로 기존 제도를 "
            "진단하고 외국·타지역 사례를 검색해 처방합니다.</p></div>",
            unsafe_allow_html=True)

gdf = aggregate_gu()
if gdf is None:
    st.error("전이예측 결과가 없어요. **전이예측** 페이지를 먼저 확인해 주세요.")
    st.stop()

gu_list = gdf["자치구"].tolist()
default = st.session_state.get("sel_gu", gu_list[0])
if default not in gu_list:
    default = gu_list[0]
gu = st.selectbox("자치구", gu_list, index=gu_list.index(default), key="rag_gu")
st.session_state["sel_gu"] = gu

row = gdf[gdf["자치구"] == gu].iloc[0]
q = str(row.get("Quadrant", "-"))
if q not in KB:
    st.error(f"진단 Q유형({q})에 대한 지식베이스가 없어요.")
    st.stop()
kb = KB[q]
q1p = float(row["Q1전이확률"])
grade = str(row["등급"])
gcol = GRADE_COLOR[grade]
qcol = Q_COLORS.get(q, "#95a5a6")
avo = float(row["Avoidance"])
needs = NEEDS.get(q, [])
drivers = gu_top_drivers(gu, k=3)

# ── ① 현재 진단 + 전이예측 현황 ───────────────────────────────────────
st.markdown(
    f"## 📍 {gu} "
    f"<span class='chip' style='background:{qcol}'>{Q_LABEL.get(q, q)}</span> "
    f"<span class='chip' style='background:{gcol}'>{grade}</span>",
    unsafe_allow_html=True,
)
m1, m2, m3 = st.columns(3)
m1.metric("Q1 전이확률 (자치구)", f"{q1p:.1f}%")
m2.metric("최고위험 행정동", f"{int(row['n_최고위험'])}/{int(row['n_행정동'])}")
m3.metric("Avoidance (복지 회피)", f"{avo:.1f}")

urgent = grade in ("최고위험", "고위험")
note_cls = "insight-warn" if urgent else "insight-ok"
urgency = ("Q1 고위험군 **전이 위험이 높습니다 — 각별한 주의가 필요**합니다."
           if urgent else "Q1 전이 위험은 상대적으로 낮은 편입니다.")
st.markdown(
    f"<div class='insight {note_cls}'>📈 <b>예측 현황</b> — {gu}는 현재 "
    f"<b>{Q_LABEL.get(q, q)}</b>(Avoidance {avo:.1f})이며, 전이예측상 "
    f"Q1 전이확률 <b>{q1p:.1f}%</b>. {urgency}</div>",
    unsafe_allow_html=True,
)
st.markdown(f"<div class='insight'>🧠 <b>이 지역의 심리기저</b> — {kb['심리기저']}</div>",
            unsafe_allow_html=True)

# ── ② 필요 제도특성 ───────────────────────────────────────────────────
st.markdown("<div class='sec'>② 이 Q지역에 필요한 제도적 특성</div>", unsafe_allow_html=True)
ncols = st.columns(len(needs) if needs else 1)
for col, nd in zip(ncols, needs):
    col.markdown(
        f"<div class='card'><b>{nd}</b><br>"
        f"<span style='color:#7b8794;font-size:.85rem'>{NEED_DESC.get(nd, '')}</span></div>",
        unsafe_allow_html=True,
    )
graph_result = graph_retrieve(get_graph(), q)
sens = graph_result["sensitive"]
if sens:
    st.caption("⚡ 이 유형이 특히 민감한 낙인요소: " + " · ".join(f"`{s}`" for s in sens)
               + " — 제도가 이걸 건드리면 역효과.")

# ── ③ retrieve ────────────────────────────────────────────────────────
st.markdown("<div class='sec'>③ 검색(Retrieve) — 이런 제도들이 있습니다</div>", unsafe_allow_html=True)
st.caption(f"🧭 그래프 1홉: **{q}** →[권장]→ 제도 →[자극]→ 낙인요소  ∩  {q} →[민감] = 충돌")
for hit in graph_result["hits"]:
    fc = FIT_COLOR[hit["fit"]]
    with st.container(border=True):
        st.markdown(
            f"**{hit['name']}** · {hit['주최']} "
            f"<span class='chip' style='background:{fc};font-size:.72rem'>{hit['fit']}</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"{hit['내용']} ({hit['시행']})")
        st.caption(f"📎 {hit['url']}")

# ── ④ 한계 ────────────────────────────────────────────────────────────
st.markdown("<div class='sec'>④ 한계 — 기존 제도로는 부족한 점</div>", unsafe_allow_html=True)
limits = []
all_conf = sorted({c for h in graph_result["hits"] for c in h["conflicts"]})
if all_conf:
    limits.append(f"**낙인 충돌** — 일부 제도가 이 지역이 민감한 "
                  f"<b>{' · '.join(all_conf)}</b>를 자극해, 발굴력은 있어도 실제 연결로 "
                  f"이어지지 않을 위험이 있습니다.")
if q in ("Q1", "Q2"):
    limits.append("**의존 심화 트레이드오프** — 비대면·편의형 개입은 낙인을 낮추지만, "
                  "동시에 비대면 자족(복지의존)을 더 굳힐 수 있습니다.")
limits.append("**자치구 귀속** — 다수 제도가 특정 자치구 예산·운영 주체에 묶여 있어, "
              "동일 모델을 다른 자치구로 그대로 옮기기 어렵습니다.")
limits.append("**중복 수혜 배제** — 특정 제도를 이미 수혜받으면 유사 사업 대상에서 "
              "제외돼, 사각지대가 다시 생길 수 있습니다.")
st.markdown(
    "<div class='card'>" + "".join(
        f"<div style='margin:4px 0;line-height:1.6'>• {x}</div>" for x in limits
    ) + "</div>", unsafe_allow_html=True,
)

# ── ⑤ 처방 (외국/타지역 사례 RAG) ─────────────────────────────────────
st.markdown("<div class='sec'>⑤ 처방 — 외국·타지역 사례 기반</div>", unsafe_allow_html=True)
GEN_BADGE = ("<span style='background:#fff3cd;color:#856404;padding:2px 10px;border-radius:10px;"
             "font-size:.78rem;font-weight:700'>RAG 생성</span> "
             "<span style='color:#999;font-size:.8rem'>— 외부 사례 코퍼스에서 이 Q에 맞는 사례를 "
             "검색해, 전이예측·한계를 근거로 처방을 구성합니다.</span>")
cases = retrieve_cases(q, needs, k=3)

driver_line = ""
if drivers:
    driver_line = "위험동인(SHAP): " + ", ".join(f"{n}" for n, _ in drivers) + ". "


def build_rx():
    head = (f"**{gu}**는 현재 {Q_LABEL.get(q, q)}이며 Q1 전이확률 {q1p:.1f}%"
            f"({grade})입니다. {driver_line}앞서 본 기존 제도는 "
            f"{'낙인 충돌·' if all_conf else ''}자치구 귀속·중복 배제의 한계가 있습니다. "
            f"이를 메우려면 아래 사례처럼 **{', '.join(needs)}**를 갖춘 제도적 복지가 "
            f"{gu}에 마련되어야 합니다.")
    bullets = []
    for c in cases:
        bullets.append(f"- **{c['name']}** ({c['지역']}) — {c['시사점']}")
    tail = ("\n\n결론적으로, 위 사례의 설계 원리를 "
            f"{gu}의 {Q_LABEL.get(q, q)} 특성에 맞게 이식하는 것이 처방의 방향입니다. "
            "(자치구 단위 신호이며 개인 단위 처방이 아닙니다.)")
    return head + "\n\n" + "\n".join(bullets) + tail


st.markdown(GEN_BADGE, unsafe_allow_html=True)
if cases:
    cc = st.columns(len(cases))
    for col, c in zip(cc, cases):
        col.markdown(
            f"<div class='card' style='height:100%'>"
            f"<div style='font-weight:800'>{c['name']}</div>"
            f"<div style='font-size:.76rem;color:#5b6ee1;font-weight:700'>{c['지역']}</div>"
            f"<div style='font-size:.83rem;color:#34404e;margin-top:6px;line-height:1.55'>"
            f"{c['설명']}</div></div>",
            unsafe_allow_html=True,
        )
    st.markdown("")

pressed = st.button("▶ 처방 생성", type="primary", key=f"rx_{gu}")
done_key = f"rx_done_{gu}"
if pressed:
    st.session_state[done_key] = True
if st.session_state.get(done_key):
    if pressed:
        with st.spinner("🧠 전이예측·한계·외부사례로 처방 생성 중..."):
            time.sleep(0.7)

        def streamer():
            for para in build_rx().split("\n\n"):
                yield para + "\n\n"
                time.sleep(0.3)
        with st.container(border=True):
            st.write_stream(streamer)
    else:
        with st.container(border=True):
            st.markdown(build_rx())
else:
    st.info("▶ 버튼을 누르면 외부 사례를 근거로 처방이 생성됩니다.")

# ── 출처 · LLM ────────────────────────────────────────────────────────
with st.expander("📎 출처 · 한계 명시"):
    st.markdown("**기존 제도 (서울/국내)**")
    for i, inst in enumerate(kb.get("제도후보", []), 1):
        st.markdown(f"[제도{i}] {inst['name']} — {inst['url']}")
    st.markdown("**참고 사례 (외국/타지역)**")
    for c in cases:
        st.markdown(f"· {c['name']} ({c['지역']})")
    st.caption("※ 외부 사례는 개념 수준 기술이며 정확한 수치/링크는 발표 전 확인 권장. "
               "본 처방은 자치구 단위 정책 설계 신호입니다.")

with st.expander("✨ 실제 LLM 연결 (Claude API)"):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.info("환경변수 `ANTHROPIC_API_KEY` 설정 시, 전이예측·그래프·외부사례를 컨텍스트로 "
                "넘겨 Claude가 처방을 실시간 생성합니다. 키 없이도 위 처방은 표시됩니다.")
    else:
        if st.button("처방문 실시간 생성", key=f"llm_{gu}"):
            try:
                import anthropic
                client = anthropic.Anthropic()
                inst_txt = "\n".join(
                    f"- {h['name']} ({h['주최']}) · 적합도 {h['fit']} · 충돌 {h['conflicts'] or '없음'}"
                    for h in graph_result["hits"])
                case_txt = "\n".join(f"- {c['name']} ({c['지역']}): {c['설명']} → {c['시사점']}"
                                     for c in cases)
                prompt = (
                    "당신은 복지 정책 상담사다. 아래 정보만 근거로(밖의 사실 생성 금지) "
                    f"{gu}(현재 {q}, Q1 전이확률 {q1p:.1f}%, 위험등급 {grade}, Avoidance {avo:.1f})에 "
                    f"대한 자치구 단위 처방을 쓰라. {driver_line}기존 제도의 한계를 짚고, "
                    "외부 사례의 설계 원리를 이 지역 특성에 맞게 이식하는 방향으로 제안하라. "
                    "마지막에 '자치구 단위 신호'임을 명시하라.\n\n"
                    f"[처방방향] {kb.get('처방방향','')}\n[기존 제도]\n{inst_txt}\n[외부 사례]\n{case_txt}"
                )
                resp = client.messages.create(model="claude-sonnet-4-6", max_tokens=900,
                                              messages=[{"role": "user", "content": prompt}])
                st.markdown(resp.content[0].text)
            except Exception as e:
                st.error(f"생성 실패: {e}")
