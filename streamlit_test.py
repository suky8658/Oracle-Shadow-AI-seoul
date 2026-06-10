# -*- coding: utf-8 -*-
"""
streamlit_test.py — 처방 엔진(shadow_rag_llm) 작동 확인용 '미니' 대시보드.

목적:
  · 내 엔진이 streamlit 화면에서 잘 도는지 눈으로 확인
  · 친구가 진짜 대시보드(shadow_service.py)에 연결할 때 "이렇게 쓰면 된다"는 살아있는 예시
  · '처방 시작' 버튼 → 진단 즉시 표시(무료) → 처방문 실시간 타이핑(OpenAI 스트리밍)

실행:
  streamlit run streamlit_test.py
  (.env 와 wallet/ 가 같은 폴더에 있어야 함)
"""
import streamlit as st
from shadow_rag_llm import get_prescription, stream_prescription

SEOUL_GU = [
    "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구",
    "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구",
    "구로구", "금천구", "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구",
]

st.set_page_config(page_title="SHADOW 처방 (테스트)", page_icon="🩺")
st.title("🩺 SHADOW 처방 — 미니 테스트")
st.caption("ADB 그래프로 진단 → OpenAI로 처방 생성 (실시간)")

gu = st.selectbox("자치구 선택", SEOUL_GU, index=SEOUL_GU.index("노원구"))

if st.button("처방 시작", type="primary"):
    # 1) 진단(그래프 추론)만 먼저 — 빠르고 무료(OpenAI 호출 안 함)
    with st.spinner(f"{gu} — ADB 그래프에서 진단 추출 중..."):
        r = get_prescription(gu, generate_text=False)

    if r["에러"]:
        st.error(r["에러"])           # 자치구 오타·접속 실패 등 → 화면 안 깨짐
        st.stop()

    p = r["사실"]
    rp = p["지역_프로파일"]

    # 진단 카드
    st.subheader(f"📊 {gu} 진단")
    c1, c2, c3 = st.columns(3)
    c1.metric("사분면", rp["quadrant"])
    c2.metric("회피점수", f'{float(rp["회피점수"]):.1f}' if rp.get("회피점수") is not None else "-")
    고위험 = ", ".join(d["행정동"] for d in rp.get("고위험_행정동", [])) or "-"
    c3.metric("고위험 행정동", 고위험)
    st.caption("회피 주동인: " + " · ".join(rp.get("회피_주동인", [])))
    st.caption("진단된 한계: " + "  |  ".join(
        f"{d['한계']}→{d['need']['name']}({d['원인_제도_총수']}건)" for d in p["진단"]))

    # 2) 처방문 — OpenAI 스트리밍으로 실시간 타이핑
    st.subheader("📝 처방문")
    full_text = st.write_stream(stream_prescription(p, "gpt-4o"))   # ★ 핵심 한 줄

    # 3) 근거(논문·이식제도) — 펼쳐보기
    with st.expander("📚 이 처방의 근거 보기"):
        st.markdown("**근거 논문**")
        seen = set()
        for d in p["진단"]:
            for e in d["근거_논문"]:
                key = e["evidence_id"]
                if key in seen:
                    continue
                seen.add(key)
                st.write(f"- ({e.get('authors','')}, {e.get('year','')}) {e['title']}")
        st.markdown("**이식 후보 제도**")
        any_cand = False
        for t in p["이식후보"]:
            for c in t["candidates"]:
                any_cand = True
                st.write(f"- [{c['reference_priority']}] {c['name']}  ·  {', '.join(c.get('regions', []))}")
        if not any_cand:
            st.write("- (이식 가능한 외부 제도 없음 = 구조적 빈틈 → 서울 전역 확대 등 신규 구현 필요)")
