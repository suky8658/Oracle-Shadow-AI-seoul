# -*- coding: utf-8 -*-
"""
streamlit_chat_test.py — SHADOW 담당자 챗봇 작동 확인용 '미니' 대시보드. (챗봇 전용)

자치구 선택 → 그 자치구 분석 데이터를 근거로 질의응답(실시간 스트리밍, 환각0).
처방 미니(streamlit_test.py)와 분리된 별도 기능. 친구가 본 대시보드에
"챗봇 탭은 이렇게 붙이면 된다"는 작동 예시.

실행: streamlit run streamlit_chat_test.py   (.env 와 wallet/ 가 같은 폴더에 있어야 함)
"""
import streamlit as st
from shadow_chat import get_chat_context, stream_chat

SEOUL_GU = [
    "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구",
    "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구",
    "구로구", "금천구", "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구",
]

st.set_page_config(page_title="SHADOW 챗봇 (테스트)", page_icon="💬")
st.title("💬 SHADOW 챗봇 — 미니 테스트")
st.caption("선택한 자치구의 분석 데이터를 근거로 답합니다 (데이터에 없으면 솔직히 '없음', 무관한 질문은 거절)")

gu = st.selectbox("자치구 선택", SEOUL_GU, index=SEOUL_GU.index("노원구"))

# 자치구가 바뀌면 → 분석 데이터 새로 로드 + 대화 초기화
if st.session_state.get("loaded_gu") != gu:
    with st.spinner(f"{gu} 분석 데이터 불러오는 중..."):
        cr = get_chat_context(gu)
    if cr["에러"]:
        st.error(cr["에러"])
        st.stop()
    st.session_state.chat_context = cr["context"]
    st.session_state.loaded_gu = gu
    st.session_state.chat_history = []

# 지난 대화 다시 그리기
for m in st.session_state.get("chat_history", []):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 첫 화면 예시 질문 안내
if not st.session_state.get("chat_history"):
    st.info(
        f'예시 질문 — "{gu}가 왜 취약한가요?"  ·  '
        f'"회피점수를 분해하면 뭐가 큰 비중인가요?"  ·  '
        f'"고위험 행정동은 어디고 왜 위험한가요?"'
    )

q = st.chat_input(f"{gu}에 대해 물어보세요")
if q:
    st.session_state.chat_history.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    prior = st.session_state.chat_history[:-1]   # 직전 질문 제외한 이전 대화
    with st.chat_message("assistant"):
        ans = st.write_stream(stream_chat(st.session_state.chat_context, prior, q))
    st.session_state.chat_history.append({"role": "assistant", "content": ans})
