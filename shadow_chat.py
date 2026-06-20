# -*- coding: utf-8 -*-
"""
shadow_chat.py — SHADOW 담당자 질의응답 '챗봇' 엔진. (처방 엔진과 분리된 별도 기능)

처방(shadow_rag_llm.py)과 별개지만, 공용 부품(ADB 접속·그래프 사실추출)은
shadow_rag_llm 에서 그대로 가져다 쓴다 — 코드 중복 없음.

흐름: 자치구 선택 → [ADB] 풍부한 분석 데이터 추출(처방에 안 쓴 축분해·SHAP까지)
      → [OpenAI] 그 데이터 근거로만 답(환각0, 무관 질문은 거절).
대시보드 사용 예시: streamlit_chat_test.py

함수:
  get_chat_context(gu)                  → {자치구, context, 에러}   (자치구 선택 시 1회)
  stream_chat(context, history, question) → 답을 실시간 조각으로 yield (st.write_stream용)
"""
import os
import json
from collections import OrderedDict

# 공용 부품은 처방 엔진에서 재사용 (중복 구현 안 함)
from shadow_rag_llm import connect, build_payload, load_evidence, _as_json, _json_default, DEFAULT_MODEL

CHAT_SYSTEM = """\
너는 'SHADOW-AI' 분석 결과를 지자체 담당자에게 설명하는 정책 데이터 분석가다.
아래 [자치구 분석 데이터]에 있는 사실만 근거로 답하라.

[답변 원칙]
1. 데이터에 있는 내용(회피/의존 축분해 수치·진단된 한계·need·이식후보 제도 등)은 구체적 근거(수치)를 들어 쉽게 설명한다. 회피·의존 요인을 설명할 땐 기여도 수치를 한두 개 자연스럽게 곁들여라(예: '도움부재가 0.38로 가장 크고, 복지불신 0.17이 뒤를 잇는다'). 단 수치는 문장 속에 자연스럽게 녹이고, '이 요인은 …에 0.xx의 기여를 하고 있습니다' 같은 동일한 틀을 항목마다 기계적으로 반복하지는 마라. 수치는 반드시 [자치구 분석 데이터]에 있는 값만 그대로 쓰고(소수 둘째 자리까지 반올림), 데이터에 없거나 헷갈리는 값은 절대 지어내지 말고 그때는 수치 없이 '가장 크다 / 그다음' 같은 순서로만 말하라.
2. 특히 고위험 행정동의 'SHAP 위험요인'과 '전이확률(전이예측 결과)'을 적극 활용해, "왜 이 동이 위험한지"를 SHAP 근거와 전이확률 수치로 구체적으로 설명하라.
3. 데이터에 정확한 사실/수치가 없는 내용(예산·법령·실행계획 등)은 지어내지 마라. 다만 "없습니다"로만 끝내지 말고, [참고_논문풀_전체](71편)에서 질문과 관련된 흐름·경향을 찾아 "정확한 데이터는 없지만, ○○(저자, 연도)에 따르면 이런 경향이 있어 참고하시면 좋습니다"처럼 도움이 되게 덧붙여라. 관련된 논문이 정말 없을 때만 솔직히 없다고 한다. (논문 제목·저자·내용을 지어내는 것은 절대 금지 — 반드시 논문풀에 실재하는 것만 인용.)
4. SHADOW 분석·5060 남성 1인가구 고립·복지 정책과 무관한 질문(일반 잡담·코딩·날씨·계산 등)에는 답을 시도하지 말고, "이 챗봇은 선택한 자치구의 SHADOW 분석에 관한 질문만 답합니다"라고 정중히 안내한다.
5. "[전국대비_순위_및_비교]"에 25개 자치구 점수·선택구 순위·서울 평균이 있다. "서울에서 몇 위냐", "평균보다 높냐/심하냐", "A구와 B구 중 어디가 더 ~하냐" 같은 질문에 이 데이터를 적극 활용해 순위·평균 대비로 답하라. 단 선택 자치구 외 다른 구는 '점수·분면'만 알 뿐 상세(축분해·제도·행정동)는 모르니, 다른 구의 상세를 물으면 "그 자치구를 선택하시면 자세히 볼 수 있습니다"라고 안내한다.
6. 담당자가 이해하기 쉽게, 그러나 정확하게 답한다.

[형식·말투 — 반드시 지킬 것]
- 별표(**)나 마크다운 강조 기호를 쓰지 마라. 강조가 필요하면 기호 대신 표현으로 풀어 써라. (별표가 화면에 그대로 노출돼 보기 흉하다)
- '1. 2. 3.' 번호나 불릿(-)으로 항목을 뚝뚝 끊어 나열하지 마라. 회피·의존 요인을 설명할 땐 수치를 근거로 삼되, '제일 큰 동인은 ~이고(수치), 그다음이 ~다' 처럼 한 편의 분석 글로 인과·맥락을 이어 써라.
- 'AI가 쓴 듯한' 기계적 맺음말("추가적인 질문이 있으시면 언제든지…", "도움이 되었길 바랍니다", "궁금한 점 있으면 편하게 물어보세요" 등)을 붙이지 마라. 답은 내용으로 자연스럽게 끝낸다.
- 매 문장을 같은 구조로 반복하지 말고, 문장 길이와 맺음말을 다양하게 한다.

[자치구 분석 데이터]
{context}
"""


def build_chat_context(cur, gu):
    """자치구의 풍부한 사실(진단·처방 + 프로파일 원본)을 문자열로 모은다.
    처방 payload(진단·이식후보)에 더해, 처방엔 안 쓰인 SHADOW_PROFILE 원본
    (회피/의존 축분해 수치, 행정동별 SHAP·전이확률 등)까지 통째로 컨텍스트에 넣는다."""
    EV = load_evidence(cur)
    payload = build_payload(cur, gu, EV)
    cur.execute("SELECT DATA FROM SHADOW_PROFILE")
    profile_all = _as_json(cur.fetchone()[0])
    raw = profile_all.get(gu, {})

    # 컨텍스트 토큰 절약 — 전체 행정동(수십 개) 대신 축분해 + 고위험 행정동만 남김
    # (SHAP·전이확률이 필요한 건 어차피 고위험 동이라 정보 손실 없음)
    동 = raw.get("행정동", []) if isinstance(raw, dict) else []
    slim = OrderedDict()
    for k in ("quadrant", "avoidance", "dependency"):
        if isinstance(raw, dict) and k in raw:
            slim[k] = raw[k]
    # 회피/의존 축분해는 '가중편차기여도'(부호 있는 기여도)만 노출한다.
    # '원본'(설문 원점수)까지 같이 주면 LLM이 두 스케일을 섞어 잘못된 수치를 말함.
    for k, label in (("회피분해", "회피_요인별_기여도"), ("의존분해", "의존_요인별_기여도")):
        blk = raw.get(k) if isinstance(raw, dict) else None
        if isinstance(blk, dict):
            slim[label] = blk.get("가중편차기여도") or blk.get("원본") or blk
    주요 = [d for d in 동 if d.get("위험등급") in ("최고위험", "고위험")]
    if not 주요:   # 고위험 동이 없는 저위험 자치구도 전이확률 상위 5개는 제공
        주요 = sorted([d for d in 동 if d.get("전이확률") is not None],
                      key=lambda d: -float(d["전이확률"]))[:5]
    slim["주요_행정동(고위험_또는_전이확률상위)"] = 주요
    slim["행정동_총수"] = len(동)
    raw = slim

    # ★ 전체 25개 자치구 점수·순위·평균 (자치구 비교·상대맥락 질문용)
    rows = []
    for g, v in profile_all.items():
        if not isinstance(v, dict) or "quadrant" not in v:
            continue
        rows.append(OrderedDict([
            ("자치구", g), ("quadrant", v.get("quadrant")),
            ("회피점수", v.get("avoidance")), ("의존점수", v.get("dependency")),
        ]))

    def rank_and_avg(key):
        vals = [(r["자치구"], float(r[key])) for r in rows if r.get(key) is not None]
        ordered = sorted(vals, key=lambda x: -x[1])           # 높은 순
        ranks = {name: i + 1 for i, (name, _) in enumerate(ordered)}
        avg = sum(v for _, v in vals) / len(vals) if vals else None
        return ranks, len(vals), avg

    a_rank, a_n, a_avg = rank_and_avg("회피점수")
    d_rank, d_n, d_avg = rank_and_avg("의존점수")
    비교 = OrderedDict([
        ("선택구_회피_순위", f"{a_rank.get(gu)}/{a_n}위 (1위=회피 가장 심함)"),
        ("선택구_의존_순위", f"{d_rank.get(gu)}/{d_n}위 (1위=의존 가장 심함)"),
        ("서울_회피_평균", round(a_avg, 1) if a_avg is not None else None),
        ("서울_의존_평균", round(d_avg, 1) if d_avg is not None else None),
        ("전체_25자치구_점수표", rows),
    ])

    # 근거논문 71편 전체 (데이터에 없는 질문을 '논문 흐름'으로 보강하기 위한 풀)
    논문풀 = [OrderedDict([
        ("저자", e.get("authors", "")), ("연도", e.get("year", "")),
        ("제목", e.get("title", "")), ("핵심발견", e.get("key_finding", "")),
        ("범위", e.get("scope", "")),
    ]) for e in EV]

    ctx = OrderedDict([
        ("선택_자치구", gu),
        ("전국대비_순위_및_비교", 비교),
        ("진단_및_처방_사실", payload),
        ("프로파일_원본_회피의존축분해_행정동SHAP", raw),
        ("참고_논문풀_전체", 논문풀),
    ])
    return json.dumps(ctx, ensure_ascii=False, indent=2, default=_json_default)


def get_chat_context(gu):
    """대시보드 진입점 — 자치구 선택 시 한 번 불러 챗봇 컨텍스트 확보.
    반환: {자치구, context(문자열), 에러(None or 메시지)}. 예외 안 던짐(대시보드 안 깨지게)."""
    result = {"자치구": gu, "context": None, "에러": None}
    conn = None
    try:
        conn = connect()
        cur = conn.cursor()
        result["context"] = build_chat_context(cur, gu)
    except ValueError as e:                 # 자치구명 오류 등
        result["에러"] = str(e)
        return result
    except Exception as e:                   # ADB 접속·쿼리 오류
        result["에러"] = f"데이터 조회 오류: {e}"
        return result
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    return result


def stream_chat(context, history, question, model=DEFAULT_MODEL):
    """담당자 질문 답을 실시간 조각으로 yield (st.write_stream용).
    context = get_chat_context(gu)["context"], history = 이전 대화 [{role,content},...]."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages = [{"role": "system", "content": CHAT_SYSTEM.format(context=context)}]
    messages += history
    messages.append({"role": "user", "content": question})
    stream = client.chat.completions.create(
        model=model, temperature=0.3, stream=True, messages=messages)
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
