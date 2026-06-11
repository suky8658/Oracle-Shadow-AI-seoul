# -*- coding: utf-8 -*-
"""
shadow_rag_llm.py — SHADOW 처방 생성 (ADB 그래프 연동판).

팀원 초안(generate_prescription.py / build_prescription_input.py)은
로컬 JSON을 파이썬으로 계산했지만, 이 파일은 같은 계산을
★ Oracle ADB(AlonearADB)의 그래프/JSON 쿼리로 ★ 수행한다.
(= 'shadow_graph' 위에서 결정적으로 사실을 추출 → OpenAI는 글쓰기만)

흐름:
  자치구명 입력
    → [ADB] 진단(Q·프로파일) · 현행제도 · 한계(충돌) · need · 이식후보(그래프 추론)
    → 근거논문 매칭(ADB EVIDENCE 테이블)
    → payload 조립
    → [OpenAI gpt-4o] ①~⑤ 처방문 생성 (입력에 있는 제도·논문만 인용)

실행:
  $env:PYTHONIOENCODING="utf-8"
  python shadow_rag_llm.py 도봉구
  python shadow_rag_llm.py 노원구 gpt-4o-mini   # 모델 교체
"""
import os, sys, io, json
from collections import OrderedDict
from decimal import Decimal
import oracledb
from dotenv import load_dotenv


def _json_default(o):
    """ADB가 준 Decimal 등 비표준 타입을 JSON 직렬화."""
    if isinstance(o, Decimal):
        return float(o)
    raise TypeError(f"직렬화 불가 타입: {type(o).__name__}")

# 한글 콘솔 깨짐 방지
if hasattr(sys.stdout, "buffer") and (getattr(sys.stdout, "encoding", "") or "").lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))
WALLET_DIR = os.path.join(HERE, "wallet")

DEFAULT_MODEL = "gpt-4o"
OK_AGE = {"중장년", "전연령"}
OK_GENDER = {"무관", "남성"}
SEOUL_WIDE = {"서울전체", "서울시"}

# ── 프롬프트: 번호 템플릿이 아니라 '자연스러운 분석 글'로 출력 ──────
SYSTEM_PROMPT = """\
너는 '서울 5060 남성 1인가구 고립' 문제의 복지 처방을 작성하는 정책 분석가다.
분석은 엄밀하게 하되, 어조는 너무 딱딱하지 않게 — 읽는 사람을 배려하는 친절하고 정중한 말투로 쓴다.
너의 일은 새로운 사실을 만드는 것이 아니라, [입력]에 주어진 재료만으로
하나의 설득력 있는 정책 분석 글을 쓰는 것이다.

[사실·근거 규칙]
1. [입력]에 없는 제도·논문·통계·한계코드를 새로 만들지 마라. (지어내면 실패)
2. 한계·need·이식후보는 이미 그래프가 계산해 주어진다. 재판단하지 말고 그대로 서술하라.
3. 논문(근거_논문)은 '한계'와 '필요한 방향'을 설명할 때만 인용한다. 외부 제도 사례나 최종 제안에는 논문을 끌어쓰지 마라. (예외: 이식후보가 0건인 한계 — 규칙7)
4. 외부 사례를 들 때의 근거는 '다른 지역의 실제 제도'(이식후보)다. 논문이 아니라 제도 이름·지역을 댄다.
5. 논문 인용은 evidence_id가 아니라 (저자, 연도) 형태로 자연스럽게 녹여라.
6. 근거(논문·통계·실제사례)로만 설득하라. 과장·미사여구는 금지하되, 차갑지 않고 정중하며 따뜻한 어조는 유지하라.
7. 이식후보 유무에 따라 외부 사례와 제안을 다르게 쓴다:
   - 이식후보(candidates)가 있는 한계(낙인충돌→무낙인, 의존심화→관계복원): 그 '실제 제도'(지역·이름)를 reference_priority(해외>국내타지역>서울타자치구) 순으로 들고, 이 자치구(특히 고위험 행정동)로의 이식을 제안한다.
   - 이식후보가 0건인 구조적 한계(자치구귀속→이식가능, 중복배제→보편접근): 먼저 '이식 가능한 외부 제도가 없는 구조적 빈틈'임을 밝히고(제도 지어내기 금지), 근거_논문 중 외국 사례성 근거가 있으면 그 사례가 방향이 옳음을 보여준다고만 덧붙인 뒤(없으면 생략), 서울 전역 확대·중복수혜 배제 완화 같은 신규 제도 구현을 제안한다.
8. 전이예측 근거(SHAP·가중편차기여도)를 적극 활용하라. '회피_악화요인'(부호 +, 위험을 키우는)과 '회피_완충요인'(부호 −, 위험을 누르는)은 부호의 뜻을 정확히 반영해 서술한다(완충요인을 위험요인처럼 쓰지 마라). 진단·이식후보의 '우선처방=true'는 그 방향이 지금 실제로 악화 중이라는 신호다 — 모든 한계는 빠짐없이 다루되, 우선처방=true인 한계·need·이식후보를 ③·⑤에서 가장 앞세워 비중 있게 다룬다.

[출력 구조 — 다섯 단계 ①~⑤ (대시보드 섹션 구분용, 이 구조는 반드시 유지)]
각 단계는 동그라미 번호 + 짧은 라벨로 시작한다. 단, 단계 '안의' 글은 재료(제도·논문·사례)를 기계적으로 나열하지 말고 인과·맥락으로 엮어 한 편의 분석처럼 자연스럽게 써라.
① 현행: 이 자치구가 왜 위험한지를 전이예측 근거로 짚는다 — 회피점수와 함께 '회피_악화요인'(위험을 키우는 +요인)을 핵심 근거로 들고, '회피_완충요인'(위험을 누르는 −요인)이 있으면 무엇이 그나마 버팀목인지 짚는다. '고위험_행정동'은 실제 동 이름과 전이확률·주요악화요인을 들어 어디가 왜 위험한지 보인 뒤, 운영 중인 관련 현행 제도를 인정한다.
② 한계: 진단에 나온 한계를 빠짐없이 모두 다룬다(하나라도 누락 금지). 각 한계를 해당 논문 근거와 엮되, '제도 A가 있다, 제도 B가 있다'식 끊긴 나열이 아니라 맥락으로 자연스럽게 이어라.
③ 방향: 한계가 가리키는 need(필요한 방향)로 넘어간다.
④ 외부 레퍼런스: 이식후보가 있으면 그 실제 제도(지역·이름)를 reference_priority(해외>국내타지역>서울타자치구) 순으로, 없으면 '구조적 빈틈'을 밝히고 외국 사례성 근거로 방향만 뒷받침한다(규칙7).
⑤ 이식 제안: 이 자치구를 위한 구체적 처방으로 닫되, '고위험_행정동'을 동 이름·전이확률·주요악화요인과 함께 짚어 '어느 동에 무엇이 왜 필요한지'를 행정동 단위로 구체화한다. '우선처방=true'인 방향(지금 실제로 악화 중)의 제도를 가장 앞세워 제안한다.

[형식 · 가독성 — 출력이 대시보드에 그대로 렌더된다]
- 다섯 단계 ①~⑤는 각각 별도 문단으로 쓰고, 단계 사이에는 반드시 빈 줄을 한 줄 넣어 시각적으로 분리하라.
- 한 단계가 길면 그 안에서도 의미 단위로 문단을 나눠 빈 줄로 띄워라. (한 덩어리로 빽빽하게 쓰지 마라)
- 각 단계의 동그라미 번호 라벨(예: **① 현행**)은 굵게 처리하라.
- 각 단계에서 정말 중요한 핵심(특히 ⑤의 이식 제안 제도명·핵심 처방, 자치구명, 결정적 한계)은 파란색으로 강조한다. 다음 HTML 태그를 그대로 출력하라: <span style="color:#3182F6;font-weight:700">강조할 말</span>. 단계당 1~2개만, 정말 핵심에만 쓴다(남발 금지).
- 그 외 가벼운 강조는 **굵게**(마크다운)로만 한다.

[문장 톤 — '짜깁기'처럼 안 보이게]
- 각 단계 안에서 '제도 A가 있다. 또 제도 B가 있다'식으로 재료를 뚝뚝 끊어 나열하지 마라. 맥락과 인과로 이어 한 편의 글처럼 읽히게 하라.
- 매 문장을 '따라서/또한'으로 시작하거나 늘 '~마련돼야 한다'로 끝맺는 반복을 피하고, 문장 구조·맺음말을 다양하게 하라.
- ①의 첫 문장은 그 자치구의 가장 두드러진 특징에서 출발해, 자치구마다 도입이 서로 달라지게 하라. (복붙·천편일률 금지)
- 단, 문장을 다듬느라 진단된 한계나 이식후보를 빠뜨리지는 마라. 내용 누락보다 매끄럽지 못한 게 낫다 — 모든 한계를 다루는 것이 우선이다.
"""
USER_TEMPLATE = "아래는 ADB 그래프가 계산한 [입력]이다. 이 재료로 ①~⑤ 다섯 단계의 처방 분석 글을 작성하라. 단계 구분(①~⑤)은 유지하되, 각 단계 안의 문장은 재료를 짜깁기한 듯 끊지 말고 자연스럽게 엮어라.\n\n[입력]\n{payload}\n"


# ── ADB 접속 ─────────────────────────────────────────────────
def connect():
    return oracledb.connect(
        user=os.environ["DB_USER"], password=os.environ["DB_PASSWORD"],
        dsn=os.environ["DB_DSN"], config_dir=WALLET_DIR,
        wallet_location=WALLET_DIR, wallet_password=os.environ["WALLET_PASSWORD"],
    )


def _as_json(v):
    """ADB JSON 컬럼 → 파이썬 (dict/str/bytes/LOB 어떤 형태로 와도 처리)."""
    if v is None:
        return None
    if isinstance(v, (dict, list)):
        return v
    if hasattr(v, "read"):       # LOB
        v = v.read()
    if isinstance(v, bytes):
        v = v.decode("utf-8")
    return json.loads(v)


def _arr(v):
    j = _as_json(v)
    return j if isinstance(j, list) else []


# ── 사실 추출 (전부 ADB) ──────────────────────────────────────
def load_rules_and_profile(cur):
    cur.execute("SELECT DATA FROM ONTOLOGY")
    ont = _as_json(cur.fetchone()[0])
    cur.execute("SELECT DATA FROM SHADOW_PROFILE")
    profile = _as_json(cur.fetchone()[0])
    return ont, profile


def gu_programs(cur, gu):
    """선택 자치구에서 시행 중인(또는 서울전체) 제도 + 코드 배열까지 ADB에서."""
    cur.execute("""
        SELECT JSON_VALUE(DATA,'$.program_id'), JSON_VALUE(DATA,'$.name'),
               JSON_VALUE(DATA,'$.access_mode'), JSON_VALUE(DATA,'$.gender'),
               JSON_VALUE(DATA,'$.beneficiary'), JSON_VALUE(DATA,'$.region_scope'),
               JSON_QUERY(DATA,'$.regions'), JSON_QUERY(DATA,'$.target_age'),
               JSON_QUERY(DATA,'$.stimulates_stigma'), JSON_QUERY(DATA,'$.deepens_dependency'),
               JSON_QUERY(DATA,'$.fulfills_needs')
        FROM PROGRAMS
        WHERE JSON_VALUE(DATA,'$.region_scope') IN ('서울자치구','서울전체')
          AND ( JSON_EXISTS(DATA, '$.regions[*]?(@ == $g)' PASSING :g AS "g")
                OR JSON_EXISTS(DATA, '$.regions[*]?(@ == "서울전체")')
                OR JSON_VALUE(DATA,'$.region_scope') = '서울전체' )
    """, g=gu)
    progs = []
    for r in cur.fetchall():
        progs.append({
            "program_id": r[0], "name": r[1], "access_mode": r[2],
            "gender": r[3] or "무관", "beneficiary": r[4] or "당사자",
            "region_scope": r[5], "regions": _arr(r[6]), "target_age": _arr(r[7]),
            "stimulates_stigma": _arr(r[8]), "deepens_dependency": _arr(r[9]),
            "fulfills_needs": _arr(r[10]),
        })
    return progs


def transplant(cur, need_id, gu, q):
    """★ 그래프 추론: fulfills∋need AND 그 자치구 미시행 AND 선택Q 무충돌(낙인/의존).
       FULFILLS_E·IN_REGION_E·STIMULATES_E·SENSITIVE_E·DEEPENS_E·VULNERABLE_E 조인."""
    cur.execute("""
        SELECT pv.program_id, pv.name, pv.region_scope
        FROM PROGRAM_V pv
        JOIN FULFILLS_E f ON f.program_id = pv.program_id AND f.code = :need
        WHERE pv.region_scope <> '서울전체'
          AND NOT EXISTS (SELECT 1 FROM IN_REGION_E ir
                          WHERE ir.program_id = pv.program_id AND ir.gu = :gu)
          AND NOT EXISTS (SELECT 1 FROM STIMULATES_E st JOIN SENSITIVE_E se
                          ON se.code = st.code AND se.quadrant = :q
                          WHERE st.program_id = pv.program_id)
          AND NOT EXISTS (SELECT 1 FROM DEEPENS_E dp JOIN VULNERABLE_E ve
                          ON ve.code = dp.code AND ve.quadrant = :q
                          WHERE dp.program_id = pv.program_id)
    """, need=need_id, gu=gu, q=q)
    return [{"program_id": r[0], "name": r[1], "region_scope": r[2]} for r in cur.fetchall()]


def fetch_meta(cur, ids):
    """이식후보 program_id들의 regions/rationale/evidence_ids/status를 ADB에서 일괄."""
    if not ids:
        return {}
    binds = {f"p{i}": pid for i, pid in enumerate(ids)}
    inlist = ",".join(f":{k}" for k in binds)
    cur.execute(f"""
        SELECT JSON_VALUE(DATA,'$.program_id'), JSON_QUERY(DATA,'$.regions'),
               JSON_VALUE(DATA,'$.rationale'), JSON_QUERY(DATA,'$.evidence_ids'),
               JSON_VALUE(DATA,'$.status')
        FROM PROGRAMS WHERE JSON_VALUE(DATA,'$.program_id') IN ({inlist})
    """, **binds)
    out = {}
    for r in cur.fetchall():
        out[r[0]] = {"regions": _arr(r[1]), "rationale": r[2] or "",
                     "evidence_ids": _arr(r[3]), "status": r[4] or ""}
    return out


# ── 보조 (논문 매칭 = 로컬, 팀원 로직 그대로) ───────────────────
def load_evidence(cur):
    """근거논문 = ADB EVIDENCE 테이블에서 (적재: load_evidence_to_adb.py)."""
    cur.execute("SELECT DATA FROM EVIDENCE")
    return [_as_json(r[0]) for r in cur.fetchall()]


def evidence_for(EV, trigger_codes, need_id, top=4):
    scored = []
    for e in EV:
        sids = set(e.get("supports_ids", []))
        score = 2 * len(sids & set(trigger_codes)) + (1 if need_id in sids else 0)
        if score > 0:
            scored.append((score, e))
    scored.sort(key=lambda x: (-x[0], x[1].get("scope", ""), -int(x[1].get("year", 0) or 0)))
    return [OrderedDict([
        ("evidence_id", e["evidence_id"]), ("title", e["title"]),
        ("authors", e.get("authors", "")), ("year", e.get("year", "")),
        ("scope", e.get("scope", "")), ("key_finding", e.get("key_finding", "")),
        ("_relevance", s),
    ]) for s, e in scored[:top]]


def ref_priority(region_scope):
    return {"국가": "해외", "국내지역": "국내타지역"}.get(region_scope, "서울타자치구")


def topk(d, k=3):
    if not isinstance(d, dict):
        return []
    return [name for name, _ in sorted(d.items(), key=lambda kv: -(kv[1] or 0))[:k]]


# ── 전이예측(SHAP·가중편차기여도) → 처방 방향 매핑 ──────────────────
# 위험을 '키우는'(부호 +) 요인이 어떤 need(처방 방향)를 요구하는지에 대한 도메인 휴리스틱.
# 이 표만 고치면 ①(진단·이식후보 우선순위) 기준이 바뀐다.
RISK_TO_NEED = {
    "복지불신":   "NEED_NoStigma",        # 복지서비스 불신·거리감 → 무낙인 전달방식
    "외로움부정": "NEED_RelationRestore",  # 고립인데 외로움 부정 → 관계복원
    "네트워크축소": "NEED_RelationRestore",  # 도움받을 사람 수 부족(PQ47A) → 관계복원
    "도움부재":   "NEED_RelationRestore",  # 도움받을 사람 없음(PQ47, 네트워크축소와 동개념) → 관계복원
}


def _signed(d):
    """기여도 dict → (악화요인[부호 +], 완충요인[부호 -]). 각 항목 (이름, 반올림값)."""
    if not isinstance(d, dict):
        return [], []
    items = [(k, round(float(v), 4)) for k, v in d.items() if v is not None]
    악화 = sorted([kv for kv in items if kv[1] > 0], key=lambda kv: -kv[1])
    완충 = sorted([kv for kv in items if kv[1] < 0], key=lambda kv: kv[1])
    return 악화, 완충


def _dong_risk_factors(shap, k=2):
    """행정동 SHAP에서 '위험↑'(shap_value>0) 상위 k개 요인 라벨.
    (기존엔 abs_shap 1순위를 그대로 '주요위험요인'으로 썼는데, 그게 보호요인이어도
     위험요인으로 오표기되던 문제를 바로잡는다.)"""
    pos = [s for s in (shap or []) if (s.get("shap_value") or 0) > 0]
    pos.sort(key=lambda s: -(s.get("shap_value") or 0))
    return [s.get("feature_label", "") for s in pos[:k]]


def region_profile(profile, gu, q, SENS, VULN, STG_NAME, DEP_NAME):
    pr = profile.get(gu, {})
    회피 = pr.get("회피분해", {}).get("가중편차기여도") or pr.get("회피분해", {}).get("원본", {})
    의존 = pr.get("의존분해", {}).get("가중편차기여도") or pr.get("의존분해", {}).get("원본", {})
    회피_악화, 회피_완충 = _signed(회피)   # 가중편차기여도는 부호가 있어 악화/완충 구분 가능

    동 = [d for d in pr.get("행정동", []) if d.get("위험등급") in ("최고위험", "고위험")]
    동.sort(key=lambda d: -(d.get("전이확률") or 0))
    위험동 = [OrderedDict([
        ("행정동", d.get("행정동")), ("위험등급", d.get("위험등급")),
        ("전이확률", d.get("전이확률")),
        ("주요악화요인", _dong_risk_factors(d.get("shap"))),
    ]) for d in 동[:4]]

    # 지금 '악화 중'(기여도 부호 +)인 회피 요인이 요구하는 처방 방향 → 진단·이식후보 우선순위 신호
    우선방향 = []
    for k, _ in 회피_악화:
        nd = RISK_TO_NEED.get(k)
        if nd and nd not in 우선방향:
            우선방향.append(nd)

    return OrderedDict([
        ("quadrant", q),
        ("quadrant_민감낙인", [STG_NAME.get(c, c) for c in SENS.get(q, [])]),
        ("quadrant_취약의존", [DEP_NAME.get(c, c) for c in VULN.get(q, [])]),
        ("회피점수", pr.get("avoidance")), ("의존점수", pr.get("dependency")),
        ("회피_주동인", topk(회피)), ("의존_주동인", topk(의존)),  # 하위호환(대시보드 소비)
        ("회피_악화요인", [f"{k} (+{v})" for k, v in 회피_악화]),
        ("회피_완충요인", [f"{k} ({v})" for k, v in 회피_완충]),
        ("고위험_행정동", 위험동),
        ("우선처방방향", 우선방향),
    ])


# ── payload 조립 ─────────────────────────────────────────────
def build_payload(cur, gu, EV, q_override=None):
    ont, profile = load_rules_and_profile(cur)
    SENS = ont["quadrant_sensitive_stigma"]
    VULN = ont["quadrant_vulnerable_dependency"]
    LIMIT_TO_NEED = ont["limit_to_needs"]
    STG_NAME = {e["id"]: e["name"] for e in ont["stigma_elements"]}
    DEP_NAME = {e["id"]: e["name"] for e in ont["dependency_elements"]}

    q = q_override or profile.get(gu, {}).get("quadrant")
    if not q:
        raise ValueError(f"'{gu}'의 quadrant를 SHADOW_PROFILE에서 못 찾음. 자치구명을 확인하세요.")

    progs = gu_programs(cur, gu)

    # ★ 전이예측 프로파일 먼저 — '우선처방방향'(지금 악화 중인 회피요인이 요구하는 need)을
    #   진단·이식후보 우선순위에 흘려보내기 위해 여기서 계산한다.
    rp_profile = region_profile(profile, gu, q, SENS, VULN, STG_NAME, DEP_NAME)
    active_needs = set(rp_profile.get("우선처방방향", []))

    # ① 현행 (coverage_rule: 연령·성별·당사자 통과)
    현행 = [OrderedDict([("program_id", p["program_id"]), ("name", p["name"]),
                        ("fulfills_needs", p["fulfills_needs"])])
           for p in progs
           if (set(p["target_age"]) & OK_AGE) and (p["gender"] in OK_GENDER)
           and (p["beneficiary"] == "당사자")]

    # ② 한계 버킷 → ③ need
    buckets = OrderedDict()
    def add(한계, p, codes, extra_key=None):
        b = buckets.setdefault(한계, {"원인_제도": [], "codes": set()})
        b["codes"] |= set(codes)
        entry = OrderedDict([("program_id", p["program_id"]), ("name", p["name"])])
        if extra_key == "stg":
            entry["stimulates_stigma"] = [c for c in p["stimulates_stigma"] if c in codes]
        elif extra_key == "dep":
            entry["deepens_dependency"] = [c for c in p["deepens_dependency"] if c in codes]
        elif extra_key == "access":
            entry["access_mode"] = p["access_mode"]
        elif extra_key == "region":
            entry["regions"] = p["regions"]
        b["원인_제도"].append(entry)

    for p in progs:
        stg = set(p["stimulates_stigma"]) & set(SENS.get(q, []))
        dep = set(p["deepens_dependency"]) & set(VULN.get(q, []))
        if stg: add("낙인충돌", p, stg, "stg")
        if dep: add("의존심화", p, dep, "dep")
        if p["region_scope"] != "서울전체" and not (set(p["regions"]) & SEOUL_WIDE):
            add("자치구귀속", p, ["NEED_Transferable"], "region")
        if p["access_mode"] == "중복배제":
            add("중복배제", p, ["NEED_UniversalAccess"], "access")

    진단 = []
    for 한계, b in buckets.items():
        need = LIMIT_TO_NEED[한계]
        trigger = {c for c in b["codes"] if c.startswith(("STG_", "DEP_"))}
        진단.append(OrderedDict([
            ("한계", 한계),
            ("원인_제도", b["원인_제도"][:8]),
            ("원인_제도_총수", len(b["원인_제도"])),
            ("need", OrderedDict([("id", need["need_id"]), ("name", need["name"]), ("def", need["def"])])),
            ("우선처방", need["need_id"] in active_needs),  # ★ 전이예측상 지금 악화 중인 방향
            ("근거_논문", evidence_for(EV, trigger, need["need_id"])),
        ]))
    # 모든 한계는 유지하되, 지금 악화 중인 방향(우선처방=True)을 앞으로 — 강조 순서만 조정(안정 정렬)
    진단.sort(key=lambda d: 0 if d["우선처방"] else 1)

    # ④ 이식후보 (★ ADB 그래프 추론)
    need_ids = []
    for d in 진단:
        if d["need"]["id"] not in need_ids:
            need_ids.append(d["need"]["id"])

    raw = {nid: transplant(cur, nid, gu, q) for nid in need_ids}
    all_ids = [c["program_id"] for lst in raw.values() for c in lst]
    meta = fetch_meta(cur, list(dict.fromkeys(all_ids)))

    이식후보 = []
    for nid in need_ids:
        cands = []
        for c in raw[nid]:
            m = meta.get(c["program_id"], {})
            cands.append(OrderedDict([
                ("program_id", c["program_id"]), ("name", c["name"]),
                ("regions", m.get("regions", [])), ("region_scope", c["region_scope"]),
                ("reference_priority", ref_priority(c["region_scope"])),
                ("rationale", m.get("rationale", "")), ("status", m.get("status", "")),
                ("evidence_ids", m.get("evidence_ids", [])),
            ]))
        order = {"해외": 0, "국내타지역": 1, "서울타자치구": 2}
        by = {"해외": [], "국내타지역": [], "서울타자치구": []}
        for c in cands:
            by[c["reference_priority"]].append(c)
        mixed = []
        for tier in ("해외", "국내타지역", "서울타자치구"):
            mixed.extend(by[tier][:3])
        mixed.sort(key=lambda c: order.get(c["reference_priority"], 9))
        이식후보.append(OrderedDict([("need_id", nid), ("우선처방", nid in active_needs), ("candidates", mixed)]))

    return OrderedDict([
        ("선택_자치구", gu), ("대상", "서울 5060 남성 1인가구"),
        ("지역_프로파일", rp_profile),
        ("현행_제도", 현행), ("진단", 진단), ("이식후보", 이식후보),
        ("_note", "ADB 그래프(shadow_graph) 규칙으로 계산됨. LLM은 재판단 말고 ①~⑤ 서술만. 입력에 없는 제도·논문 인용 금지."),
    ])


# ── 생성 ─────────────────────────────────────────────────────
def generate(payload, model):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model, temperature=0.4,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                payload=json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))},
        ],
    )
    return resp.choices[0].message.content, resp.usage


def stream_prescription(payload, model=DEFAULT_MODEL):
    """처방문을 '실시간 조각(delta)'으로 흘려보내는 제너레이터.
    대시보드에서 st.write_stream(stream_prescription(payload))로 쓰면 타이핑되듯 표시된다.
    (payload는 get_prescription(gu, generate_text=False)["사실"]을 그대로 넘기면 됨)
    """
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    stream = client.chat.completions.create(
        model=model, temperature=0.4, stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                payload=json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))},
        ],
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ── 진입점 (★ 대시보드·외부에서 부르는 함수) ───────────────────
def get_prescription(gu, model=DEFAULT_MODEL, generate_text=True):
    """자치구명 하나로 처방을 만들어 dict로 돌려준다. 대시보드는 이 함수만 부르면 됨.

    인자:
      gu            : 자치구명 (예: "노원구")
      model         : 생성 모델 (기본 gpt-4o)
      generate_text : False면 사실(payload)만 뽑고 OpenAI 호출은 생략 (빠르게 진단만 볼 때)

    반환 dict:
      {
        "자치구": "노원구",
        "처방문": "① 현행 ... ⑤ 이식 제안 ...",   # generate_text=False면 None
        "사실":   { 지역_프로파일·현행_제도·진단·이식후보·... },  # 근거(논문·이식제도) 다 들어있음
        "토큰":   {"in":..., "out":..., "합":...},   # 실패 시 None
        "에러":   None 또는 "사람이 읽을 오류 메시지"   # ★ 예외를 던지지 않고 여기에 담음
      }
    에러가 나도 예외를 던지지 않으므로, 대시보드는 result["에러"]만 확인하면 화면이 안 깨진다.
    """
    result = {"자치구": gu, "처방문": None, "사실": None, "토큰": None, "에러": None}
    conn = None
    try:
        conn = connect()
        cur = conn.cursor()
        EV = load_evidence(cur)
        result["사실"] = build_payload(cur, gu, EV)
    except ValueError as e:          # 자치구명 오타 등 사용자 입력 문제
        result["에러"] = str(e)
        return result
    except Exception as e:           # ADB 접속·쿼리 등 시스템 문제
        result["에러"] = f"사실 추출 중 오류: {e}"
        return result
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    if not generate_text:
        return result

    try:
        text, usage = generate(result["사실"], model)
        result["처방문"] = text
        result["토큰"] = {"in": usage.prompt_tokens, "out": usage.completion_tokens, "합": usage.total_tokens}
    except Exception as e:           # OpenAI 호출 실패 (키·결제·네트워크 등)
        result["에러"] = f"처방 생성(OpenAI) 오류: {e}"
    return result


def main():
    gu = sys.argv[1] if len(sys.argv) > 1 else "도봉구"
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL

    print(f"[{gu}] ADB에서 사실 추출 + 처방 생성 중...")
    r = get_prescription(gu, model)
    if r["에러"]:
        print(f"❌ {r['에러']}")
        return
    payload = r["사실"]

    rp = payload["지역_프로파일"]
    print(f"  Q={rp['quadrant']} | 회피_악화요인={rp.get('회피_악화요인')} | 우선처방방향={rp.get('우선처방방향')}")
    print(f"  고위험동={[(d['행정동'], d.get('전이확률'), d.get('주요악화요인')) for d in rp.get('고위험_행정동', [])]}")
    print(f"  ① 현행 제도: {len(payload['현행_제도'])}건")
    for d in payload["진단"]:
        star = "★" if d.get("우선처방") else " "
        print(f"  {star}② {d['한계']}: 원인 {d['원인_제도_총수']}건 → ③ {d['need']['name']}({d['need']['id']}) · 근거논문 {len(d['근거_논문'])}편")
    for t in payload["이식후보"]:
        pr = {}
        for c in t["candidates"]:
            pr[c["reference_priority"]] = pr.get(c["reference_priority"], 0) + 1
        star = "★" if t.get("우선처방") else " "
        print(f"  {star}④ {t['need_id']} 이식후보: {len(t['candidates'])}건 {pr}")

    # 입력 저장 (베이스라인/디버깅용)
    in_path = os.path.join(HERE, f"처방입력_{gu}.json")
    json.dump(payload, open(in_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=_json_default)

    text = r["처방문"]
    out_path = os.path.join(HERE, f"처방문_{gu}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# 처방문 — {gu} (모델: {model}, ADB 그래프 연동)\n\n{text}\n")
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)
    tok = r["토큰"]
    if tok:
        print(f"\n토큰: in {tok['in']} / out {tok['out']} / 합 {tok['합']}")
    print(f"저장: {in_path}\n      {out_path}")


if __name__ == "__main__":
    main()
