# -*- coding: utf-8 -*-
"""
SHADOW 처방 지식그래프 — '그래프+LLM' 검색 엔진
================================================
Q유형 · 제도 · 낙인요소를 노드로, 그 사이의 실제 관계를 엣지로 표현한 작은 지식그래프.
RAG의 '검색(Retrieve)' 단계를, 벡터 유사도가 아니라 그래프 1홉 룩업으로 수행한다.
  → 데이터가 '구조화된 정확한 관계 매칭'이라서, 그 모양 그대로 그래프로 표현 + 탐색하면
    임베딩·벡터DB 없이도 정확하고 설명 가능한 검색이 된다. (ontology_schema.png 의 설계 그대로)

엣지 3종 (모두 KB 데이터에서 그대로 파생 — 새 사실 추가 없음):
  권장 (Q유형 → 제도)    : KB[Q]['제도후보'] 그대로
  자극 (제도 → 낙인요소)  : KB 제도의 rubric 에서 True 인 것
  민감 (Q유형 → 낙인요소) : 그 Q의 심리기저(KB['심리기저'])가 특히 반응하는 낙인 요소

★ 핵심 추론 = 1홉 교집합 (ontology_schema.png 의 "★ 핵심 추론" 그대로 구현):
   ( Q유형이 '민감'한 낙인요소 ) ∩ ( 제도가 '자극'하는 낙인요소 )  =  진짜 충돌
   교집합 밖의 자극은 '부수효과' — 충돌은 아니지만 참고할 만한 신호로 별도 표시.
"""
import networkx as nx

from kb.prescriptions import KB, RUBRIC_DIMS

# Q유형별 '민감' 낙인요소 — 회피(Avoidance)가 높은 Q1·Q4는 여러 요소에 민감,
# 회피가 낮은 Q2·Q3는 둔감하다 (KB 심리기저 서술과 동일한 근거).
#   Q1: 의존·회피 모두↑ → 비대면조차 라벨·대면·행동 요구에 민감
#   Q2: 회피↓ → 안내만으로 진입 가능, 특별히 민감한 요소 없음
#   Q3: 회피↓하지만 '본인이 나서야 하는' 것에는 다소 보수적
#   Q4: 회피 매우↑ → 대면·공개·정체성 규정에 가장 민감
SENSITIVE = {
    "Q1": {"복지라벨", "대면노출", "본인행동"},
    "Q2": set(),
    "Q3": {"본인행동"},
    "Q4": {"대면노출", "공개노출", "수혜정체성"},
}


def build_graph():
    """KB 데이터 그대로 지식그래프를 구성한다 (사실 추가 없음 — 관계만 노드·엣지로 표현)."""
    G = nx.DiGraph()
    for q in KB:
        G.add_node(q, type="Q유형", name=KB[q]["name"])
    for s, desc in RUBRIC_DIMS.items():
        G.add_node(s, type="낙인요소", desc=desc)

    for q_type, qd in KB.items():
        for s in SENSITIVE.get(q_type, set()):
            G.add_edge(q_type, s, rel="민감")
        for inst in qd["제도후보"]:
            name = inst["name"]
            if name not in G:
                G.add_node(name, type="제도",
                           주최=inst["주최"], 내용=inst["내용"], url=inst["url"], 시행=inst["시행"])
            G.add_edge(q_type, name, rel="권장")
            for dim, triggered in inst.get("rubric", {}).items():
                if triggered:
                    G.add_edge(name, dim, rel="자극")
    return G


def _out(G, node, rel):
    return {v for _, v, d in G.out_edges(node, data=True) if d["rel"] == rel}


def retrieve(G, q_type):
    """1홉 그래프 룩업 — 'Q유형' 노드에서 엣지를 따라가며 제도·충돌을 가져온다.
    벡터 유사도 계산이 전혀 없다 — 그래프의 엣지를 그대로 따라가는 정확한 매�칭이라서다.

    반환: {"sensitive": [Q유형이 민감한 낙인요소들],
           "hits": [{제도 정보..., conflicts(진짜 충돌), side_effects(부수효과), fit}]}
    """
    sensitive = _out(G, q_type, "민감")
    hits = []
    for inst in _out(G, q_type, "권장"):
        stimulated = _out(G, inst, "자극")
        conflicts = sorted(stimulated & sensitive)      # 민감 ∩ 자극 = 진짜 충돌
        side_effects = sorted(stimulated - sensitive)   # 자극은 하나, 이 Q유형엔 비민감 영역
        n = G.nodes[inst]
        hits.append({
            "name": inst, "주최": n["주최"], "내용": n["내용"], "url": n["url"], "시행": n["시행"],
            "conflicts": conflicts, "side_effects": side_effects,
            "fit": "적합" if not conflicts else ("부분 적합" if len(conflicts) == 1 else "주의 필요"),
        })
    hits.sort(key=lambda h: len(h["conflicts"]))
    return {"sensitive": sorted(sensitive), "hits": hits}


def trace(q_type, inst_name):
    """탐색 경로를 사람이 읽는 문장으로 — '왜 이 결과가 나왔는지' 그래프 경로 그대로 설명."""
    return (f"{q_type} →[권장]→ {inst_name} →[자극]→ 낙인요소,  "
            f"{q_type} →[민감]→ 낙인요소  ⇒  두 경로의 교집합이 '충돌'")
