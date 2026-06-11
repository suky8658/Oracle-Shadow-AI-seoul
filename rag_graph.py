# -*- coding: utf-8 -*-
"""
rag_graph.py - RAG 지식그래프 시각화 (payload → 계층형 Plotly 그래프)
================================================================
shadow_rag_llm.get_prescription(gu, generate_text=False)["사실"](=payload P)만으로
노드·엣지를 구성해, ADB 그래프 추론 결과를 '그림'으로 보여준다. (추가 ADB 호출 0)

컨셉 정합 (Map·전이예측과 동일한 시각 언어):
  · 의미 있는 좌표축 - 랜덤 force 가 아니라 좌→우 5계층(①~④ 파이프라인과 1:1)
        [Q유형] → [현행제도] → [낙인·의존 충돌] → [need 방향] → [이식후보]
  · 색 = 기존 토큰 재사용 - Q=Q_COLORS, 충돌=down(빨강), need=amber, 이식=출처위계색
  · ★ 핵심 추론(교집합) - Q가 '민감'한 낙인 ∩ 제도가 '자극'하는 낙인 = 진짜 충돌.
        payload 진단의 stimulates_stigma/deepens_dependency 는 이미 교집합 결과라,
        그 코드들은 Q→요소(민감/취약) 엣지와 제도→요소(자극/심화) 엣지를 모두 받는다.

반환:
  build_rag_graph(P)  -> (nodes: dict[id->node], edges: list[(u,v,rel)], q)
  render_rag_graph(P) -> (go.Figure, node_index: dict[id->상세dict])
"""
import plotly.graph_objects as go

from theme import COLOR, Q_COLORS, KO_FONT, fig_base

# 출처 위계 색 (shadow_rag.py SRC_COLOR 와 동일 토큰)
SRC_COLOR = {"해외": COLOR["blue"], "국내타지역": COLOR["amber"], "서울타자치구": "#5E6B7B"}

# 낙인/의존 코드 → 사람이 읽는 이름 (ontology stigma_elements/dependency_elements 고정 id)
STIG_NAME = {
    "STG_WelfareLabel": "복지라벨", "STG_FaceExposure": "대면노출",
    "STG_SelfAction": "본인행동", "STG_PublicExposure": "공개노출",
    "STG_RecipientIdentity": "수혜정체성",
}
DEP_NAME = {
    "DEP_NoContact": "비대면완결", "DEP_NoOuting": "외출불요",
    "DEP_NoRelation": "관계미형성", "DEP_LowEfficacy": "자기효능감저하",
    "DEP_FragileSupply": "중단취약",
}
Q_STATE = {"Q1": "의존↑·회피↑", "Q2": "의존↑·회피↓", "Q3": "의존↓·회피↓", "Q4": "의존↓·회피↑"}

# 계층 컬럼 (x = col * COL_GAP)
COL_GAP = 2.4
ROW_GAP = 1.0


def _name(code):
    return STIG_NAME.get(code) or DEP_NAME.get(code) or code


def _short(s, n=11):
    s = s or "-"
    return s if len(s) <= n else s[: n - 1] + "…"


# ════════════════════════════════════════════════════════════════════
# 1. payload → 노드·엣지 (순수 함수 · ADB/네트워크 의존 없음)
# ════════════════════════════════════════════════════════════════════
def build_rag_graph(P):
    """payload(P)에서 5계층 노드와 엣지를 만든다. (테스트는 dict 목업으로 가능)"""
    rp = P.get("지역_프로파일", {}) or {}
    q = str(rp.get("quadrant", "-"))
    진단 = P.get("진단", []) or []
    이식 = {t.get("need_id"): t.get("candidates", []) for t in P.get("이식후보", []) or []}

    nodes = {}
    edges = set()

    def add(nid, **kw):
        if nid not in nodes:
            nodes[nid] = {"id": nid, **kw}
        return nodes[nid]

    # ── col0 · Q유형 (앵커) ───────────────────────────────────────────
    sens = rp.get("quadrant_민감낙인", []) or []
    vuln = rp.get("quadrant_취약의존", []) or []
    add("Q", type="Q", col=0, label=q, color=Q_COLORS.get(q, "#95a5a6"),
        hover=f"<b>{q} · 진단 유형</b><br>{Q_STATE.get(q, '')}"
              f"<br>민감 낙인 {len(sens)} · 취약 의존 {len(vuln)}",
        detail=("이 자치구의 SHADOW Map 진단 유형(앵커). 아래 '충돌'은 이 유형이 "
                f"<b>민감</b>한 요소({', '.join(sens) or '없음'})와 "
                f"<b>취약</b>한 요소({', '.join(vuln) or '없음'})에서 발생합니다."))

    for d in 진단:
        한계 = d.get("한계", "")
        need = d.get("need", {}) or {}
        nid = need.get("id")
        if not nid:
            continue
        need_node = f"need:{nid}"
        add(need_node, type="need", col=3, label=need.get("name", ""),
            hover=f"<b>{need.get('name','')}</b> · 필요 방향<br>{need.get('def','')}",
            detail=f"필요한 처방 방향 - {need.get('def','')}")

        for p in d.get("원인_제도", []) or []:
            pid = f"prog:{p.get('program_id')}"
            nm = p.get("name", "-")
            add(pid, type="prog", col=1, label=_short(nm),
                hover=f"<b>{nm}</b><br>현행 제도",
                detail=f"현행 제도 <b>{nm}</b> - 이 제도가 아래 요소를 자극/귀속시켜 한계를 만듭니다.")

            stg = p.get("stimulates_stigma", []) or []
            dep = p.get("deepens_dependency", []) or []
            if 한계 in ("낙인충돌", "의존심화") and (stg or dep):
                for c in stg:
                    cid = f"conf:{c}"
                    add(cid, type="conf", col=2, label=_name(c),
                        hover=f"<b>{_name(c)}</b> · 낙인 충돌<br>Q가 민감 ∩ 제도가 자극 = 교집합",
                        detail=(f"<b>{q} →[민감]→ {_name(c)} ←[자극]← 현행 제도</b><br>"
                                f"두 경로의 교집합 = 진짜 충돌 → '{need.get('name','')}' 필요"))
                    edges.add((pid, cid, "자극"))
                    edges.add(("Q", cid, "민감"))
                    edges.add((cid, need_node, "필요"))
                for c in dep:
                    cid = f"conf:{c}"
                    add(cid, type="conf", col=2, label=_name(c),
                        hover=f"<b>{_name(c)}</b> · 의존 충돌<br>Q가 취약 ∩ 제도가 심화 = 교집합",
                        detail=(f"<b>{q} →[취약]→ {_name(c)} ←[심화]← 현행 제도</b><br>"
                                f"두 경로의 교집합 = 진짜 충돌 → '{need.get('name','')}' 필요"))
                    edges.add((pid, cid, "심화"))
                    edges.add(("Q", cid, "취약"))
                    edges.add((cid, need_node, "필요"))
            else:
                # 구조적 한계(자치구귀속·중복배제): 낙인/의존 교집합이 아니라 제도 구조 자체
                lid = f"limit:{한계}"
                add(lid, type="limit", col=2, label=한계,
                    hover=f"<b>{한계}</b> · 구조적 한계<br>제도 설계 자체의 빈틈",
                    detail=f"구조적 한계 <b>{한계}</b> - 낙인 교집합이 아니라 제도 설계의 빈틈입니다.")
                edges.add((pid, lid, 한계))
                edges.add((lid, need_node, "필요"))

        # ── col4 · 이식후보 (없으면 '구조적 빈틈' 노드) ─────────────────
        cands = 이식.get(nid, []) or []
        if cands:
            for c in cands:
                cid = f"cand:{c.get('program_id')}"
                pr = c.get("reference_priority", "서울타자치구")
                regions = ", ".join(c.get("regions", []) or [])
                add(cid, type="cand", col=4, label=_short(c.get("name", "-")),
                    color=SRC_COLOR.get(pr, "#5E6B7B"), priority=pr,
                    hover=f"<b>{c.get('name','')}</b><br>{pr} · {regions}",
                    detail=(f"<b>{c.get('name','')}</b> · {pr} ({regions})<br>"
                            f"{c.get('rationale','') or ''}"))
                edges.add((need_node, cid, "이식"))
        else:
            gid = f"gap:{nid}"
            add(gid, type="gap", col=4, label="구조적 빈틈",
                hover="이식 가능한 외부 제도 없음<br>→ 서울 전역 확대 등 신규 구현 필요",
                detail=("이식 가능한 외부 제도가 <b>없는 구조적 빈틈</b> - 기존 사례 이식이 아니라 "
                        "서울 전역 확대·보편접근 같은 신규 제도 구현이 필요합니다."))
            edges.add((need_node, gid, "빈틈"))

    return nodes, list(edges), q


# ════════════════════════════════════════════════════════════════════
# 2. 계층 레이아웃 (컬럼 = x, 컬럼 내 균등 분포 = y)
# ════════════════════════════════════════════════════════════════════
def layered_layout(nodes):
    cols = {}
    for n in nodes.values():
        cols.setdefault(n["col"], []).append(n)
    pos = {}
    for col, lst in cols.items():
        k = len(lst)
        for i, n in enumerate(lst):
            y = ((k - 1) / 2.0 - i) * ROW_GAP
            pos[n["id"]] = (col * COL_GAP, y)
    return pos


# ════════════════════════════════════════════════════════════════════
# 3. Plotly 렌더 (Map·예측과 같은 fig_base 톤)
# ════════════════════════════════════════════════════════════════════
# 타입별 스타일 (color 가 노드에 박혀 있으면 그걸, 없으면 여기 기본색)
_STYLE = {
    "Q":     dict(name="Q유형",        color=None,           size=40, symbol="circle",      tpos="bottom center"),
    "prog":  dict(name="현행 제도",     color=COLOR["blue"],  size=20, symbol="circle",      tpos="bottom center"),
    "conf":  dict(name="낙인·의존 충돌", color=COLOR["down"],  size=26, symbol="circle",      tpos="bottom center"),
    "limit": dict(name="구조적 한계",   color="#B0B8C1",      size=22, symbol="square",      tpos="bottom center"),
    "need":  dict(name="필요 방향",     color=COLOR["amber"], size=26, symbol="diamond",     tpos="bottom center"),
    "cand":  dict(name="이식후보",      color=None,           size=20, symbol="square",      tpos="bottom center"),
    "gap":   dict(name="구조적 빈틈",   color="#C7CDD4",      size=20, symbol="circle-open",  tpos="bottom center"),
}
# 컬럼 헤더 (① ~ ④ 파이프라인 라벨)
_COL_HEAD = {0: "진단 유형", 1: "① 현행 제도", 2: "② 한계 ⭐", 3: "③ 필요 방향", 4: "④ 이식후보"}


def render_rag_graph(P):
    """payload → (Plotly Figure, node_index). node_index[id]=상세(클릭 패널용)."""
    nodes, edges, q = build_rag_graph(P)
    pos = layered_layout(nodes)

    fig = go.Figure()

    # ── 엣지 (옅은 회색 라인 1트레이스) ───────────────────────────────
    ex, ey = [], []
    for u, v, _ in edges:
        if u in pos and v in pos:
            ex += [pos[u][0], pos[v][0], None]
            ey += [pos[u][1], pos[v][1], None]
    fig.add_trace(go.Scatter(
        x=ex, y=ey, mode="lines", hoverinfo="skip", showlegend=False,
        line=dict(color="#D8DEE5", width=1.2),
    ))

    # ── 컬럼 헤더 주석 ────────────────────────────────────────────────
    ymax = max((y for _, y in pos.values()), default=0) + 1.1
    for col, head in _COL_HEAD.items():
        if any(n["col"] == col for n in nodes.values()):
            fig.add_annotation(x=col * COL_GAP, y=ymax, text=head, showarrow=False,
                               font=dict(color=COLOR["muted"], size=12, family=KO_FONT),
                               yanchor="bottom")

    # ── 노드 (타입별 트레이스 · 이식후보는 출처위계 색배열) ────────────
    order = ["Q", "prog", "limit", "conf", "need", "cand", "gap"]
    node_index = {}
    for t in order:
        items = [n for n in nodes.values() if n["type"] == t]
        if not items:
            continue
        sty = _STYLE[t]
        xs = [pos[n["id"]][0] for n in items]
        ys = [pos[n["id"]][1] for n in items]
        txt = [n["label"] for n in items]
        cd = [[n["id"], n.get("hover", n["label"])] for n in items]
        colors = [n.get("color") or sty["color"] for n in items]
        for n in items:
            node_index[n["id"]] = {"label": n["label"], "type": t,
                                   "detail": n.get("detail", ""), "color": n.get("color") or sty["color"]}
        # conf(교집합 충돌)·Q 는 흰 테두리 굵게 강조
        lw = 2.4 if t in ("conf", "Q") else 1.4
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text", name=sty["name"],
            text=txt, textposition=sty["tpos"],
            textfont=dict(size=10, family=KO_FONT, color=COLOR["ink"]),
            marker=dict(size=sty["size"], color=colors, symbol=sty["symbol"],
                        line=dict(color="white", width=lw)),
            customdata=cd,
            hovertemplate="%{customdata[1]}<extra></extra>",
        ))

    xmax = max((x for x, _ in pos.values()), default=COL_GAP * 4)
    ymin = min((y for _, y in pos.values()), default=0) - 1.0
    fig.update_layout(
        height=560, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=11, family=KO_FONT, color=COLOR["muted"])),
        xaxis=dict(visible=False, range=[-1.0, xmax + 1.0]),
        yaxis=dict(visible=False, range=[ymin, ymax + 0.6]),
        hoverlabel=dict(font=dict(family=KO_FONT)),
    )
    fig = fig_base(fig, height=560)
    # fig_base 가 축 그리드를 켜므로 다시 끈다 (그래프는 좌표축이 무의미) +
    # 상단 가로범례·컬럼헤더가 들어갈 여백 확보
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(t=64, b=12, l=12, r=12))
    return fig, node_index
