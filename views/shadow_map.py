"""
🗺️ SHADOW Map — 현재 진단 맵 + 💊 RAG 처방 (한 페이지, 탭 2개)
================================================================
탭1 [진단 맵] : Dependency × Avoidance 두 축 사분면 (자치구 Q1~Q4 진단)
탭2 [RAG 처방]: 자치구 선택 → Q유형·두 지수 → 제도 후보 → 낙인-적합성 진단 → 처방카드

★ 설계 원칙(B안): 처방(RAG)은 '현재 진단(SHADOW Map)'에만 기반한다.
   → Q유형 + Dependency + Avoidance 만 사용. (전이확률·SHAP = SHADOW AI 값은 쓰지 않음)
   → 미래 전이예측(SHADOW AI)은 완전히 별도 섹션.
※ 기존 산출물(shadow_index.csv, 편의의 역설/dependency_index.csv)을 '읽기'만 한다.
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from kb.prescriptions import KB, Q_COLORS, RAG_EXAMPLES  # noqa: E402
from kb.graph import build_graph, retrieve as graph_retrieve  # noqa: E402


@st.cache_resource
def get_kb_graph():
    """Q유형·제도·낙인요소 지식그래프 (1회 구성, 세션 내 재사용)."""
    return build_graph()

KO_FONT = "Malgun Gothic, Apple SD Gothic Neo, sans-serif"
Q_LABEL = {
    "Q1": "Q1 · 최고위험 (의존↑·회피↑)",
    "Q2": "Q2 · 정서취약 (의존↑·회피↓)",
    "Q3": "Q3 · 상대안전 (의존↓·회피↓)",
    "Q4": "Q4 · 자기해결 (의존↓·회피↑)",
}


@st.cache_data
def load_shadow():
    p = ROOT / "Outputs" / "shadow_index.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_dep_dong():
    """행정동별 Dependency (편의의 역설). 보조 디테일용 — SHADOW Map 산출물."""
    p = ROOT / "Outputs" / "편의의 역설" / "dependency_index.csv"
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None


shadow = load_shadow()
dep_dong = load_dep_dong()

st.title("🗺️ SHADOW Map")
st.caption("Dependency(편의 의존) × Avoidance(복지 회피)로 진단하고, 그 자리에서 맞춤 처방까지")

tab_map, tab_rag = st.tabs(["🗺️ 진단 맵", "💊 RAG 처방"])

# ════════════════════════════════════════════════════════════════════════
# 탭 1 — 진단 맵 (Dependency × Avoidance 사분면)
# ════════════════════════════════════════════════════════════════════════
with tab_map:
    if shadow is None:
        st.warning("shadow_index.csv 가 없어요.")
    else:
        sdf = shadow.dropna(subset=["Dependency", "Avoidance", "Quadrant"]).copy()

        def split_point(low_qs, high_qs, col):
            lo = sdf[sdf["Quadrant"].isin(low_qs)][col]
            hi = sdf[sdf["Quadrant"].isin(high_qs)][col]
            if len(lo) and len(hi):
                return (lo.max() + hi.min()) / 2
            return sdf[col].median()

        dep_split = split_point(["Q3", "Q4"], ["Q1", "Q2"], "Dependency")
        avo_split = split_point(["Q2", "Q3"], ["Q1", "Q4"], "Avoidance")

        qcounts = sdf["Quadrant"].value_counts()
        cc = st.columns(4)
        for col, q in zip(cc, ["Q1", "Q2", "Q3", "Q4"]):
            n = int(qcounts.get(q, 0))
            col.markdown(
                f"<div style='border-top:4px solid {Q_COLORS[q]};border-radius:10px;padding:10px;"
                f"background:#fff;box-shadow:0 1px 6px rgba(0,0,0,.06);text-align:center'>"
                f"<div style='font-size:1.6rem;font-weight:800;color:{Q_COLORS[q]}'>{n}</div>"
                f"<div style='font-size:.74rem;color:#666'>{Q_LABEL[q]}</div></div>",
                unsafe_allow_html=True,
            )
        st.markdown("")

        col_map, col_side = st.columns([3, 1.1])
        with col_map:
            fig = go.Figure()
            xr = [sdf["Dependency"].min() - 5, sdf["Dependency"].max() + 5]
            yr = [sdf["Avoidance"].min() - 5, sdf["Avoidance"].max() + 5]
            for x0, x1, y0, y1, q in [
                (dep_split, xr[1], avo_split, yr[1], "Q1"),
                (dep_split, xr[1], yr[0], avo_split, "Q2"),
                (xr[0], dep_split, yr[0], avo_split, "Q3"),
                (xr[0], dep_split, avo_split, yr[1], "Q4"),
            ]:
                fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                              fillcolor=Q_COLORS[q], opacity=0.06, line_width=0, layer="below")
            fig.add_vline(x=dep_split, line_dash="dot", line_color="#aaa")
            fig.add_hline(y=avo_split, line_dash="dot", line_color="#aaa")
            for x, y, txt, q, ax, ay in [
                (xr[1], yr[1], "Q1 최고위험", "Q1", "right", "top"),
                (xr[1], yr[0], "Q2 정서취약", "Q2", "right", "bottom"),
                (xr[0], yr[0], "Q3 상대안전", "Q3", "left", "bottom"),
                (xr[0], yr[1], "Q4 자기해결", "Q4", "left", "top"),
            ]:
                fig.add_annotation(x=x, y=y, text=txt, showarrow=False,
                                   font=dict(color=Q_COLORS[q], size=12, family=KO_FONT),
                                   xanchor=ax, yanchor=ay, opacity=0.7)
            for q in ["Q1", "Q2", "Q3", "Q4"]:
                sub = sdf[sdf["Quadrant"] == q]
                if sub.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=sub["Dependency"], y=sub["Avoidance"],
                    mode="markers+text", name=Q_LABEL[q],
                    text=sub["자치구"], textposition="top center",
                    textfont=dict(size=10, family=KO_FONT),
                    marker=dict(size=16, color=Q_COLORS[q], opacity=0.85,
                                line=dict(color="white", width=1.5)),
                    customdata=sub[["자치구", "Quadrant"]],
                    hovertemplate="<b>%{customdata[0]}</b><br>Dependency %{x:.1f}<br>"
                                  "Avoidance %{y:.1f}<br>%{customdata[1]}<extra></extra>",
                ))
            fig.update_layout(
                height=560, paper_bgcolor="white", plot_bgcolor="#fafafa",
                margin={"t": 10, "b": 40, "l": 10, "r": 10},
                xaxis=dict(title="← Dependency (편의 의존) →", range=xr, gridcolor="#f0f0f0"),
                yaxis=dict(title="← Avoidance (복지 회피) →", range=yr, gridcolor="#f0f0f0"),
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                            font=dict(family=KO_FONT, size=10)),
                font=dict(family=KO_FONT),
            )
            st.plotly_chart(fig, width="stretch")
            st.caption("점선 = 사분면 경계 · 색 = 진단된 Q유형 · 위치 = 두 지수 좌표")

        with col_side:
            st.markdown("#### 🔍 자치구 진단")
            gu_m = st.selectbox("자치구", sorted(sdf["자치구"].unique()), key="diag_gu")
            r = sdf[sdf["자치구"] == gu_m].iloc[0]
            q = str(r["Quadrant"])
            qcol = Q_COLORS.get(q, "#95a5a6")
            st.markdown(
                f"**{gu_m}** <span style='background:{qcol};color:#fff;padding:2px 10px;"
                f"border-radius:12px;font-size:.8rem;font-weight:700'>{q}</span>",
                unsafe_allow_html=True,
            )
            d1, d2 = st.columns(2)
            d1.metric("Dependency", f"{r['Dependency']:.1f}")
            d2.metric("Avoidance", f"{r['Avoidance']:.1f}")
            if q in KB:
                st.markdown("**심리기저**"); st.caption(KB[q]["심리기저"])
                st.markdown("**처방 방향**"); st.caption(KB[q]["처방방향"])
            st.info("💊 구체 제도·적합성 진단 → 위 **'RAG 처방'** 탭")

        with st.expander("📋 자치구별 진단 전체"):
            show = sdf[["자치구", "Dependency", "Avoidance", "Quadrant"]].copy()
            show = show.sort_values(["Quadrant", "Avoidance"], ascending=[True, False]).reset_index(drop=True)
            st.dataframe(show.round(1), width="stretch", height=400)

# ════════════════════════════════════════════════════════════════════════
# 탭 2 — RAG 처방 (★B안: 현재 진단 = Q유형 + 두 지수 기반. AI값 사용 안 함★)
# ════════════════════════════════════════════════════════════════════════
with tab_rag:
    if shadow is None:
        st.warning("shadow_index.csv 가 필요해요.")
    else:
        st.caption("자치구 선택 → Q유형·두 지수 진단 → 제도 후보 → 낙인-적합성 진단 → 근거 기반 처방")
        st.markdown(
            "<div style='background:#eef6ff;border-radius:6px;padding:6px 12px;font-size:.82rem;color:#555'>"
            "ℹ️ 이 처방은 <b>SHADOW Map(현재 진단)</b>에만 기반합니다 — Q유형 + Dependency + Avoidance. "
            "미래 전이예측(SHADOW AI)과는 분리됩니다.</div>", unsafe_allow_html=True,
        )
        st.markdown("")

        sdf = shadow.dropna(subset=["Dependency", "Avoidance", "Quadrant"]).copy()
        gu = st.selectbox("자치구", sorted(sdf["자치구"].unique()), key="rag_gu")
        r = sdf[sdf["자치구"] == gu].iloc[0]
        q_type = str(r["Quadrant"])
        dep_v, avo_v = float(r["Dependency"]), float(r["Avoidance"])

        if q_type not in KB:
            st.error(f"Q유형({q_type}) 처방 지식베이스가 없어요.")
        else:
            kb = KB[q_type]
            qcol = Q_COLORS.get(q_type, "#95a5a6")

            st.markdown(
                f"## 📍 {gu} "
                f"<span style='background:{qcol};color:#fff;padding:3px 14px;border-radius:14px;"
                f"font-size:1rem;font-weight:700'>{kb['name']}</span>",
                unsafe_allow_html=True,
            )
            m1, m2 = st.columns(2)
            m1.metric("Dependency (편의 의존)", f"{dep_v:.1f}")
            m2.metric("Avoidance (복지 회피)", f"{avo_v:.1f}")

            st.markdown(
                f"<div style='background:#f8f9fa;border-left:4px solid {qcol};border-radius:0 8px 8px 0;"
                f"padding:12px 16px;margin:8px 0;line-height:1.7'><b>이 지역의 심리기저</b> · "
                f"{kb['심리기저']}</div>", unsafe_allow_html=True,
            )

            # 보조 디테일 — 자치구 내 행정동별 Dependency 분포 (SHADOW Map 산출물)
            if dep_dong is not None:
                sub_dd = dep_dong[dep_dong["자치구"] == gu].sort_values("Dependency", ascending=False)
                if not sub_dd.empty:
                    with st.expander(f"🏘️ {gu} 행정동별 편의 의존도 (Dependency) — {len(sub_dd)}개 동"):
                        st.dataframe(sub_dd[["행정동", "Dependency"]].round(1).reset_index(drop=True),
                                     width="stretch", height=260)

            st.divider()
            RETR_BADGE = ("<span style='background:#e8f8f0;color:#1e8449;padding:2px 10px;border-radius:10px;"
                          "font-size:.78rem;font-weight:700'>✅ 그래프 검색 · 1홉 룩업</span> "
                          "<span style='color:#999;font-size:.8rem'>— 임베딩·벡터DB 없이, "
                          f"'{q_type}' 노드에서 [권장]→[자극] 엣지를 그대로 따라가 [민감]과 겹치는 "
                          "충돌을 계산한 <b>실제 그래프 탐색 결과</b>입니다 (하드코딩·시뮬레이션 아님).</span>")
            GEN_BADGE = ("<span style='background:#fff3cd;color:#856404;padding:2px 10px;border-radius:10px;"
                         "font-size:.78rem;font-weight:700'>⚠️ LLM 생성 · 데모</span> "
                         "<span style='color:#999;font-size:.8rem'>— 위 그래프 검색·충돌분석 결과를 근거로 "
                         "LLM이 실시간 생성하는 자리. 지금은 그 결과와 일치하는 사전 작성 예시로 흐름만 시연.</span>")
            example = RAG_EXAMPLES.get(q_type, "").format(gu=gu, dep=f"{dep_v:.1f}", avo=f"{avo_v:.1f}")
            full_gen = f"**[처방 방향]**  {kb['처방방향']}\n\n---\n\n{example}"

            graph_result = graph_retrieve(get_kb_graph(), q_type)
            FIT_COLOR = {"적합": "#27ae60", "부분 적합": "#e67e22", "주의 필요": "#c0392b"}

            def show_retrieved(animate):
                st.markdown("### 🔎 1단계 · 검색 (Retrieve) — 그래프 1홉 룩업")
                st.markdown(RETR_BADGE, unsafe_allow_html=True)
                sens_list = graph_result["sensitive"]
                sens_text = (", ".join(f"`{s}`" for s in sens_list) if sens_list
                             else "해당 없음 — 회피축이 낮아 특별히 민감한 요소 없음")
                st.caption(f"🧭 탐색 경로: **{q_type}** →[민감]→ 낙인요소 = {sens_text}")
                for hit in graph_result["hits"]:
                    fc = FIT_COLOR[hit["fit"]]
                    with st.container(border=True):
                        st.markdown(
                            f"**{hit['name']}** · {hit['주최']} "
                            f"<span style='background:{fc};color:#fff;padding:1px 9px;border-radius:9px;"
                            f"font-size:.74rem;font-weight:700'>{hit['fit']}</span>",
                            unsafe_allow_html=True,
                        )
                        st.caption(f"{hit['내용']} ({hit['시행']})")
                        st.caption(f"📎 {hit['url']}")
                        if hit["conflicts"]:
                            st.markdown(
                                f"<span style='color:#c0392b;font-size:.8rem'>⚡ 충돌(민감 ∩ 자극): "
                                f"{' · '.join(hit['conflicts'])}</span>", unsafe_allow_html=True,
                            )
                        if hit["side_effects"]:
                            st.markdown(
                                f"<span style='color:#888;font-size:.8rem'>· 부수효과(자극하나 이 지역엔 비민감): "
                                f"{' · '.join(hit['side_effects'])}</span>", unsafe_allow_html=True,
                            )
                    if animate:
                        time.sleep(0.35)

            done_key = f"rag_done_{gu}"
            pressed = st.button("▶ RAG 처방 생성 시작", type="primary", key=f"rag_btn_{gu}")
            if pressed:
                st.session_state[done_key] = True

            if st.session_state.get(done_key):
                if pressed:
                    # ── 시연 애니메이션 ──
                    with st.spinner("🔎 지식그래프에서 1홉 탐색 중..."):
                        time.sleep(1.0)
                    show_retrieved(animate=True)
                    st.markdown("### 🧠 2단계 · 생성 (Generate)")
                    st.markdown(GEN_BADGE, unsafe_allow_html=True)
                    with st.spinner("🧠 그래프 검색 결과로 처방 생성 중..."):
                        time.sleep(0.9)

                    def streamer():
                        for para in full_gen.split("\n\n"):
                            yield para + "\n\n"
                            time.sleep(0.45)
                    with st.container(border=True):
                        st.write_stream(streamer)
                else:
                    # ── 정적 표시 (재실행 시) ──
                    show_retrieved(animate=False)
                    st.markdown("### 🧠 2단계 · 생성 (Generate)")
                    st.markdown(GEN_BADGE, unsafe_allow_html=True)
                    with st.container(border=True):
                        st.markdown(full_gen)
            else:
                st.info("▶ 위 버튼을 누르면 **검색(Retrieve) → 생성(Generate)** 과정이 순서대로 시연됩니다.")

            with st.expander("📎 출처 전체 · 한계 명시"):
                for i, inst in enumerate(kb["제도후보"], 1):
                    st.markdown(f"[출처{i}] {inst['name']} — {inst['url']}")
                st.caption("※ 본 처방은 SHADOW Map(현재 진단)의 Q유형 심리기저에 근거한 정책 설계 적합성 평가이며, "
                           "자치구 단위 신호입니다 (개인 단위 처방 아님).")

            with st.expander("✨ 실제 LLM 연결 (Claude API · 데모 예시를 실시간 생성으로 대체)"):
                if not os.environ.get("ANTHROPIC_API_KEY"):
                    st.info("환경변수 `ANTHROPIC_API_KEY` 설정 시, 위 **그래프 검색·충돌분석 결과**를 "
                            "컨텍스트로 넘겨 Claude가 처방글을 **실시간 생성**합니다 (데모 예시를 대체). "
                            "키 없이도 위 예시 출력은 표시됩니다.")
                else:
                    if st.button("처방문 실시간 생성"):
                        try:
                            import anthropic
                            client = anthropic.Anthropic()
                            docs = "\n".join(
                                f"- {h['name']} ({h['주최']}) · 그래프 적합도: {h['fit']} "
                                f"· 충돌(민감∩자극): {h['conflicts'] or '없음'} "
                                f"· 부수효과: {h['side_effects'] or '없음'}\n"
                                f"  내용: {h['내용']} (URL: {h['url']})"
                                for h in graph_result["hits"])
                            prompt = (
                                "당신은 복지 정책 상담사다. 아래는 지식그래프를 1홉 탐색해 찾은 제도 후보와, "
                                "그래프가 [권장]→[자극] 엣지와 [민감] 엣지의 교집합으로 계산한 낙인-적합성 "
                                f"분석(충돌·부수효과·적합도)이다 — 이 정보만 근거로(밖의 사실 생성 금지) "
                                f"{gu}(Q유형 {q_type}, Dependency {dep_v:.1f}, Avoidance {avo_v:.1f})에 대한 "
                                "처방을 쓰라. 그래프가 표시한 충돌·적합도를 그대로 반영해 설명하고, "
                                "마지막에 '자치구 단위 신호'임을 명시하라.\n\n"
                                f"[처방방향] {kb['처방방향']}\n[그래프 검색·분석 결과]\n{docs}"
                            )
                            resp = client.messages.create(model="claude-sonnet-4-6", max_tokens=900,
                                                          messages=[{"role": "user", "content": prompt}])
                            st.markdown(resp.content[0].text)
                        except Exception as e:
                            st.error(f"생성 실패: {e}")
