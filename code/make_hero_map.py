# -*- coding: utf-8 -*-
"""
Hero 랜딩용 서울 행정동 코로플레스 PNG 생성기
- seoul_dong.geojson (427개 행정동) 폴리곤을 matplotlib로 직접 렌더
- shadow_prescriptions.csv 위험등급으로 색칠 (베이스맵 없는 미니멀 버전)
출력: Outputs/shadow_ai/hero_seoul_map.png  (투명 배경, 2x 고해상도)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from matplotlib.collections import PatchCollection

ROOT = Path(__file__).resolve().parent.parent
GEO  = ROOT / "Data" / "seoul_dong.geojson"
CSV  = ROOT / "Outputs" / "shadow_ai" / "shadow_prescriptions.csv"
OUT  = ROOT / "Outputs" / "shadow_ai" / "hero_seoul_map.png"

GRADE_COLOR = {
    "최고위험": "#E5484D",
    "고위험":   "#F0883E",
    "중위험":   "#F5C843",
    "저위험":   "#35C26F",
}
MISS_COLOR = "#E7E9ED"   # 매칭 안 되는 행정동(군부대/비거주 등)

# ── 위험등급 매핑 ──────────────────────────────────────────────
df = pd.read_csv(CSV, encoding="utf-8-sig")
grade_map = {(r["자치구"], r["행정동"]): r["위험등급"] for _, r in df.iterrows()}

# ── geojson 로드 ──────────────────────────────────────────────
geo = json.load(open(GEO, encoding="utf-8"))

patches, colors = [], []
xs, ys = [], []
matched = 0

for feat in geo["features"]:
    pr   = feat["properties"]
    sgg  = pr.get("sggnm", "")
    dong = pr.get("adm_nm", "").split()[-1]
    grade = grade_map.get((sgg, dong))
    if grade:
        matched += 1
    color = GRADE_COLOR.get(grade, MISS_COLOR)

    g = feat["geometry"]
    if g is None:
        continue
    polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
    for poly in polys:
        ring = np.asarray(poly[0])           # 외곽 링
        patches.append(MplPoly(ring, closed=True))
        colors.append(color)
        xs.extend(ring[:, 0]); ys.extend(ring[:, 1])

print(f"행정동 매칭: {matched}/{len(geo['features'])}")

# ── 렌더 ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=240)
pc = PatchCollection(patches, facecolor=colors, edgecolor="white", linewidth=0.35)
ax.add_collection(pc)

ax.set_xlim(min(xs), max(xs))
ax.set_ylim(min(ys), max(ys))
# 위경도 왜곡 보정 (서울 위도 ≈ 37.55)
ax.set_aspect(1 / np.cos(np.radians(37.55)))
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, transparent=True, bbox_inches="tight", pad_inches=0.02)
print(f"저장 완료: {OUT}")
