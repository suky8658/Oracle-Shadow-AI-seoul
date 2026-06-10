# -*- coding: utf-8 -*-
"""programs.json 재생성 — 과병합 버그 수정판.
규칙: <>안 이름 보존, 기수/회차/월/날짜만 제거 → 다른 활동은 분리, 같은 프로그램 회차는 병합.
content는 enrich에서 재작성할 임시 발췌. 분석필드는 빈 값.
"""
import csv, re, json, os
csv.field_size_limit(2147483647)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = [("서울시 1인가구 참여프로그램 현황 2025.csv", 2025),
       ("서울시 1인가구 참여프로그램 현황 2026.csv", 2026)]
C_TITLE, C_GUBUN, C_CAT, C_REGION = 1, 2, 3, 4
C_AGE, C_TARGET, C_METHOD, C_BODY, C_AGENCY = 9, 11, 16, 21, 23

HTML_TAG = re.compile(r"</?(p|br|div|span|img|b|strong|table|tr|td|em|h\d|ul|li|ol|a|font)[^>]*>", re.I)

def norm_key(t):
    t = HTML_TAG.sub("", t or "")
    t = re.sub(r"[\[\]\(\)<>]", "", t)          # 괄호문자만 제거(안 내용 보존)
    t = re.sub(r"20\d{2}\s*년?", "", t)
    t = re.sub(r"\d+\s*~\s*\d*\s*(회차|회|차수|차|월|주|일|기|분기)", "", t)  # 범위(1~3월 등)
    t = re.sub(r"(특별|정규)?\s*\d+\s*(기|기수)", "", t)   # 기수 제거
    t = re.sub(r"\d+\s*(회차|회|차수|차|월|주|분기)", "", t) # 회차/월 제거
    t = re.sub(r"\d+", "", t)
    t = re.sub(r"[~·,.\-_/!?’'\"*~]+", "", t)    # 잔여 구두점 제거
    t = re.sub(r"\s+", "", t)
    return t.strip()

def strip_html(s):
    s = re.sub(r"<[^>]+>", " ", s or "")
    s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return re.sub(r"\s+", " ", s).strip()

def age_map(ages):
    s = set()
    for a in ages:
        if "20대 이하" in a or "20~30대" in a: s.add("청년")
        if "40~50대" in a: s.add("중장년")
        if "60대 이상" in a: s.add("노인")
    if {"청년", "중장년", "노인"} <= s: return ["전연령"]
    return [x for x in ["청년", "중장년", "노인"] if x in s] or ["전연령"]

from collections import defaultdict, Counter
groups = defaultdict(list)
for fn, yr in CSV:
    with open(os.path.join(BASE, fn), encoding="cp949", errors="replace", newline="") as f:
        r = csv.reader(f); next(r)
        for row in r:
            if len(row) <= C_AGENCY: continue
            ag = (row[C_AGENCY] or "").strip()
            key = (ag, norm_key(row[C_TITLE]) or (row[C_TITLE] or "").strip())
            groups[key].append((row, yr))

progs = []
for (ag, k), items in groups.items():
    rows = [it[0] for it in items]
    years = {it[1] for it in items}
    titles = [(r[C_TITLE] or "").strip() for r in rows]
    name = Counter(titles).most_common(1)[0][0]            # 대표 제목(최빈)
    regions = sorted({(r[C_REGION] or "").strip() for r in rows if (r[C_REGION] or "").strip()})
    target_age = age_map([(r[C_AGE] or "") for r in rows])
    gubuns = sorted({(r[C_GUBUN] or "").strip() for r in rows if (r[C_GUBUN] or "").strip()})
    bodies = sorted({strip_html(r[C_BODY]) for r in rows if strip_html(r[C_BODY])}, key=len, reverse=True)
    content = (bodies[0] if bodies else "")[:300]
    status = "운영중" if 2026 in years else "종료"
    progs.append({
        "name": name, "content": content, "rationale": "",
        "regions": regions or ["서울전체"], "target_age": target_age,
        "access_mode": None, "stimulates_stigma": [], "deepens_dependency": [],
        "fulfills_needs": [], "status": status, "evidence_ids": [],
        "_meta": {"담당기관": ag, "구분": gubuns, "member_rows": len(rows)},
    })

# 정렬: 담당기관 → 이름 (결정적 ID 부여)
progs.sort(key=lambda p: (p["_meta"]["담당기관"], p["name"]))
for i, p in enumerate(progs, 1):
    p2 = {"program_id": f"P-{i:04d}"}; p2.update(p); progs[i-1] = p2

out = os.path.join(BASE, "programs.json")
if os.path.exists(out) and not os.path.exists(out + ".skeleton.bak"):
    import shutil; shutil.copyfile(out, out + ".skeleton.bak")
tmp = out + ".tmp"
json.dump(progs, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
os.replace(tmp, out)

# 통계
gub_count = Counter()
for p in progs:
    for g in p["_meta"]["구분"]: gub_count[g] += 1
print("새 programs.json 제도수:", len(progs))
print("구분 분포(중복포함):", dict(gub_count))
print("운영중/종료:", Counter(p["status"] for p in progs))
print("백업:", out + ".skeleton.bak (기존 1743판)")
