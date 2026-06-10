# -*- coding: utf-8 -*-
"""골격 programs.json의 각 제도에 대해 CSV 원문(내용+구조필드)을 재결합해 뽑는다.
분류는 사람(에이전트)이 직접 한다. 이 스크립트는 '원문 확보'만 담당.
사용: python extract_source.py START COUNT  (예: 0 20  → program_id 정렬 0..19)
출력: work/source_<START>_<END>.json
"""
import csv, sys, json, re, os

csv.field_size_limit(2147483647)
# programs.json이 있는 폴더를 자동 탐색: 같은 폴더 → 상위 폴더 → 현재 작업폴더 순. (평면/하위 폴더 둘 다 동작)
_here = os.path.dirname(os.path.abspath(__file__))
BASE = next((p for p in [_here, os.path.dirname(_here), os.getcwd()]
             if os.path.exists(os.path.join(p, "programs.json"))), _here)
CSV_FILES = [
    os.path.join(BASE, "서울시 1인가구 참여프로그램 현황 2025.csv"),
    os.path.join(BASE, "서울시 1인가구 참여프로그램 현황 2026.csv"),
]

# 컬럼 인덱스(헤더 위치 기준; 표시는 깨져도 순서는 고정)
C_TITLE, C_GUBUN, C_CAT, C_REGION = 1, 2, 3, 4
C_AGE, C_SEX, C_TARGET = 9, 10, 11
C_METHOD, C_LINK, C_ADDR = 16, 17, 19
C_BODY, C_AGENCY = 21, 23


_HTML_TAG = re.compile(r"</?(p|br|div|span|img|b|strong|table|tr|td|em|h\d|ul|li|ol|a|font)[^>]*>", re.I)

def norm_title(t: str) -> str:
    # rebuild_programs.py의 norm_key와 반드시 동일하게 유지할 것 (매칭 일치)
    t = _HTML_TAG.sub("", t or "")
    t = re.sub(r"[\[\]\(\)<>]", "", t)
    t = re.sub(r"20\d{2}\s*년?", "", t)
    t = re.sub(r"\d+\s*~\s*\d*\s*(회차|회|차수|차|월|주|일|기|분기)", "", t)
    t = re.sub(r"(특별|정규)?\s*\d+\s*(기|기수)", "", t)
    t = re.sub(r"\d+\s*(회차|회|차수|차|월|주|분기)", "", t)
    t = re.sub(r"\d+", "", t)
    t = re.sub(r"[~·,.\-_/!?’'\"*~]+", "", t)
    t = re.sub(r"\s+", "", t)
    return t.strip()


def strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# CSV 인덱스: (담당기관, 정규화제목) -> [원행 dict...]
index = {}
for fn in CSV_FILES:
    with open(fn, encoding="cp949", errors="replace", newline="") as f:
        r = csv.reader(f)
        next(r)  # header
        for row in r:
            if len(row) <= C_AGENCY:
                continue
            agency = (row[C_AGENCY] or "").strip()
            key = (agency, norm_title(row[C_TITLE]) or row[C_TITLE].strip())
            index.setdefault(key, []).append({
                "category": row[C_CAT].strip(),
                "gubun": row[C_GUBUN].strip(),
                "age": row[C_AGE].strip(),
                "sex": row[C_SEX].strip(),
                "target": row[C_TARGET].strip(),
                "method": row[C_METHOD].strip(),
                "addr": row[C_ADDR].strip(),
                "body": strip_html(row[C_BODY]),
            })

skeleton = json.load(open(os.path.join(BASE, "programs.json"), encoding="utf-8"))
skeleton.sort(key=lambda p: p["program_id"])

start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
count = int(sys.argv[2]) if len(sys.argv) > 2 else 20
sub = skeleton[start:start + count]

out = []
miss = 0
for p in sub:
    agency = (p.get("_meta") or {}).get("담당기관") or (p.get("_meta") or {}).get("담당기관") or ""
    # _meta key가 깨진 한글일 수 있으니 첫 문자열 값을 담당기관으로 시도
    if not agency:
        for v in (p.get("_meta") or {}).values():
            if isinstance(v, str):
                agency = v.strip(); break
    key = (agency.strip(), norm_title(p["name"]) or p["name"].strip())
    rows = index.get(key, [])
    # 본문 모음(중복 제거, 긴 것 우선)
    bodies, seen = [], set()
    for rw in sorted(rows, key=lambda x: -len(x["body"])):
        b = rw["body"]
        if b and b[:80] not in seen:
            seen.add(b[:80]); bodies.append(b)
    merged_body = "\n---\n".join(bodies)[:4000]
    methods = sorted({rw["method"] for rw in rows if rw["method"]})
    cats = sorted({rw["category"] for rw in rows if rw["category"]})
    ages = sorted({rw["age"] for rw in rows if rw["age"]})
    targets = sorted({rw["target"] for rw in rows if rw["target"]})
    if not rows:
        miss += 1
    out.append({
        "program_id": p["program_id"],
        "name": p["name"],
        "regions": p["regions"],
        "skeleton_target_age": p.get("target_age"),
        "skeleton_status": p.get("status"),
        "source_urls": p.get("source_urls"),
        "matched_rows": len(rows),
        "csv_category": cats,
        "csv_age": ages,
        "csv_target": targets,
        "csv_method": methods,
        "csv_body": merged_body,
    })

os.makedirs(os.path.join(BASE, "work"), exist_ok=True)
outfn = os.path.join(BASE, "work", f"source_{start}_{start+count}.json")
json.dump(out, open(outfn, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"wrote {outfn}  ({len(out)} programs, {miss} unmatched)")
