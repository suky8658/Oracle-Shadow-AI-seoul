# -*- coding: utf-8 -*-
"""
load_evidence_to_adb.py — 근거논문 evidence.json(71편)을 ADB에 적재.
한 번만 실행하면 됨. 기존 EVIDENCE 테이블이 있으면 비우고 다시 넣음(멱등).

테이블: EVIDENCE (DATA JSON)  ← ONTOLOGY/PROGRAMS와 동일한 'DATA JSON 컬럼' 패턴.
읽기:  SELECT DATA FROM EVIDENCE  (논문 1편 = 1행)

실행:  $env:PYTHONIOENCODING="utf-8"; python load_evidence_to_adb.py
"""
import os, sys, io, json
import oracledb
from dotenv import load_dotenv

if hasattr(sys.stdout, "buffer") and (getattr(sys.stdout, "encoding", "") or "").lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))
WALLET_DIR = os.path.join(HERE, "wallet")
EVIDENCE_PATH = os.path.join(HERE, "Data", "Prescription", "evidence.json")

conn = oracledb.connect(
    user=os.environ["DB_USER"], password=os.environ["DB_PASSWORD"],
    dsn=os.environ["DB_DSN"], config_dir=WALLET_DIR,
    wallet_location=WALLET_DIR, wallet_password=os.environ["WALLET_PASSWORD"],
)
cur = conn.cursor()

# 1) 테이블 준비 (없으면 생성)
try:
    cur.execute("CREATE TABLE EVIDENCE (DATA JSON)")
    print("EVIDENCE 테이블 생성.")
except oracledb.DatabaseError as e:
    if "ORA-00955" in str(e):   # 이미 존재
        print("EVIDENCE 테이블 이미 있음 → 비우고 다시 적재.")
        cur.execute("DELETE FROM EVIDENCE")
    else:
        raise

# 2) evidence.json 로드 + 적재
EV = json.load(open(EVIDENCE_PATH, encoding="utf-8"))
cur.setinputsizes(data=oracledb.DB_TYPE_JSON)
cur.executemany("INSERT INTO EVIDENCE (DATA) VALUES (:data)", [{"data": e} for e in EV])
conn.commit()

# 3) 검증
cur.execute("SELECT COUNT(*) FROM EVIDENCE")
print(f"적재 완료: {cur.fetchone()[0]}건 (원본 {len(EV)}건)")
cur.execute("SELECT JSON_VALUE(DATA,'$.evidence_id'), JSON_VALUE(DATA,'$.title') FROM EVIDENCE FETCH FIRST 3 ROWS ONLY")
print("샘플 3건:")
for r in cur.fetchall():
    print(f"   {r[0]} | {r[1]}")

cur.close()
conn.close()
print("끝. 이제 shadow_rag_llm.py가 ADB에서 논문을 읽습니다.")
