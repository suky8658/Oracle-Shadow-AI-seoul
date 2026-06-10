# -*- coding: utf-8 -*-
"""
test_db.py — ADB(AlonearADB) 접속 + 그래프 데이터 확인 단독 테스트.
실행: python test_db.py
성공하면 제도 2403건 + 도봉구 이식후보 134건이 찍혀야 함.
"""
import os
import oracledb
from dotenv import load_dotenv

# .env 읽기 (이 파일과 같은 폴더)
HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))

WALLET_DIR = os.path.join(HERE, "wallet")  # 압축 푼 wallet 폴더

print("ADB에 접속 시도 중...")
conn = oracledb.connect(
    user=os.environ["DB_USER"],
    password=os.environ["DB_PASSWORD"],
    dsn=os.environ["DB_DSN"],
    config_dir=WALLET_DIR,        # tnsnames.ora 위치
    wallet_location=WALLET_DIR,   # ewallet.pem 위치
    wallet_password=os.environ["WALLET_PASSWORD"],
)
print("접속 성공.\n")

cur = conn.cursor()

# 1) 제도 노드 건수 (2403 기대)
cur.execute("SELECT COUNT(*) FROM PROGRAM_V")
print("제도(PROGRAM_V) 건수:", cur.fetchone()[0])

# 2) region_scope 분포
cur.execute("SELECT region_scope, COUNT(*) FROM PROGRAM_V GROUP BY region_scope ORDER BY 2 DESC")
print("region_scope 분포:")
for scope, cnt in cur.fetchall():
    print(f"   {scope}: {cnt}")

# 3) 도봉구 이식후보 (rebuild_graph.sql 맨 아래 쿼리, 134 기대)
cur.execute("""
SELECT COUNT(DISTINCT p.program_id)
FROM PROGRAM_V p JOIN FULFILLS_E f ON f.program_id = p.program_id
WHERE f.code IN ('NEED_RelationRestore','NEED_NoStigma')
  AND p.region_scope IN ('국내지역','국가')
  AND NOT EXISTS (
        SELECT 1 FROM STIMULATES_E st JOIN SENSITIVE_E se ON se.code = st.code
        WHERE st.program_id = p.program_id AND se.quadrant = 'Q4')
""")
print("\n도봉구(Q4) 이식후보 수:", cur.fetchone()[0])

cur.close()
conn.close()
print("\n전부 정상. ADB 그래프 연결 OK.")
