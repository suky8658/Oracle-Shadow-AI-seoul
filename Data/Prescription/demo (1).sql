SELECT COUNT(*) FROM PROGRAMS;
SELECT * FROM PROGRAMS FETCH FIRST 2 ROWS ONLY;
SELECT column_name, data_type FROM user_tab_columns WHERE table_name = 'PROGRAMS';

CREATE TABLE PROGRAM_V AS
SELECT JSON_VALUE(p.DATA,'$.program_id') program_id,
       JSON_VALUE(p.DATA,'$.name') name,
       JSON_VALUE(p.DATA,'$.access_mode') access_mode,
       JSON_VALUE(p.DATA,'$.gender') gender,
       JSON_VALUE(p.DATA,'$.beneficiary') beneficiary,
       JSON_VALUE(p.DATA,'$.status') status
FROM PROGRAMS p;
ALTER TABLE PROGRAM_V ADD PRIMARY KEY (program_id);

CREATE TABLE CODE_V (code VARCHAR2(40) PRIMARY KEY, kind VARCHAR2(10));
INSERT ALL
 INTO CODE_V VALUES('STG_WelfareLabel','stigma') INTO CODE_V VALUES('STG_FaceExposure','stigma')
 INTO CODE_V VALUES('STG_SelfAction','stigma')   INTO CODE_V VALUES('STG_PublicExposure','stigma')
 INTO CODE_V VALUES('STG_RecipientIdentity','stigma')
 INTO CODE_V VALUES('DEP_NoContact','dependency') INTO CODE_V VALUES('DEP_NoOuting','dependency')
 INTO CODE_V VALUES('DEP_NoRelation','dependency') INTO CODE_V VALUES('DEP_LowEfficacy','dependency')
 INTO CODE_V VALUES('DEP_FragileSupply','dependency')
 INTO CODE_V VALUES('NEED_NoStigma','need') INTO CODE_V VALUES('NEED_RelationRestore','need')
 INTO CODE_V VALUES('NEED_Transferable','need') INTO CODE_V VALUES('NEED_UniversalAccess','need')
SELECT * FROM dual;

CREATE TABLE QUADRANT_V (quadrant VARCHAR2(4) PRIMARY KEY);
INSERT ALL INTO QUADRANT_V VALUES('Q1') INTO QUADRANT_V VALUES('Q2')
 INTO QUADRANT_V VALUES('Q3') INTO QUADRANT_V VALUES('Q4') SELECT * FROM dual;

CREATE TABLE SENSITIVE_E (quadrant VARCHAR2(4), code VARCHAR2(40));
INSERT ALL
 INTO SENSITIVE_E VALUES('Q1','STG_WelfareLabel') INTO SENSITIVE_E VALUES('Q1','STG_FaceExposure')
 INTO SENSITIVE_E VALUES('Q1','STG_SelfAction')   INTO SENSITIVE_E VALUES('Q3','STG_SelfAction')
 INTO SENSITIVE_E VALUES('Q4','STG_FaceExposure') INTO SENSITIVE_E VALUES('Q4','STG_PublicExposure')
 INTO SENSITIVE_E VALUES('Q4','STG_RecipientIdentity')
SELECT * FROM dual;

CREATE TABLE VULNERABLE_E (quadrant VARCHAR2(4), code VARCHAR2(40));
INSERT ALL
 INTO VULNERABLE_E VALUES('Q1','DEP_NoContact') INTO VULNERABLE_E VALUES('Q1','DEP_NoOuting')
 INTO VULNERABLE_E VALUES('Q1','DEP_NoRelation') INTO VULNERABLE_E VALUES('Q1','DEP_LowEfficacy')
 INTO VULNERABLE_E VALUES('Q1','DEP_FragileSupply')
 INTO VULNERABLE_E VALUES('Q2','DEP_NoOuting') INTO VULNERABLE_E VALUES('Q2','DEP_NoRelation')
 INTO VULNERABLE_E VALUES('Q2','DEP_LowEfficacy') INTO VULNERABLE_E VALUES('Q2','DEP_FragileSupply')
SELECT * FROM dual;

CREATE TABLE STIMULATES_E AS SELECT JSON_VALUE(p.DATA,'$.program_id') program_id, jt.code
 FROM PROGRAMS p, JSON_TABLE(p.DATA,'$.stimulates_stigma[*]' COLUMNS(code VARCHAR2(40) PATH '$')) jt;
CREATE TABLE DEEPENS_E AS SELECT JSON_VALUE(p.DATA,'$.program_id') program_id, jt.code
 FROM PROGRAMS p, JSON_TABLE(p.DATA,'$.deepens_dependency[*]' COLUMNS(code VARCHAR2(40) PATH '$')) jt;
CREATE TABLE FULFILLS_E AS SELECT JSON_VALUE(p.DATA,'$.program_id') program_id, jt.code
 FROM PROGRAMS p, JSON_TABLE(p.DATA,'$.fulfills_needs[*]' COLUMNS(code VARCHAR2(40) PATH '$')) jt;

 CREATE PROPERTY GRAPH shadow_graph
 VERTEX TABLES (
   PROGRAM_V  KEY(program_id) LABEL program  PROPERTIES(program_id,name,access_mode,gender,beneficiary,status),
   CODE_V     KEY(code)       LABEL code      PROPERTIES(code,kind),
   QUADRANT_V KEY(quadrant)   LABEL quadrant  PROPERTIES(quadrant) )
 EDGE TABLES (
   STIMULATES_E KEY(program_id,code) SOURCE KEY(program_id) REFERENCES PROGRAM_V(program_id) DESTINATION KEY(code) REFERENCES CODE_V(code) LABEL stimulates,
   DEEPENS_E    KEY(program_id,code) SOURCE KEY(program_id) REFERENCES PROGRAM_V(program_id) DESTINATION KEY(code) REFERENCES CODE_V(code) LABEL deepens,
   FULFILLS_E   KEY(program_id,code) SOURCE KEY(program_id) REFERENCES PROGRAM_V(program_id) DESTINATION KEY(code) REFERENCES CODE_V(code) LABEL fulfills,
   SENSITIVE_E  KEY(quadrant,code)   SOURCE KEY(quadrant)   REFERENCES QUADRANT_V(quadrant)   DESTINATION KEY(code) REFERENCES CODE_V(code) LABEL sensitive_to,
   VULNERABLE_E KEY(quadrant,code)   SOURCE KEY(quadrant)   REFERENCES QUADRANT_V(quadrant)   DESTINATION KEY(code) REFERENCES CODE_V(code) LABEL vulnerable_to );


SELECT * FROM GRAPH_TABLE (shadow_graph
  MATCH (q IS quadrant WHERE q.quadrant='Q4') -[IS sensitive_to]-> (s IS code) <-[IS stimulates]- (p IS program)
  COLUMNS (p.program_id, p.name, s.code) )
FETCH FIRST 20 ROWS ONLY;


SELECT COUNT(DISTINCT program_id) AS 충돌제도수 FROM GRAPH_TABLE (shadow_graph
  MATCH (q IS quadrant WHERE q.quadrant='Q4')-[IS sensitive_to]->(s IS code)<-[IS stimulates]-(p IS program)
  COLUMNS (p.program_id AS program_id));


SELECT column_name FROM user_tab_columns WHERE table_name = 'SHADOW_PROFILE';
SELECT SUBSTR(JSON_SERIALIZE(DATA RETURNING CLOB), 1, 2000) AS preview
FROM SHADOW_PROFILE;

SELECT jt.quadrant, COUNT(*)
FROM SHADOW_PROFILE s,
     JSON_TABLE(s.DATA, '$.*' COLUMNS (quadrant VARCHAR2(10) PATH '$.quadrant')) jt
GROUP BY jt.quadrant;


#######################################################

# 자치구 그래프에 연결

CREATE TABLE GU_V AS
SELECT '종로구' gu, JSON_VALUE(DATA,'$."종로구".quadrant') quadrant, JSON_VALUE(DATA,'$."종로구".avoidance' RETURNING NUMBER) avoidance FROM SHADOW_PROFILE UNION ALL
SELECT '중구',     JSON_VALUE(DATA,'$."중구".quadrant'),     JSON_VALUE(DATA,'$."중구".avoidance' RETURNING NUMBER) FROM SHADOW_PROFILE UNION ALL
SELECT '용산구',   JSON_VALUE(DATA,'$."용산구".quadrant'),   JSON_VALUE(DATA,'$."용산구".avoidance' RETURNING NUMBER) FROM SHADOW_PROFILE UNION ALL
SELECT '성동구',   JSON_VALUE(DATA,'$."성동구".quadrant'),   JSON_VALUE(DATA,'$."성동구".avoidance' RETURNING NUMBER) FROM SHADOW_PROFILE UNION ALL
SELECT '광진구',   JSON_VALUE(DATA,'$."광진구".quadrant'),   JSON_VALUE(DATA,'$."광진구".avoidance' RETURNING NUMBER) FROM SHADOW_PROFILE UNION ALL
SELECT '동대문구', JSON_VALUE(DATA,'$."동대문구".quadrant'), JSON_VALUE(DATA,'$."동대문구".avoidance' RETURNING NUMBER) FROM SHADOW_PROFILE UNION ALL
SELECT '중랑구',   JSON_VALUE(DATA,'$."중랑구".quadrant'),   JSON_VALUE(DATA,'$."중랑구".avoidance' RETURNING NUMBER) FROM SHADOW_PROFILE UNION ALL
SELECT '성북구',   JSON_VALUE(DATA,'$."성북구".quadrant'),   JSON_VALUE(DATA,'$."성북구".avoidance' RETURNING NUMBER) FROM SHADOW_PROFILE UNION ALL
SELECT '강북구',   JSON_VALUE(DATA,'$."강북구".quadrant'),   JSON_VALUE(DATA,'$."강북구".avoidance' RETURNING NUMBER) FROM SHADOW_PROFILE UNION ALL
SELECT '도봉구',   JSON_VALUE(DATA,'$."도봉구".quadrant'),   JSON_VALUE(DATA,'$."도봉구".avoidance' RETURNING NUMBER) FROM SHADOW_PROFILE UNION ALL
SELECT '노원구',   JSON_VALUE(DATA,'$."노원구".quadrant'),   JSON_VALUE(DATA,'$."노원구".avoidance' RETURNING NUMBER) FROM SHADOW_PROFILE UNION ALL
SELECT '은평구',   JSON_VALUE(DATA,'$."은평구".quadrant'),   JSON_VALUE(DATA,'$."은평구".avoidance' RETURNING NUMBER) FROM SHADOW_PROFILE UNION ALL
SELECT '서대문구', JSON_VALUE(DATA,'$."서대문구".quadrant'), JSON_VALUE(DATA,'$."서대문구".avoidance' RETURNING NUMBER) FROM SHADOW_PROFILE UNION ALL
SELECT '마포구',   JSON_VALUE(DATA,'$."마포구".quadrant'),   JSON_VALUE(DATA,'$."마포구".avoidance' RETURNING NUMBER) FROM SHADOW_PROFILE UNION ALL
SELECT '양천구',   JSON_VALUE(DATA,'$."양천구".quadrant'),   JSON_VALUE(DATA,'$."양천구".avoidance' RETURNING NUMBER) FROM SHADOW_PROFILE UNION ALL
SELECT '강서구',   JSON_VALUE(DATA,'$."강서구".quadrant'),   JSON_VALUE(DATA,'$."강서구".avoidance' RETURNING NUMBER) FROM SHADOW_PROFILE UNION ALL
SELECT '구로구',   JSON_VALUE(DATA,'$."구로구".quadrant'),   JSON_VALUE(DATA,'$."구로구".avoidance' RETURNING NUMBER) FROM SHADOW_PROFILE UNION ALL
SELECT '금천구',   JSON_VALUE(DATA,'$."금천구".quadrant'),   JSON_VALUE(DATA,'$."금천구".avoidance' RETURNING NUMBER) FROM SHADOW_PROFILE UNION ALL
SELECT '영등포구', JSON_VALUE(DATA,'$."영등포구".quadrant'), JSON_VALUE(DATA,'$."영등포구".avoidance' RETURNING NUMBER) FROM SHADOW_PROFILE UNION ALL
SELECT '동작구',   JSON_VALUE(DATA,'$."동작구".quadrant'),   JSON_VALUE(DATA,'$."동작구".avoidance' RETURNING NUMBER) FROM SHADOW_PROFILE UNION ALL
SELECT '관악구',   JSON_VALUE(DATA,'$."관악구".quadrant'),   JSON_VALUE(DATA,'$."관악구".avoidance' RETURNING NUMBER) FROM SHADOW_PROFILE UNION ALL
SELECT '서초구',   JSON_VALUE(DATA,'$."서초구".quadrant'),   JSON_VALUE(DATA,'$."서초구".avoidance' RETURNING NUMBER) FROM SHADOW_PROFILE UNION ALL
SELECT '강남구',   JSON_VALUE(DATA,'$."강남구".quadrant'),   JSON_VALUE(DATA,'$."강남구".avoidance' RETURNING NUMBER) FROM SHADOW_PROFILE UNION ALL
SELECT '송파구',   JSON_VALUE(DATA,'$."송파구".quadrant'),   JSON_VALUE(DATA,'$."송파구".avoidance' RETURNING NUMBER) FROM SHADOW_PROFILE UNION ALL
SELECT '강동구',   JSON_VALUE(DATA,'$."강동구".quadrant'),   JSON_VALUE(DATA,'$."강동구".avoidance' RETURNING NUMBER) FROM SHADOW_PROFILE;

ALTER TABLE GU_V ADD PRIMARY KEY (gu);


CREATE TABLE IN_REGION_E AS
SELECT JSON_VALUE(p.DATA,'$.program_id') program_id, jt.gu
FROM PROGRAMS p, JSON_TABLE(p.DATA,'$.regions[*]' COLUMNS(gu VARCHAR2(20) PATH '$')) jt
WHERE jt.gu <> '서울전체';


SELECT g.gu 자치구, g.quadrant Q, ROUND(g.avoidance,1) 회피지수,
       COUNT(DISTINCT st.program_id) 충돌제도수
FROM GU_V g
JOIN SENSITIVE_E se ON se.quadrant = g.quadrant
JOIN STIMULATES_E st ON st.code = se.code
JOIN IN_REGION_E ir ON ir.program_id = st.program_id AND ir.gu = g.gu
GROUP BY g.gu, g.quadrant, g.avoidance
ORDER BY 충돌제도수 DESC;


SELECT g.gu 자치구, g.quadrant Q, ROUND(g.avoidance,1) 회피지수,
       COUNT(DISTINCT st.program_id) 낙인충돌
FROM GU_V g
LEFT JOIN SENSITIVE_E se ON se.quadrant = g.quadrant
LEFT JOIN STIMULATES_E st ON st.code = se.code
LEFT JOIN IN_REGION_E ir ON ir.program_id = st.program_id AND ir.gu = g.gu
GROUP BY g.gu, g.quadrant, g.avoidance
ORDER BY 낙인충돌 DESC;

SELECT COUNT(*) FROM GU_V;        -- 25 나와야 정상
SELECT COUNT(*) FROM IN_REGION_E; -- 제도-자치구 연결 (수천 개)

SELECT g.gu 자치구, g.quadrant Q, ROUND(g.avoidance,1) 회피지수,
  (SELECT COUNT(DISTINCT st.program_id)
     FROM SENSITIVE_E se JOIN STIMULATES_E st ON st.code=se.code
     JOIN IN_REGION_E ir ON ir.program_id=st.program_id AND ir.gu=g.gu
    WHERE se.quadrant=g.quadrant) 낙인충돌,
  (SELECT COUNT(DISTINCT de.program_id)
     FROM VULNERABLE_E ve JOIN DEEPENS_E de ON de.code=ve.code
     JOIN IN_REGION_E ir ON ir.program_id=de.program_id AND ir.gu=g.gu
    WHERE ve.quadrant=g.quadrant) 의존충돌
FROM GU_V g
ORDER BY 회피지수 DESC;

################################################################

# 이식후보 처방
SELECT DISTINCT p.program_id, p.name
FROM PROGRAM_V p
JOIN FULFILLS_E f ON f.program_id = p.program_id
WHERE f.code IN ('NEED_RelationRestore','NEED_NoStigma')
  AND NOT EXISTS (SELECT 1 FROM STIMULATES_E st JOIN SENSITIVE_E se ON se.code=st.code
                  WHERE st.program_id=p.program_id AND se.quadrant='Q4')
  AND NOT EXISTS (SELECT 1 FROM IN_REGION_E ir WHERE ir.program_id=p.program_id AND ir.gu='도봉구')
FETCH FIRST 20 ROWS ONLY;

SELECT * FROM GRAPH_TABLE (shadow_graph
  MATCH (p IS program WHERE p.program_id='P-0002') -[]-> (c IS code) <-[]- (q IS quadrant)
  COLUMNS (p.name AS 제도, c.code AS 코드, q.quadrant AS 민감분면) );