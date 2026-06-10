-- ============================================================
-- SHADOW 그래프 재빌드 — PROGRAMS(2403) 재업로드 후 실행
-- 위에서 아래로 한 번에 (F5 = 스크립트 실행)
-- ============================================================

-- 1) 제도 노드 (region_scope 추가 = 서울/국내/해외 구분)
CREATE TABLE PROGRAM_V AS
SELECT JSON_VALUE(p.DATA,'$.program_id')  program_id,
       JSON_VALUE(p.DATA,'$.name')        name,
       JSON_VALUE(p.DATA,'$.access_mode') access_mode,
       JSON_VALUE(p.DATA,'$.gender')      gender,
       JSON_VALUE(p.DATA,'$.beneficiary') beneficiary,
       JSON_VALUE(p.DATA,'$.status')      status,
       JSON_VALUE(p.DATA,'$.region_scope') region_scope
FROM PROGRAMS p;
ALTER TABLE PROGRAM_V ADD PRIMARY KEY (program_id);

-- 2) 코드 노드 14개 + 분면 + Q민감/취약 규칙 (고정값)
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

-- 3) 제도 -> 코드 엣지 (배열 펼치기)
CREATE TABLE STIMULATES_E AS SELECT JSON_VALUE(p.DATA,'$.program_id') program_id, jt.code
 FROM PROGRAMS p, JSON_TABLE(p.DATA,'$.stimulates_stigma[*]' COLUMNS(code VARCHAR2(40) PATH '$')) jt;
CREATE TABLE DEEPENS_E AS SELECT JSON_VALUE(p.DATA,'$.program_id') program_id, jt.code
 FROM PROGRAMS p, JSON_TABLE(p.DATA,'$.deepens_dependency[*]' COLUMNS(code VARCHAR2(40) PATH '$')) jt;
CREATE TABLE FULFILLS_E AS SELECT JSON_VALUE(p.DATA,'$.program_id') program_id, jt.code
 FROM PROGRAMS p, JSON_TABLE(p.DATA,'$.fulfills_needs[*]' COLUMNS(code VARCHAR2(40) PATH '$')) jt;

-- 4) 자치구 노드 (자치구 -> Q + 회피지수)
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

-- 5) 제도 -> 자치구 (시행지). 해외/국내타지역도 regions가 자치구가 아니라서 자연히 빠짐(=이식후보)
CREATE TABLE IN_REGION_E AS
SELECT JSON_VALUE(p.DATA,'$.program_id') program_id, jt.gu
FROM PROGRAMS p, JSON_TABLE(p.DATA,'$.regions[*]' COLUMNS(gu VARCHAR2(20) PATH '$')) jt
WHERE jt.gu <> '서울전체';

-- 6) 그래프 생성 (region_scope 속성 포함)
CREATE PROPERTY GRAPH shadow_graph
 VERTEX TABLES (
   PROGRAM_V  KEY(program_id) LABEL program  PROPERTIES(program_id,name,access_mode,gender,beneficiary,status,region_scope),
   CODE_V     KEY(code)       LABEL code      PROPERTIES(code,kind),
   QUADRANT_V KEY(quadrant)   LABEL quadrant  PROPERTIES(quadrant) )
 EDGE TABLES (
   STIMULATES_E KEY(program_id,code) SOURCE KEY(program_id) REFERENCES PROGRAM_V(program_id) DESTINATION KEY(code) REFERENCES CODE_V(code) LABEL stimulates,
   DEEPENS_E    KEY(program_id,code) SOURCE KEY(program_id) REFERENCES PROGRAM_V(program_id) DESTINATION KEY(code) REFERENCES CODE_V(code) LABEL deepens,
   FULFILLS_E   KEY(program_id,code) SOURCE KEY(program_id) REFERENCES PROGRAM_V(program_id) DESTINATION KEY(code) REFERENCES CODE_V(code) LABEL fulfills,
   SENSITIVE_E  KEY(quadrant,code)   SOURCE KEY(quadrant)   REFERENCES QUADRANT_V(quadrant)   DESTINATION KEY(code) REFERENCES CODE_V(code) LABEL sensitive_to,
   VULNERABLE_E KEY(quadrant,code)   SOURCE KEY(quadrant)   REFERENCES QUADRANT_V(quadrant)   DESTINATION KEY(code) REFERENCES CODE_V(code) LABEL vulnerable_to );

-- 확인: 건수
SELECT COUNT(*) AS 제도수 FROM PROGRAM_V;
SELECT region_scope, COUNT(*) FROM PROGRAM_V GROUP BY region_scope;


SELECT COUNT(DISTINCT p.program_id) AS 도봉구_이식후보수
FROM PROGRAM_V p JOIN FULFILLS_E f ON f.program_id=p.program_id
WHERE f.code IN ('NEED_RelationRestore','NEED_NoStigma')
  AND p.region_scope IN ('국내지역','국가')
  AND NOT EXISTS (SELECT 1 FROM STIMULATES_E st JOIN SENSITIVE_E se ON se.code=st.code
                  WHERE st.program_id=p.program_id AND se.quadrant='Q4');
