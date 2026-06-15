# SHADOW-AI 배포 (OCI 클라우드)

내 노트북 없이 OCI 가상머신(VM)에서 대시보드를 띄워 공개 접속되게 하는 배포 절차.
(= AWS EC2 배포와 같은 IaaS 방식, 클라우드만 오라클)

```
[내 노트북]                         [OCI 클라우드]
 코드 + 데이터 + wallet/.env  ──scp──▶   VM (Ubuntu)
                                         └ Streamlit 대시보드
                                                 │
                                                 └──▶ ADB (그래프 shadow_graph)
```

## 이 폴더
- `vm_setup.sh` — VM(Ubuntu)에서 1회 실행: 파이썬·가상환경·패키지 설치 + 8501 포트 개방
- `requirements-vm.txt` — 시연용 슬림 의존성 (분석 전용 shap/sklearn/scipy/matplotlib 제외)

## 배포 절차 요약

### 1. VM 생성 (OCI Compute → Create instance)
- Image: Canonical Ubuntu 24.04
- Shape: `VM.Standard.E4.Flex` (1 OCPU / 8GB)  ※ 무료 A1은 용량부족 잦음
- Networking: 새 VCN + public subnet, 공인 IP 자동할당
- SSH keys: 공개키 붙여넣기
- 생성 후 Public IP 확인

### 2. OCI 방화벽 포트 열기
인스턴스 → Networking → Subnet → Security → Default Security List → Add Ingress Rules
- Source `0.0.0.0/0` / TCP / Destination Port `8501`

### 3. 파일 전송 + 접속 (내 노트북 Git Bash)
```bash
scp -i ~/.ssh/shadow_oci shadow_deploy.zip ubuntu@<VM_IP>:~/
ssh -i ~/.ssh/shadow_oci ubuntu@<VM_IP>
```

### 4. 설치 (VM 안)
```bash
sudo apt-get update -y && sudo apt-get install -y unzip
unzip shadow_deploy.zip -d shadow && cd shadow
bash deploy/vm_setup.sh
```

### 5. 한글 폴더명 복구 (윈도우 압축 → 리눅스에서 깨짐)
```bash
cd ~/shadow/Outputs
mv "$(find . -name avoidance_index.csv -printf '%h\n')" "복지의 역설"
mv "$(find . -name dependency_index.csv -printf '%h\n')" "편의의 역설"
mv "$(find . -name risk_predictions_final.csv -printf '%h\n')" "전이예측"
cd ~/shadow
```

### 6. 실행 (백그라운드)
```bash
source .venv/bin/activate
nohup streamlit run shadow_service.py --server.port 8501 --server.address 0.0.0.0 \
  --server.headless true --browser.gatherUsageStats false > shadow.log 2>&1 &
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8501   # 200 = 정상
```

### 7. VM 내부 방화벽 순서 수정 (접속 timeout 시)
iptables의 REJECT 규칙이 8501 ACCEPT보다 위에 있으면 차단됨 → 순서 교정:
```bash
sudo iptables -L INPUT --line-numbers -n | grep -E "8501|REJECT|dpt:22"
sudo iptables -D INPUT 6
sudo iptables -I INPUT 5 -p tcp --dport 8501 -j ACCEPT
sudo netfilter-persistent save
```

### 8. 접속
`http://<VM_IP>:8501`

## 트러블슈팅
- 접속 timeout → OCI Security List 8501 + VM iptables 순서
- 차트 일부 안 뜸 → 한글 폴더명 깨짐(5번)
- 처방 안 나옴 → `.env`/`wallet/` 존재 + ADB 접속 확인
- Out of capacity → AD 변경 또는 유료 Shape

## 비용
- E4.Flex 1코어/8GB ≈ 하루 약 1,200원, 다음 달 카드 청구
- **발표 종료 후 인스턴스 Terminate(삭제)** → 과금 정지
