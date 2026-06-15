#!/usr/bin/env bash
# ============================================================
# SHADOW-AI : OCI VM(Ubuntu) 1회 설치 + 실행 스크립트
# 사용법:  VM에 파일 올린 뒤  ->  bash vm_setup.sh
# ============================================================
set -e
cd "$(dirname "$0")"

echo "[1/5] 시스템 패키지 설치..."
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv fonts-nanum   # fonts-nanum = 혹시 모를 한글 대비

echo "[2/5] 파이썬 가상환경..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

echo "[3/5] 패키지 설치 (requirements.txt)..."
pip install -r requirements.txt

echo "[4/5] VM 자체 방화벽에 8501 포트 열기 (OCI Ubuntu 이미지 필수!)..."
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8501 -j ACCEPT || true
sudo netfilter-persistent save 2>/dev/null || sudo bash -c 'iptables-save > /etc/iptables/rules.v4' 2>/dev/null || true

echo "[5/5] 완료. 아래 명령으로 대시보드 실행:"
echo "------------------------------------------------------------"
echo "  source .venv/bin/activate"
echo "  nohup streamlit run shadow_service.py \\"
echo "        --server.port 8501 --server.address 0.0.0.0 \\"
echo "        --server.headless true --browser.gatherUsageStats false \\"
echo "        > shadow.log 2>&1 &"
echo "------------------------------------------------------------"
echo "그 뒤 브라우저:  http://<VM_공인IP>:8501"
