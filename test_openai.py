# -*- coding: utf-8 -*-
"""
test_openai.py — OpenAI 호출 단독 테스트.
실행: python test_openai.py
성공하면 모델이 만든 한 줄이 찍힘.
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

print("OpenAI 호출 중...")
resp = client.chat.completions.create(
    model="gpt-4o-mini",  # 테스트용 저렴한 모델. 나중에 gpt-4o 등으로 교체 가능
    messages=[
        {"role": "system", "content": "너는 한국어로 답하는 비서야."},
        {"role": "user", "content": "연결 테스트야. '안녕, 연결됐어' 라고만 답해줘."},
    ],
)
print("응답:", resp.choices[0].message.content)
print("\nOpenAI 연결 OK.")
