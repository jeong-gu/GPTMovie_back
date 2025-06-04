from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os
from fastapi.middleware.cors import CORSMiddleware

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # 또는 직접 문자열로 입력 가능

app = FastAPI()

# CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class GPTRequest(BaseModel):
    message: str

@app.post("/recommend")
def recommend(req: GPTRequest):
    response = client.chat.completions.create(
        model="gpt-4o",  # 또는 "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": "당신은 영화 추천을 도와주는 조력자입니다."},
            {"role": "user", "content": req.message}
        ]
    )
    return {"reply": response.choices[0].message.content}
