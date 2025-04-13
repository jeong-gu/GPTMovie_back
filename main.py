from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os
from fastapi.middleware.cors import CORSMiddleware

openai.api_key = os.getenv("OPENAI_API_KEY")  # .env에 키 저장 권장

app = FastAPI()

# CORS 허용 (앱에서 접근 가능하게)
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
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": req.message}]
    )
    return {"reply": response.choices[0].message["content"]}
