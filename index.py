from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64, os, requests
from io import BytesIO
from PIL import Image
import pytesseract
import os
from mangum import Mangum
from retriever import SubthreadRetriever
from context_builder import build_context
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

USE_PROXY = True
API_URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
PROXY_API_KEY = os.getenv("AI_API_KEY")

if USE_PROXY:
    HEADERS = {
        "Authorization": f"Bearer {PROXY_API_KEY}",
        "Content-Type": "application/json"
    }

class Query(BaseModel):
    question: str
    image: str | None = None

retriever = SubthreadRetriever()

def extract_text(base64_str):
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))
        return pytesseract.image_to_string(image).strip()
    except Exception as e:
        return f"OCR Error: {e}"

@app.post("/")
async def handler(query: Query):
    q = query.question
    if query.image:
        q += "\n\n[Image OCR]:\n" + extract_text(query.image)

    results = retriever.retrieve(q, top_k=5)
    context, sources = build_context(results)

    payload = {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "messages": [
            {"role": "system", "content": "You are a helpful TA."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q}"}
        ]
    }

    try:
        r = requests.post(API_URL, headers=HEADERS, json=payload)
        r.raise_for_status()
        result = r.json()
        return {
            "answer": result["choices"][0]["message"]["content"].strip(),
            "links": sources
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

handler = Mangum(app)