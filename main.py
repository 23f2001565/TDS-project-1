from fastapi import FastAPI
from pydantic import BaseModel
import requests
import base64
from io import BytesIO
from PIL import Image
import pytesseract
from retriever import SubthreadRetriever
from context_builder import build_context
import os
from dotenv import load_dotenv


load_dotenv()


app = FastAPI()


OPENAI_API_URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
API_KEY = os.getenv("AI_API_KEY")  

if not API_KEY:
    raise ValueError("❌ OPENAI_API_KEY is not set in your .env file")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


class Query(BaseModel):
    question: str
    image: str | None = None 


retriever = SubthreadRetriever()


def extract_text_from_image(base64_str: str) -> str:
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))
        return pytesseract.image_to_string(image).strip()
    except Exception as e:
        print("❌ OCR error:", e)
        return ""


@app.post("/api/")
def get_answer(query: Query):
    user_question = query.question


    if query.image:
        ocr_text = extract_text_from_image(query.image)
        if ocr_text:
            user_question += f"\n\n[Extracted from image]:\n{ocr_text}"

    results = retriever.retrieve(user_question, top_k=5)
    context, sources = build_context(results)

    system_prompt = "You are a helpful TA. Answer clearly and cite relevant sources if provided."
    user_prompt = f"""Use the context below to answer the question.

Context:
{context}

Question: {user_question}
Answer:"""

    payload = {
        "model": "gpt-4o-mini",  
        "temperature": 0.7,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

 
    try:
        response = requests.post(OPENAI_API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        result = response.json()
        
    
        source_citations = "\n\nSources:\n" + "\n".join(f"- {src}" for src in sources if src)
        final_answer = result["choices"][0]["message"]["content"].strip() + source_citations

        return {
            "answer": final_answer,
            "links": sources
        }


        # return {
        #     "answer": result["choices"][0]["message"]["content"].strip(),
        #     "links": sources
        # }

    except requests.exceptions.RequestException as e:
        print("❌ Request error:", e)
        return {
            "answer": "⚠️ OpenAI Proxy request failed.",
            "links": sources,
            "error": str(e)
        }
