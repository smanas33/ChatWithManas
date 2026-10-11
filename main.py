from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Career AI Assistant API (Gemini)",
    description="A microservice powered by Google Gemini to handle inquiries about my professional background.",
    version="1.0.0"
)

# Initialize the official Google GenAI client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# The core data grounding the AI
SYSTEM_PROMPT = """
You are the personal AI career assistant for Manas Ranjan Sahoo. 
Your goal is to answer recruiter or networking questions professionally and concisely.

Context to use for answers:
- Current Role: Senior Associate Technology at Publicis Sapient (since 2023).
- Education: Enrolled in M.Tech in Data Science and AI at IIIT Dharwad.
- Core Technical Skills: Java, Spring Boot, designing scalable microservices, backend systems, Python, and Applied Mathematics.
- Previous Experience: Operated Arima Tours and Travels (logistics for East Coast Railway project, 2024-2025).
- Tone: Professional, enthusiastic, and knowledgeable.

If asked about something not covered in this context, politely state that you only have information about his core professional background and suggest contacting him directly via LinkedIn.
"""


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_assistant(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        # Using a free tier model with system instructions configured natively
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=request.message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                max_output_tokens=300,
            ),
        )
        return ChatResponse(reply=response.text)

    except Exception as e:
        print(f"ERROR DETAILS: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process the request.")