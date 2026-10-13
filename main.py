from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from google import genai
from typing import List, Optional
from google.genai import types
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Rate limiting imports
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Static files
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

load_dotenv()

# Initialize the rate limiter tracking requests by client IP address
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Career AI Assistant API (Gemini)",
    description="A microservice powered by Google Gemini to handle inquiries about my professional background.",
    version="1.0.0"
)

# Register rate limiter error handler into the FastAPI app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your specific portfolio domain (e.g., "https://manassahoo.dev")
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static directory to serve HTML, CSS, and JS files
app.mount("/static", StaticFiles(directory="static"), name="static")

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

class MessageParam(BaseModel):
    role: str  # "user" or "model"
    content: str

class ChatRequest(BaseModel):
    # Restricts the incoming message string to a max length of 500 characters
    message: str = Field(..., max_length=500, description="The user's chat prompt")
    history: Optional[List[MessageParam]] = []  # Pass prior conversation turns here


class ChatResponse(BaseModel):
    reply: str


@app.post("/api/chat", response_model=ChatResponse)
@limiter.limit("5/minute")  # Limits each IP address to a maximum of 5 requests per minute
async def chat_with_assistant(request: Request, body: ChatRequest):
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        # Using a free tier model with system instructions configured natively
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=body.message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                max_output_tokens=500,
            ),
        )
        return ChatResponse(reply=response.text)

    except Exception as e:
        print(f"ERROR DETAILS: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process the request.")


@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")