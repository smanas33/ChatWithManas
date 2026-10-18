# main.py
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Rate limiting imports
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import compiled LangGraph workflow from graph subfolder
from graph.graph import nexus_graph

load_dotenv()

# Initialize the rate limiter tracking requests by client IP address
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="NEXUS AI Agent API (LangGraph + RAG)",
    description="A microservice powered by LangGraph, Pinecone, and Gemini to answer career inquiries for Manas Ranjan Sahoo.",
    version="1.0.0"
)

# Register rate limiter into FastAPI app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite local development port
        "https://your-production-frontend-domain.com"  # Your future deployed frontend URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MessageParam(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=500, description="The user's chat prompt")
    history: Optional[List[MessageParam]] = []


class ChatResponse(BaseModel):
    reply: str


@app.post("/api/chat", response_model=ChatResponse)
@limiter.limit("5/minute")
async def chat_with_assistant(request: Request, body: ChatRequest):
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Format previous history turns into a single string for LangGraph
    history_str = ""
    if body.history:
        turns = [f"{'User' if m.role == 'user' else 'NEXUS'}: {m.content}" for m in body.history]
        history_str = "\n".join(turns)

    try:
        # Prepare graph input state matching GraphState in graph/graph.py
        inputs = {
            "question": body.message,
            "history": history_str
        }

        # Execute compiled LangGraph workflow
        result = nexus_graph.invoke(inputs)

        # Ensure the reply is strictly a string
        raw_reply = result.get("response", "")
        if isinstance(raw_reply, list):
            reply_text = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in raw_reply
            )
        else:
            reply_text = str(raw_reply)

        return ChatResponse(reply=reply_text)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(e)}"
        )


@app.get("/")
async def root_health_check():
    return {
        "status": "online",
        "service": "NEXUS AI Agent API",
        "creator": "Manas Ranjan Sahoo",
        "docs": "/docs"
    }