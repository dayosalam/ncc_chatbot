import logging
import sys
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
PACKAGE_DIR = PROJECT_DIR / "packages" / "pirate-speak"

if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

app = FastAPI(title="NCC Chatbot")

# CORS Middleware to allow requests from the frontend or other domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust this as needed for security)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Pydantic model for request validation
class ChatRequest(BaseModel):
    message: str


# Mount the static directory
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# Initialize Jinja2 templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@lru_cache(maxsize=1)
def get_rag_chain():
    try:
        from pirate_speak.ncc_rag import rag_chain
    except Exception as exc:
        LOGGER.exception("Unable to initialize RAG chain: %s", exc)
        return None
    return rag_chain


@app.get("/", response_class=HTMLResponse)
async def render_home(request: Request):
    return templates.TemplateResponse("chat-new.html", {"request": request})


@app.get("/healthz")
async def healthcheck():
    rag_chain = get_rag_chain()
    return {
        "status": "ok",
        "rag_available": rag_chain is not None,
    }


def build_fallback_response() -> str:
    return (
        "The chatbot service is running, but the knowledge engine is not configured yet. "
        "Set `GROQ_API_KEY`, `OPENAI_API_KEY`, and build the persisted vector store before trying again."
    )


@app.post("/chat")
async def chat(request: ChatRequest):
    question = request.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    start_time = time.time()
    rag_chain = get_rag_chain()

    if rag_chain is None:
        return {"response": build_fallback_response()}

    try:
        response = rag_chain.invoke({"question": question})
    except Exception as exc:
        LOGGER.exception("RAG invocation failed: %s", exc)
        response = build_fallback_response()

    LOGGER.info("Chat request completed in %.2fs", time.time() - start_time)
    return {"response": response}


@app.get("/ncc_chatbot", response_class=HTMLResponse)
async def render_template(request: Request):
    return templates.TemplateResponse("chat-new.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
