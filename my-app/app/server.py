from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from langserve import add_routes
from pirate_speak.ncc_rag import rag_chain
from fastapi.staticfiles import StaticFiles
import time
from pydantic import BaseModel
import uvicorn

app = FastAPI()

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
app.mount("/static", StaticFiles(directory="static"), name="static")


# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Add the /ncc_rag route
add_routes(app, rag_chain, path="/ncc_rag")

# Add the new /chat route
@app.post("/chat")
async def chat(request: ChatRequest):
    question = request.message
    ncc = "ncc"
    question = question + ncc
    start_time = time.time()
    print(f"Start time is: {start_time}")
    print(f"This is my data coming from the frontend {request}")

    # Invoke the rag_chain with the question
    response = rag_chain.invoke({"question": question})

    print(f"This is my data sending to the frontend {response}")
    end_time = time.time()
    print(f"Time Taken is: {end_time - start_time}")

    # Return the response
    return {"response": response}

# Add a new route to render the HTML template
@app.get("/ncc_chatbot")
async def render_template(request: Request):
    return templates.TemplateResponse("chat-new.html", {"request": request, "message": "Hello, FastAPI!"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
