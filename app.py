from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from langserve import add_routes
from pirate_speak.ncc_rag import rag_chain
import time
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Pydantic model for request validation
class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Add the /ncc_rag route
add_routes(app, rag_chain, path="/ncc_rag")


# Add the new /chat route
@app.post("/chat")
async def chat(request: ChatRequest):
    question = request.message
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
