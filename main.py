"""
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from chat import get_response
from proto import NLPProto

app = FastAPI(title="chatbot", description="Botsito", version="0.0.1")
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Endpoint de Chatbot Acecom"}


@app.post("/nlp")
def nlp_task(q: NLPProto):
    try:
        return {
            "type": "response",
            "data": get_response(q.query),
        }
    except BaseException as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
