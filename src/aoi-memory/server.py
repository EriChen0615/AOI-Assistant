from fastapi import FastAPI
from schema import MemoryItem
from db import init_db, insert_memory
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to ["chrome-extension://<your-extension-id>"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
init_db()

@app.post("/remember")
def remember(item: MemoryItem):
    insert_memory(item)
    return {"status": "ok"}
