# Updated memory insertion logic to use the new readable ID format

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime, timezone
import sqlite3
import json
import os
import random
import string

DB_PATH = "../../data/aoi-memory/memory.db"

# --- Helper function to generate readable IDs ---
def generate_memory_id(device: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H')
    device_clean = ''.join(c for c in device if c.isalnum()).lower()
    suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"M{timestamp}-{device_clean}-{suffix}"

# --- Pydantic model ---
class MemoryItem(BaseModel):
    content: str
    timestamp: datetime
    source: Optional[str] = None
    device: Optional[str] = "chrome_extension"
    source_type: Optional[str] = "active"
    type: Optional[str] = "dom_selection"
    tags: Optional[List[str]] = []
    metadata: Optional[Dict[str, str]] = None

# --- DB initialization ---
def init_db():
    # Ensure the directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            source TEXT,
            device TEXT,
            source_type TEXT,
            type TEXT,
            tags TEXT,
            metadata TEXT
        )
    """)
    conn.commit()
    conn.close()

# --- Insert memory record ---
def insert_memory(item: MemoryItem):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    memory_id = generate_memory_id(item.device)
    c.execute("""
        INSERT INTO memory (id, content, timestamp, source, device, source_type, type, tags, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        memory_id,
        item.content,
        item.timestamp.isoformat(),
        item.source,
        item.device,
        item.source_type,
        item.type,
        json.dumps(item.tags or []),
        json.dumps(item.metadata or {})
    ))
    conn.commit()
    conn.close()

# --- FastAPI setup ---
app = FastAPI()
init_db()

@app.post("/remember")
def remember(item: MemoryItem):
    insert_memory(item)
    return {"status": "ok", "message": "Memory stored successfully."}

