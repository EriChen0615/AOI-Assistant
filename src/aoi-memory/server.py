from fastapi import FastAPI
from schema import MemoryItem
from db import init_db, insert_memory, recall_memory
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to ["chrome-extension://<your-extension-id>"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
init_db()

@app.get("/recall")
def recall(query: str, time_range: str=None, source_type: str=None, tags: List[str]=None):
    logger.info(f"Recall memory: {query}, {time_range}, {source_type}, {tags}")
    return recall_memory(query, time_range, source_type, tags)

@app.post("/remember")
def remember(item: MemoryItem):
    logger.info(f"Active memory received: {item.type} from {item.source} - '{item.content[:50]}{'...' if len(item.content) > 50 else ''}'")
    insert_memory(item)
    return {"status": "ok"}

@app.post("/passive-log")
def passive_log(item: MemoryItem):
    """
    Handle passive tracking logs from the Chrome extension.
    These are automatic logs for page views, idle states, etc.
    """
    # For passive logs, content is typically null, so we create a descriptive content
    if not item.content:
        if item.type == "passive_page_view":
            item.content = f"Viewed page: {item.title or item.source or 'unknown page'}"
        elif item.type == "passive_idle_start":
            last_title = item.metadata.get('last_active_title') if item.metadata else 'unknown page'
            item.content = f"User became idle while on: {last_title}"
        elif item.type == "passive_idle_end":
            duration = item.metadata.get('idle_duration', 0) if item.metadata else 0
            item.content = f"User returned from idle state (duration: {duration}ms)"
        elif item.type == "passive_tab_switch":
            from_title = item.metadata.get('from_title', 'unknown') if item.metadata else 'unknown'
            to_title = item.metadata.get('to_title', 'unknown') if item.metadata else 'unknown'
            item.content = f"Switched from '{from_title}' to '{to_title}'"
        else:
            item.content = f"Passive event: {item.type}"
    
    logger.info(f"Passive log received: {item.type} from {item.source} - '{item.content}'")
    insert_memory(item)
    return {"status": "ok", "type": "passive_log"}
