from pydantic import BaseModel
from typing import Optional, List, Any
from datetime import datetime
from typing import Dict

class MemoryItem(BaseModel):
    content: Optional[str] = None  # Make content optional for passive logs
    timestamp: datetime
    source: Optional[str] = None          # URL or app context
    title: Optional[str] = None           # Page title
    device: Optional[str] = "chrome_extension"
    source_type: Optional[str] = "active" # 'active' or 'passive'
    type: Optional[str] = "dom_selection" # more fine-grained
    tags: Optional[List[str]] = []
    metadata: Optional[Dict[str, Any]] = None  # Allow any type in metadata
