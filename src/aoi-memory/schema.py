from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from typing import Dict

class MemoryItem(BaseModel):
    content: str
    timestamp: datetime
    source: Optional[str] = None          # URL or app context
    device: Optional[str] = "chrome_extension"
    source_type: Optional[str] = "active" # 'active' or 'ambient'
    type: Optional[str] = "dom_selection" # more fine-grained
    tags: Optional[List[str]] = []
    metadata: Optional[Dict[str, str]] = None
