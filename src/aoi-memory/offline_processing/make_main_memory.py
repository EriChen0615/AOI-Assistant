""""
Project AOI.
Jinghong Chen 2025.6. 

This script takes the raw capture database to obtain a main memory rather for indexing & RAG.

A captured event is a tuple of (id, when, where, what, who). The `what` item may contain other events. 
A memory item is a key-value pair. The key is used for indexing for sparse/dense retrieval and is purely text-based. Value contains augmentation items.

def MakeMainMemory(all_events):
    all_memory_items = []
    for events in slice_by_time(all_events):
        events = events.filter(FilterEvents).map(NormalizeEvents) # filter dummy events
        memory_items = MakeMemByClustering(events, llm)
        memory_items = memory_items.map(AugmentMemoryItems)
        all_memory_items.extend(memory_items)
    return all_memory_items


all_events = ReadEvents(source_file)
all_memory_items = MakeMainMemory(all_events)
retrieval_db = UpdateDatabase(all_memory_items)
"""

from dataclasses import dataclass
from typing import List
@dataclass
class CapturedEvent:
    _id: str
    user_id: str
    capture_type: str # "passive", "active"
    device: str  # "chromeextension", "gallery", ...
    timestamp: str
    location: str
    content: str
    detail_content: dict

@dataclass
class MemoryItem:
    _id: str
    key: str
    summary: str
    recall_cue: str
    tags: str
    detail_content: dict

import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_db_type", type=str, default="sqlite")
    parser.add_argument("--input_db_path", type=str, required=True)
    parser.add_argument("--output_db_type", type=str, default="sqlite")
    parser.add_argument("--output_db_path", type=str, required=True)
    parser.add_argument("--llm_model", type=str, default="gpt-4.1-nano")
    parser.add_argument("--timeslice_size", type=str, default="1day")
    return parser.parse_args()

def ReadEventsSliceByTime(db_path, db_type, timeslice_size):
    if db_type == "sqlite":
        import sqlite3
        from datetime import datetime, timedelta
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get the date range from the database
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM memory")
        min_date, max_date = cursor.fetchone()
        
        if not min_date or not max_date:
            conn.close()
            return

        print("Connected to database: ", db_path)
        print("Min date: ", min_date)
        print("Max date: ", max_date)
        
        # Parse dates - handle the format 2025-06-19T18:14:00.561000+00:00
        start_date = datetime.fromisoformat(min_date)
        end_date = datetime.fromisoformat(max_date)
        
        # Ensure both dates are timezone-aware or timezone-naive
        if start_date.tzinfo is None and end_date.tzinfo is not None:
            start_date = start_date.replace(tzinfo=end_date.tzinfo)
        elif start_date.tzinfo is not None and end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=start_date.tzinfo)
        
        # Parse timeslice_size (assuming format like "1day", "2day", etc.)
        if timeslice_size.endswith('day'):
            days = int(timeslice_size[:-3]) if timeslice_size[:-3] else 1
        else:
            days = 1  # default to 1 day
        
        current_date = start_date
        
        while current_date <= end_date:
            next_date = current_date + timedelta(days=days)
            
            # Query events for this time slice
            cursor.execute("""
                SELECT id, content, timestamp, source, device, source_type, tags, metadata FROM memory 
                WHERE timestamp >= ? AND timestamp < ?
                ORDER BY timestamp
            """, (current_date.isoformat(), next_date.isoformat()))
            
            events = cursor.fetchall()
            if events:  # Only yield if there are events in this slice
                returned_events = [
                    CapturedEvent(
                        _id=event[0],
                        content=event[1][:2048] + "\nURL: " + event[3],
                        timestamp=event[2],
                        user_id='jinghong_chen',
                        device=event[4],
                        capture_type=event[5],
                        location='web',
                        detail_content={'content': event[1], 'tags': event[6], 'metadata': event[7]}
                    ) for event in events]
                yield returned_events
            current_date = next_date
        conn.close()
    else:
        raise NotImplementedError(f"Unsupported database type: {db_type}")

def FilterEvent(event):
    if event.device == "chrome_extension":
        if event.capture_type == "active":
            return True
        else:
            if event.device == "chrome_extension":
                if any(filter_kw in event.detail_content['tags'] for filter_kw in ["idle_start", "idle_end", "tab_switch"]) \
                    or event.content.startswith("Viewed page: New Tab"):
                    return False
                else:
                    return True
            else:
                return False
    else:
        return False

def AugmentEvent(event, llm_model):
    pass

def UpdateDatabase(memory_items, db_path, db_type):
    pass

def MakeMemByClustering(events, llm_model):
    pass

from functools import partial
def main(args):
    all_memory_items = []
    for events in ReadEventsSliceByTime(args.input_db_path, args.input_db_type, args.timeslice_size):
        print("Before filtering len(events)= ", len(events))
        """Step 1: Filter & normalize events"""
        events = list(filter(FilterEvent, events))
        print("After filtering len(events)= ", len(events))
        breakpoint()

        """Step 3: Make memory items by LLM generative clustering"""
        memory_items = MakeMemByClustering(events, llm_model=args.llm_model)

        """Step 4: Augment memory items with LLM"""
        AugmentEventFnc = partial(AugmentEvent, llm_model=args.llm_model)
        memory_items = map(AugmentEventFnc, memory_items)
        all_memory_items.extend(memory_items)

    """Step 5: Update database"""
    UpdateDatabase(all_memory_items, args.output_db_path, args.output_db_type)


if __name__ == "__main__":
    args = get_args()
    main(args)

