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

from dataclasses import dataclass, field
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
    recall_cue: str
    source_event_ids: List[str] 
    key: str = ""
    summary: str = ""
    tags: str = ""
    detail_content: dict = field(default_factory=dict)

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

import os
from tqdm import tqdm
import uuid
def MakeMemByClustering(events, llm_model):
    LLM_INSTRUCTION = (
        "You are a helpful assistant that clusters events into memory items. "
        "User will providea list of events indexed by Event 0, 1, ..., etc."
        "You should cluster similar events into a memory item and provide a cue for recalling the memory item. The cue should be a single sentence that succiently summarizes the memory item."
        "You should use the events with CaptureType=active to guide your recall cue. But you should not omit events with CaptureType=passive entirely."
        "You should aim to produce no more than 10 memory items while making sure that far-apart events that are captured in different memory items."
        "You should respond in the following format: "
        "[Reasoning] ...\n"
        "[Memory Item 1] Events=[<event indices> ... ] <sep> Recall Cue=...\n"
        "[Memory Item 2] Events=[<event indices> ... ] <sep> Recall Cue=...\n"
        "You may use the shorthand <start_idx>-<end_idx> to represent a continuous range of event indices."
    )
    def _make_memory_item_id():
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d-%H")
        uuid_str = str(uuid.uuid4())[:8]  # Take first 8 characters for readability
        id_str = f"{timestamp}-{uuid_str}"
        return id_str
    
    def _parse_llm_resp_to_memory_items(events, llm_resp):
        memory_items = []
        for line in llm_resp.split("\n"):
            if line.startswith("[Memory Item"):
                event_indices = []
                to_parse = line.split("Events=[")[1].split("]")[0].strip().split(',')
                for v in to_parse:
                    if '-' in v:
                        try:
                            start_idx, end_idx = v.split('-')
                            event_indices.extend(range(int(start_idx), int(end_idx)+1))
                        except Exception as e:
                            print(f"Error parsing event indices: {e}")
                            print(f"string to parse: {v}")
                    else:
                        try:
                            event_indices.append(int(v))
                        except Exception as e:
                            print(f"Error parsing event indices: {e}")
                            print(f"string to parse: {v}")

                source_events = []
                for idx in event_indices:
                    if int(idx) < len(events):
                        source_events.append(events[int(idx)])
                    else:
                        print(f"Event index out of range: {idx}")
                        print(f"events length: {len(events)}")
                memory_items.append(MemoryItem(
                    _id=_make_memory_item_id(),
                    recall_cue=line.split("<sep>")[1].split("Recall Cue=")[1].strip(),
                    source_event_ids=[event._id for event in source_events]
                ))
        return memory_items

    all_memory_items = []
    if llm_model in ["gpt-4o-mini", "gpt-4.1-nano"]:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        for i in tqdm(range(0, len(events), 100), total=len(events)//100 + 1, desc=f"Clustering events into memory items with LLM={llm_model}"):
            events_to_cluster = events[i:i+100]
            system_message = LLM_INSTRUCTION
            user_message = "\n".join([
                "Event {}: CaptureType={}; Device={}; Location={}; Timestamp={}; Content={}; ".format(
                    i, event.capture_type, event.device, event.location, event.timestamp, event.content.replace("\n", " "))
                for i, event in enumerate(events_to_cluster)])
            llm_input_meesages = [
                {"role": "system", "content": system_message}, 
                {"role": "user", "content": user_message}
            ]
            response = client.chat.completions.create(
                model=llm_model,
                messages=llm_input_meesages,
            )
            llm_resp = response.choices[0].message.content
            memory_items = _parse_llm_resp_to_memory_items(events_to_cluster,llm_resp)
            all_memory_items.extend(memory_items)
    return all_memory_items

def AugmentMemoryItem(memory_item, llm_model):
    pass

def UpdateDatabase(memory_items, db_path, db_type):
    pass


from functools import partial
def main(args):
    all_memory_items = []
    for events in ReadEventsSliceByTime(args.input_db_path, args.input_db_type, args.timeslice_size):
        """Step 1: Filter & normalize events"""
        print("Before filtering len(events)= ", len(events))
        events = list(filter(FilterEvent, events))
        print("After filtering len(events)= ", len(events))

        """Step 3: Make memory items by LLM generative clustering"""
        memory_items = MakeMemByClustering(events, llm_model=args.llm_model)

        """Step 4: Augment memory items with LLM"""
        AugmentMemoryItemFnc = partial(AugmentMemoryItem, llm_model=args.llm_model)
        memory_items = map(AugmentMemoryItemFnc, memory_items)
        all_memory_items.extend(memory_items)

    """Step 5: Update database"""
    UpdateDatabase(all_memory_items, args.output_db_path, args.output_db_type)


if __name__ == "__main__":
    args = get_args()
    main(args)

