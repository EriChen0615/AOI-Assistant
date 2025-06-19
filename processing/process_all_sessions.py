import json

sessions = []
with open('outputs/0618/dev1/all_sessions.jsonl', 'r') as f:
    for line in f:
        sessions.append(json.loads(line))

for session in sessions:
    print(session)
    breakpoint()
    break