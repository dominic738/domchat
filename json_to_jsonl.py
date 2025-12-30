import json

with open("data/qa_data.json", "r", encoding="utf-8") as f:
    qa_pairs = json.load(f)

with open("data/fine_tune.jsonl", "w", encoding="utf-8") as out:
    for qa in qa_pairs:
        out.write(json.dumps({
            "messages": [
                {"role": "user", "content": qa["question"]},
                {"role": "assistant", "content": qa["answer"]}
            ]
        }) + "\n")

print(f"Wrote {len(qa_pairs)} examples")