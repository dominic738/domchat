from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

file = client.files.create(
    file = open("data/fine_tune.jsonl", "rb"),
    purpose='fine-tune'
)

print(file.id)

job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-4.1-mini-2025-04-14"
)

print(job.id)

with open("data/job_id.txt", "w") as f:
    f.write(job.id)