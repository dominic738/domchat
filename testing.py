from openai import OpenAI
from dotenv import load_dotenv
import os
import time


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

with open("data/job_id.txt") as f:
    job_id = f.read().strip()


model_id = os.getenv("MODEL_V1")

print("USING MODEL:", model_id)


response = client.chat.completions.create(
    model = model_id,
    messages = [
        {
            "role" : "system", 
            "content" : "You are Dominic Bankovitch. Answer casually and honestly, like on your personal website. Do not invent a different name, school, or background."
        },
        {
            "role": "user",
            "content": "Hi, what's your name? Tell me a little bit about yourself."
        }
    ]
)

print(response.choices[0].message.content)
