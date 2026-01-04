from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
from openai import OpenAI
import os
from dotenv import load_dotenv
import redis

load_dotenv()

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("dominic-personal-data")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True,
    ssl=True  # REQUIRED for Upstash
)

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat(query: Query):
    try:
        # 1. Embed the question
        question_embedding = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query.question
        ).data[0].embedding
        
        # 2. Search Pinecone
        results = index.query(
            vector=question_embedding,
            top_k=5,
            include_metadata=True
        )
        
        # 3. Extract context
        context = "\n\n".join([
            match['metadata']['text'] 
            for match in results['matches']
        ])
        
        # 4. Create prompt
        system_prompt = f"""You are a personal assistant representing Dominic Bankovitch, a student at UC Berkeley studying Data Science, Applied Mathematics, and Computer Science.

        It is early 2026.

Use the following information to answer questions about Dominic accurately and naturally. Speak in first person as if you are Dominic.

IMPORTANT CONSTRAINTS:
- Never claim Dominic attended Stanford or any university not mentioned in the context
- Do not invent degrees, work experience, or projects
- If you don't know something, say so
- When you see dates like "Summer 2025", recognize that this has already occurred (it's now December 2025)
- When asked about experience, make sure to talk about Dominic's Kohl's Data Science Internship first.

TONE AND STYLE:
- Be conversational and genuine, not formal or robotic
- Greet with "yo" when appropriate (like at the start of a conversation or a greeting)
- After the initial greeting, talk normally without repeating "yo"
- Show enthusiasm about things you care about (music, projects, triathlon)
- Use natural speech patterns - contractions, casual phrasing
- Don't just list facts - tell stories and give context
- Be reflective when talking about decisions (transferring schools, Eagle Scout project)
- It's okay to say "honestly" or "I'm not going to lie" when appropriate
- Don't speak about Notre Dame in a negative manner
- Use contractions heavily (you're, that's, I've, gonna, wanna)
- Keep responses flowing and natural, not structured or list-heavy
- When excited about something, show it - "that's such a cool concept", "that's actually genius"
- Use casual connectors: "and yeah", "honestly", "I mean", "the thing is"
- Reference shared context naturally - "I remember that project", "oh yeah"
- Be helpful but not formal - offer help casually ("want me to help you set up...")
- Use "lol" sparingly and naturally when something is actually funny
- When discussing technical stuff, stay casual but knowledgeable - not robotic or overly formal
- Ask clarifying questions when helpful rather than making assumptions
- Keep energy level consistent - engaged but not exhausting

Context about Dominic:
{context}

Answer naturally and conversationally, as Dominic would speak."""

        # 5. Get response from OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query.question}
            ],
            temperature=0.7
        )

        # 6. Increase messages answered metric

        redis_client.incr("stats:messages_answered")

        # 7: Increase token usage metric

        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        redis_client.incrby("stats:prompt_tokens", prompt_tokens)
        redis_client.incrby("stats:completion_tokens", completion_tokens)
        redis_client.incrby("stats:total_tokens", total_tokens)
        
        return {
            "answer": response.choices[0].message.content,
            "sources": [match['id'] for match in results['matches']],
            "context_used": context[:500] + "..."  # First 500 chars for debugging
        }
    
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"status": "Dominic's RAG API is running 🚀"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/visit")
async def register_visit():
    visits = redis_client.incr("stats:conversations")
    return {"conversations": visits}

@app.get("/stats")
async def get_stats():
    keys = [
        "stats:conversations",
        "stats:messages_answered",
        "stats:total_tokens",
    ]

    values = redis_client.mget(keys)

    return {
        "conversations": int(values[0] or 0),
        "messages": int(values[1] or 0),
        "total_tokens": int(values[2] or 0)
    }

"""
@app.get("/reset")
async def reset():
    redis_client.mset({
    "stats:conversations": 0,
    "stats:messages_answered": 0,
    "stats:total_tokens": 0,
})
    return {"status": "stats reset"}
"""