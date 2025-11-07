import asyncio
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from app.core.settings import settings
from app.core.db import AsyncSessionLocal
from app.models import Chunk

from langchain_openai import ChatOpenAI
from sqlalchemy import text
import json

PROMPT = """Generate 3 test queries for a RAG system that helps LLMs write code.

Documentation chunk:
{chunk}

Generate 3 queries from a CODE GENERATION perspective - queries an LLM would ask when implementing functionality:

Requirements:
- Focus on HOW TO IMPLEMENT specific features/functions
- Ask about code usage, parameters, syntax, patterns
- Must be answerable ONLY from information in this chunk
- Avoid meta-questions about documentation structure (no "What section...", "What does X include...")
- Under 15 words
- Natural developer intent, as if asking "how do I code this?"

Good examples:
- "How do I create a retrieval chain with custom parameters?"
- "What parameters does the Agent constructor accept?"
- "How do I handle errors in async retrieval functions?"

Bad examples (avoid these):
- "What does the Show Full example code section include?" (meta/navigational)
- "What is Step 1 to build an agent?" (procedural/tutorial structure)
- "What topics does the quickstart list?" (documentation structure)

Format: Questions starting with "How do I", "What parameters", "How can I", or "What's the syntax"
"""

class OutputSchema(BaseModel):
    questions : List[str] = Field(..., description="Questions that an llm could ask related to the chunk")

async def generate_test_cases(model: ChatOpenAI, chunk: Chunk):
    prompt = ChatPromptTemplate.from_template(PROMPT)
    chain = prompt | model
    result = await chain.ainvoke({"chunk": chunk})
    return {
        "id": str(chunk[0]),
        # "content": chunk[3],
        "questions": result.questions
    }

async def get_random_chunks(num_chunks: int) -> List[Chunk]:
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            text(f"SELECT * FROM chunks TABLESAMPLE BERNOULLI(30) LIMIT :limit"),
            {"limit": num_chunks}
        )
        chunks = result.fetchall()
    return chunks

async def main():
    model = ChatOpenAI(
        model='gpt-5',
        api_key=settings.OPENAI_API_KEY
    ).with_structured_output(OutputSchema)

    chunks = await get_random_chunks(30)
    tasks = [asyncio.create_task(generate_test_cases(model,chunk)) for chunk in chunks]
    results = await asyncio.gather(*tasks)
    with open("test_chunks.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2 )



if __name__ == "__main__":

   asyncio.run(main())
