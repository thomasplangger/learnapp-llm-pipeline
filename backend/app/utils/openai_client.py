# app/utils/openai_utility.py
import os
import json
from typing import Tuple, Any, Dict
from fastapi import HTTPException
from openai import OpenAI

def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(500, "OpenAI API key not configured")
    return OpenAI(api_key=api_key)

def strip_markdown_json(content: str) -> str:
    """
    Strip any leading/trailing markdown so that only the JSON object remains.
    """
    start = content.find('{')
    end   = content.rfind('}')
    if start != -1 and end != -1:
        return content[start:end+1]
    return content

def chat_completion(
    messages: list[Dict[str, str]],
    model: str = "gpt-4o-mini",
    max_tokens: int = 500,
    temperature: float = 0.7
) -> str:
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return resp.choices[0].message.content.strip()

async def evaluate_text_answer(
    question: str,
    correct_answer: str,
    user_answer: str
) -> Tuple[int, str]:
    """
    Grade a free-form text answer on a 0–2 scale:
      - 2 if essentially fully correct
      - 1 if roughly half is correct
      - 0 if no part is correct

    Return JSON with:
      { "score": 0|1|2,
        "explanation": "Correct answer: …; Student's answer error: …"
      }
    """
    prompt = f"""
You are an expert tutor.
Question: "{question}"
Correct answer: "{correct_answer}"

A student answered: "{user_answer}"

1) Assign a score out of 2:
   - 2 if their answer is essentially fully correct.
   - 1 if roughly half is correct.
   - 0 if no part is correct.

2) Produce **valid JSON only** with two keys:
{{
  "score": 0|1|2,
  "explanation": "Correct answer: {correct_answer}; Student's answer error: …"
}}

The "explanation" must begin by restating the correct answer, then clearly point out what was wrong in the student's answer.
""".strip()

    raw = chat_completion(
        [
            {"role": "system", "content": "You are a helpful tutor and grader."},
            {"role": "user",   "content": prompt}
        ],
        max_tokens=200,
        temperature=0
    )

    clean = strip_markdown_json(raw)
    try:
        data: Dict[str, Any] = json.loads(clean)
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"Invalid JSON from grading AI: {e}")

    return data["score"], data["explanation"]
