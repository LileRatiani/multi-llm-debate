import json
import re
import time
from typing import Optional, Type

import groq
from groq import Groq
from pydantic import BaseModel

from src.config import Config

groq_client = Groq(api_key=Config.GROQ_API_KEY) if Config.GROQ_API_KEY else None


def clean_json_string(raw_str: str) -> str:
    """Removes markdown code block wrappers and returns the cleaned JSON string."""
    raw_str = raw_str.strip()
    marker = "`" * 3

    if raw_str.startswith(marker):
        raw_str = re.sub(r"^" + marker + r"(?:json)?\s*\n", "", raw_str, flags=re.IGNORECASE)
        raw_str = re.sub(r"\n\s*" + marker + r"$", "", raw_str)

    return raw_str.strip()


def repair_json_string(raw_str: str) -> str:
    """Attempt to recover valid JSON from malformed model output."""
    raw_str = clean_json_string(raw_str)
    if not raw_str:
        return raw_str

    try:
        json.loads(raw_str)
        return raw_str
    except json.JSONDecodeError:
        pass

    start = raw_str.find("{")
    if start == -1:
        return raw_str

    candidate = raw_str[start:]
    for end in range(len(candidate), 0, -1):
        snippet = candidate[:end].rstrip()
        if not snippet:
            continue
        try:
            json.loads(snippet)
            return snippet
        except json.JSONDecodeError:
            continue

    return raw_str


def extract_failed_generation(error: groq.APIStatusError) -> Optional[str]:
    """Pull partially generated JSON from Groq json_validate_failed errors."""
    response = getattr(error, "response", None)
    if response is None:
        return None

    try:
        payload = response.json()
    except Exception:
        return None

    return payload.get("error", {}).get("failed_generation")


def generate_json_response(
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    response_schema: Optional[Type[BaseModel]] = None,
    temperature: float = 0.7,
    max_retries: int = 3,
) -> str:
    """Sends a chat completion request to Groq and returns cleaned JSON text."""

    if not groq_client:
        raise ValueError("Groq client is missing API key configuration.")

    actual_model = model_name.replace("groq/", "")
    final_user_prompt = user_prompt

    if response_schema:
        schema_fields = response_schema.model_fields
        schema_desc = ", ".join([
            f"'{name}' ({field.annotation.__name__ if hasattr(field.annotation, '__name__') else str(field.annotation)})"
            for name, field in schema_fields.items()
        ])
        final_user_prompt += f"\n\nYou MUST return your output strictly as a JSON object matching this schema: {{{schema_desc}}}"
    else:
        final_user_prompt += "\n\nYou MUST return your output strictly as a JSON object."

    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                model=actual_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": final_user_prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            return clean_json_string(response.choices[0].message.content)

        except groq.RateLimitError as e:
            resp = getattr(e, "response", None)
            retry_after = float(resp.headers.get("retry-after", 10)) if resp else 10
            if attempt < max_retries - 1:
                print(f"   [Rate Limit] {actual_model} hit its quota. Retrying in {retry_after:.0f}s... (attempt {attempt + 2}/{max_retries})")
                time.sleep(retry_after)
            else:
                print(f"   [Rate Limit] {actual_model} exhausted retries: {e}")
                return "{}"

        except groq.APIStatusError as e:
            failed_generation = extract_failed_generation(e)
            if failed_generation:
                repaired = repair_json_string(failed_generation)
                try:
                    json.loads(repaired)
                    print(f"   [Warning] Recovered malformed JSON from {actual_model}")
                    return repaired
                except json.JSONDecodeError:
                    pass

            if e.status_code >= 500 and attempt < max_retries - 1:
                wait_time = 15
                print(f"   [Server Error {e.status_code}] {actual_model} unavailable. Retrying in {wait_time}s... (attempt {attempt + 2}/{max_retries})")
                time.sleep(wait_time)
            elif e.status_code == 400 and attempt < max_retries - 1:
                print(f"   [JSON Error] {actual_model} returned invalid JSON. Retrying... (attempt {attempt + 2}/{max_retries})")
                time.sleep(2)
            else:
                print(f"   [Error] API call failed for model {actual_model}: {e}")
                return "{}"

        except groq.APIConnectionError as e:
            print(f"   [Connection Error] Could not reach Groq for {actual_model}: {e}")
            return "{}"

    return "{}"