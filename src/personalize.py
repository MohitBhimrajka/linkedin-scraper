import json, os
from typing import Dict, List
from google import genai
from google.genai import types
from src.logging_conf import logger

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash-preview-04-17"

PROMPT_TMPL = """
You are a friendly, insightful sales‑navigator bot.

INPUT:
---------
{profile_json}
---------

Return STRICTLY the following JSON:
{
 "connection": "<120‑char personalised connection line>",
 "comment": "<reply to a recent post if post provided else empty>",
 "followups": ["<FU1>", "<FU2>", "<FU3>"],
 "inmail_subject": "<subject>",
 "inmail_body": "<body up to 500 chars>"
}
No additional keys.
"""

def craft_messages(profile_data:Dict, recent_post:str="") -> Dict:
    pj = json.dumps({"profile":profile_data,"recent_post":recent_post},ensure_ascii=False,indent=2)
    prompt = PROMPT_TMPL.format(profile_json=pj)
    res = client.models.generate_content(
        model=MODEL,
        contents=[types.Content(role="user",parts=[types.Part.from_text(prompt)])],
        config=types.GenerateContentConfig(response_mime_type="text/plain"))
    try:
        return json.loads(res.text)
    except json.JSONDecodeError as e:
        logger.error(f"Gemini JSON parse error: {e}")
        raise 