import json, os
from typing import Dict, List
from google import genai
from google.genai import types
from src.logging_conf import logger

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash-preview-04-17"

PROMPT_TMPL = """
You are an expert LinkedIn relationship builder with exceptional skills in personalized outreach that feels authentic, thoughtful, and genuinely valuable to the recipient. 

LinkedIn PROFILE INPUT:
---------
{profile_json}
---------

Your task is to craft hyper-personalized messages that demonstrate you've genuinely analyzed this person's specific background, achievements, content, and interests. Follow these critical principles:

1. RESEARCH-BASED PERSONALIZATION:
   - Reference specific details from their profile (skills, experience, education, etc.)
   - If they've shared a recent post, engage DIRECTLY with the substance of that content
   - Mention relevant industry trends or challenges specific to their role/industry
   - Reference their company's recent news or achievements when appropriate

2. AUTHENTICITY & VALUE FIRST:
   - Start with genuine appreciation, insight, or shared interest - NEVER with a sales pitch
   - Avoid generic templates or obvious flattery that could apply to anyone
   - Focus on providing value or insight before asking for anything
   - Write in a natural, conversational tone appropriate for professional context

3. CONNECTION REQUEST (120 chars max):
   - Must establish immediate relevance ("I noticed your work on...")
   - Include a specific, personalized detail they'll recognize from their background 
   - Explain the value/reason for connecting beyond just "networking"
   - ZERO sales language or generic statements in the initial connection

4. COMMENT APPROACH:
   - Respond to the SPECIFIC CONTENT they posted with thoughtful insight
   - Add valuable perspective, relevant experience, or a thought-provoking question
   - Avoid generic praise like "Great post!" - be substantive

5. FOLLOW-UP STRATEGY:
   - Each follow-up should provide NEW value or perspective (industry insights, relevant resources)
   - Progressively deepen the relationship with each message
   - Only the final follow-up should suggest a specific next step or meeting
   - Space messages appropriately (don't appear desperate)

6. FORMAT REQUIREMENTS:
   - Connection request: Conversational, genuinely interesting, max 120 chars
   - Comment: Thoughtful response to their content, adding value
   - Follow-ups: Three distinct messages that build relationship
   - InMail: Professional but warm, clear value proposition

You MUST return a valid JSON object with EXACTLY these keys:
{
 "connection": "<120‑char hyper-personalized connection request>",
 "comment": "<substantive, thoughtful reply to recent post or empty if no post>",
 "followups": ["<value-adding follow-up 1>", "<relationship-building follow-up 2>", "<action-oriented follow-up 3>"],
 "inmail_subject": "<personalized, attention-grabbing subject>",
 "inmail_body": "<personalized message up to 500 chars>"
}
"""

def craft_messages(profile_data:Dict, recent_post:str="") -> Dict:
    """
    Craft hyper-personalized LinkedIn messages based on profile data and recent post content.
    
    Args:
        profile_data: Dictionary containing LinkedIn profile information
        recent_post: Text of recipient's recent LinkedIn post (if available)
        
    Returns:
        Dictionary with personalized connection request, comment, follow-ups, and InMail
    """
    # Enrich the input data to provide more context for personalization
    input_data = {
        "profile": profile_data,
        "recent_post": recent_post,
        # Add industry context if available
        "industry_context": profile_data.get("industry", ""),
        # Add current role details if available
        "current_role": profile_data.get("headline", "") or profile_data.get("title", ""),
        # Extract any skills if available
        "skills": profile_data.get("skills", []),
        # Extract company information if available
        "company": profile_data.get("company", profile_data.get("companyName", ""))
    }
    
    # Create well-formatted JSON for the prompt
    pj = json.dumps(input_data, ensure_ascii=False, indent=2)
    prompt = PROMPT_TMPL.format(profile_json=pj)
    
    # Configure Gemini for more thoughtful responses
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )
    
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]
    
    # Make 3 attempts in case of JSON parsing failures
    for attempt in range(3):
        try:
            res = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=generate_content_config,
            )
            
            # Extract just the JSON part using a simple heuristic if needed
            text = res.text.strip()
            if text.startswith("{") and text.endswith("}"):
                # Already proper JSON
                return json.loads(text)
            else:
                # Try to extract JSON if there's additional text
                import re
                json_match = re.search(r'\{[\s\S]*\}', text)
                if json_match:
                    return json.loads(json_match.group(0))
                else:
                    logger.warning(f"Attempt {attempt+1}: Could not find JSON in response")
        except json.JSONDecodeError as e:
            if attempt < 2:  # If not the last attempt
                logger.warning(f"Attempt {attempt+1}: JSON parse error: {e}. Retrying...")
            else:
                logger.error(f"Failed to parse JSON after 3 attempts: {e}")
                raise
    
    # Fallback response if all attempts fail but don't raise an exception
    return {
        "connection": f"Noticed your work in {profile_data.get('industry', 'your industry')}. I'm exploring similar challenges - would love to connect!",
        "comment": "Insightful perspective! I've been thinking about this topic recently and appreciate your take.",
        "followups": [
            "Hope you're having a great week! Wanted to share this article that relates to your recent post.",
            "Just came across this resource that might interest you given your background.",
            "Would value your perspective on a project I'm working on - any chance we could chat briefly?"
        ],
        "inmail_subject": f"Quick question about {profile_data.get('industry', 'your industry')} approach",
        "inmail_body": f"Hi {profile_data.get('firstName', 'there')}, I noticed your experience with {profile_data.get('headline', 'your current role')}. I'm working on similar challenges and would love to learn from your approach. Would you be open to a brief conversation about how you've navigated this space? I'm happy to share my insights as well."
    } 