import json, os
from typing import Dict, List
from google import genai
from google.genai import types
from src.logging_conf import logger
from src.transform import Profile  # Import the Profile class

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

3. CONNECTION REQUEST (Strictly 120 chars max):
   - Must establish immediate relevance ("I noticed your work on...")
   - Include a specific, personalized detail they'll recognize from their background 
   - Explain the value/reason for connecting beyond just "networking"
   - ABSOLUTELY NO sales language or generic statements.
   - The 120-character limit is a TECHNICAL CONSTRAINT. Adhere to it strictly.

4. COMMENT APPROACH:
   - If a recent post is provided in the input, respond to the SPECIFIC CONTENT of that post with thoughtful insight.
   - Add valuable perspective, relevant experience, or a thought-provoking question related to their post.
   - Avoid generic praise like "Great post!" - be substantive.
   - If no recent post is available (recent_post field is empty or null), craft a comment referencing their profile headline, recent achievements, or a key aspect of their current role or company instead. Make it engaging.

5. FOLLOW-UP STRATEGY (3 distinct messages required):
   - ALL THREE follow-up messages MUST be generated. Do not omit any.
   - Each follow-up MUST provide NEW value, insight, or a gentle relationship-building touchpoint.
   - DO NOT use bracketed placeholders like "[Link to relevant article]" or "[mention a specific area]". Instead, if suggesting an article, describe the *topic* of such an article (e.g., "I recalled an insightful article discussing the future of AI in product operations..."). If asking about an area, be specific based on their profile (e.g., "Curious about your experience with scaling product teams in the fintech sector...").
   - Follow-up 1: Gentle value (e.g., share a thought on a relevant trend, mention a conceptual article topic).
   - Follow-up 2: Slightly deeper engagement (e.g., ask a thoughtful question based on their role/industry, share another conceptual insight).
   - Follow-up 3: If appropriate, suggest a very low-pressure next step (e.g., "would be open to a brief virtual coffee to exchange perspectives if your schedule ever allows").
   - Ensure messages are distinct and progressively build the connection.

6. INMAIL (Subject and Body required):
   - Both `inmail_subject` and `inmail_body` MUST be generated.
   - Subject: Personalized, concise, and attention-grabbing (max 70 chars).
   - Body: Professional but warm, clearly state the value proposition or reason for reaching out (max 500 chars). Focus on mutual interest or learning. Avoid being overly salesy.

You MUST return a valid JSON object with EXACTLY these keys and non-empty string values for all messages:
{{{{
 "connection": "<120‑char hyper-personalized connection request>",
 "comment": "<substantive reply (if no post, reference profile headline/role)>",
 "followups": ["<value-adding follow-up 1>", "<relationship-building follow-up 2>", "<action-oriented follow-up 3>"],
 "inmail_subject": "<personalized, attention-grabbing subject>",
 "inmail_body": "<personalized message up to 500 chars>"
}}}}
All text values in the JSON must be complete, professional, and ready for direct use.
"""

def craft_messages(profile_data: Profile, recent_post:str="") -> Dict:
    """
    Craft hyper-personalized LinkedIn messages based on profile data and recent post content.
    
    Args:
        profile_data: Profile object containing LinkedIn profile information
        recent_post: Text of recipient's recent LinkedIn post (if available)
        
    Returns:
        Dictionary with personalized connection request, comment, follow-ups, and InMail
    """
    # Access fields using attributes
    first_name = profile_data.first_name or ""
    last_name = profile_data.last_name or ""
    headline = profile_data.title or ""
    industry = ""  # Profile model doesn't have a dedicated 'industry' field currently
    company = profile_data.company or ""
    
    # Prepare profile dictionary for the prompt
    # Convert Pydantic model to dict for JSON serialization
    profile_dict_for_prompt = profile_data.model_dump(exclude_none=True)

    input_data = {
        "profile": profile_dict_for_prompt,
        "recent_post": recent_post,
        "first_name": first_name,
        "last_name": last_name,
        "industry_context": industry,
        "current_role": headline,
        "skills": [],  # Profile model doesn't have 'skills' field
        "company": company,
        "provider_id": getattr(profile_data, 'provider_id', ''),  # Safely access if it exists
        "followers_count": profile_data.followers_count or 0
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
                    logger.warning(f"Attempt {attempt+1}: Could not find JSON in response. Response text: {text[:200]}...")
        except json.JSONDecodeError as e:
            logger.warning(f"Attempt {attempt+1}: JSON parse error: {e}. Response text: {text[:200]}...")
            if attempt == 2:  # Last attempt
                logger.error(f"Failed to parse JSON after 3 attempts: {e}")
                # Fall through to fallback to avoid raising here, campaign can log error
        except Exception as e:  # Catch other potential API errors
            logger.warning(f"Attempt {attempt+1}: Error calling Gemini API: {str(e)}")
            if attempt == 2:
                logger.error(f"Failed to call Gemini API after 3 attempts: {str(e)}")
    
    # Fallback response
    logger.warning(f"All attempts to craft messages via Gemini failed for profile: {profile_data.linkedin_url}. Using fallback.")
    
    # Build industry/role info for fallback messages
    role_info = headline.split('|')[0].strip() if headline and '|' in headline else headline or 'your field'
    industry_or_role = industry or role_info
    
    return {
        "connection": f"Noticed your work in {industry_or_role}. I'm exploring similar challenges - would love to connect!",
        "comment": f"Really enjoyed reading about your work in {role_info}! The approach you're taking is inspiring.",
        "followups": [
            f"Hope you're having a great week, {first_name}! Wanted to share that I've been thinking about {role_info} challenges lately, particularly around optimization and efficiency.",
            f"Hi {first_name}, I recently came across some interesting insights on trends affecting {company or 'companies in your sector'}. Would love to hear your perspective on how you're navigating these changes.",
            f"Hi {first_name}, if your schedule allows, I'd value a brief conversation to exchange ideas on {industry_or_role}. I've been working on some approaches that might align with your interests."
        ],
        "inmail_subject": f"Regarding your expertise in {industry_or_role}",
        "inmail_body": f"Hi {first_name or 'there'},\n\nI noticed your experience with {headline or 'your current role'} at {company or 'your company'}. I'm working on similar challenges in the {industry_or_role} space and would value your perspective.\n\nWould you be open to a brief conversation about how you've approached these challenges? I'd be happy to share my insights as well.\n\nLooking forward to potentially connecting,\n\nYour Name"
    } 