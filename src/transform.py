from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from src.logging_conf import logger


class Profile(BaseModel):
    """
    Pydantic model for a LinkedIn profile.
    
    Represents the structured data extracted from Google CSE results.
    """
    linkedin_url: str
    title: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    description: Optional[str] = None
    profile_image_url: Optional[str] = None
    followers_count: Optional[int] = 0  # Added followers count field
    connection_state: Optional[str] = "NOT_CONNECTED"  # Default connection state
    contact_status: Optional[str] = "Not contacted"  # Default contact status
    connection_msg: Optional[str] = ""  # Connection message
    comment_msg: Optional[str] = ""  # Comment message
    followup1: Optional[str] = ""  # First follow-up message
    followup2: Optional[str] = ""  # Second follow-up message
    followup3: Optional[str] = ""  # Third follow-up message
    inmail: Optional[str] = ""  # InMail message
    location: Optional[str] = None
    company: Optional[str] = None


def normalize_results(raw_items: List[Dict]) -> List[Profile]:
    """
    Extract and normalize profile data from Google CSE results.
    
    Args:
        raw_items: Raw items from Google Custom Search API
        
    Returns:
        List of normalized Profile objects with deduplicated URLs
    """
    profiles = []
    seen_urls = set()
    
    for item in raw_items:
        # Get the LinkedIn URL
        linkedin_url = item.get("link", "")
        
        # Skip if not a LinkedIn URL or already processed
        if "linkedin.com/in/" not in linkedin_url or linkedin_url in seen_urls:
            continue
        
        seen_urls.add(linkedin_url)
        
        # Extract metatags if available
        metatags = {}
        if "pagemap" in item and "metatags" in item["pagemap"]:
            metatags = item["pagemap"]["metatags"][0]
        
        # Extract fields from metatags or fallback to default fields
        profile = extract_profile_data(item, metatags, linkedin_url)
        profiles.append(profile)
    
    logger.info(f"Normalized {len(profiles)} unique profiles from {len(raw_items)} results")
    return profiles


def extract_profile_data(item: Dict, metatags: Dict, linkedin_url: str) -> Profile:
    """
    Extract profile data from an item and its metatags.
    
    Args:
        item: Google CSE result item
        metatags: Extracted metatags from the item
        linkedin_url: LinkedIn profile URL
        
    Returns:
        Profile object with extracted data
    """
    # Extract title (prefer metatags)
    title = metatags.get("profile:title") or metatags.get("og:title") or item.get("title", "")
    
    # Clean the title - remove "LinkedIn" mentions and other common suffixes
    if title:
        for suffix in [" | LinkedIn", " - LinkedIn", " – LinkedIn", ", LinkedIn"]:
            if title.endswith(suffix):
                title = title[:-(len(suffix))]
                break
    
    # Extract name parts if available
    name_parts = parse_name(metatags.get("profile:first_name", ""), 
                           metatags.get("profile:last_name", ""),
                           title)
    
    # Extract description (prefer metatags)
    description = (metatags.get("og:description") or 
                  metatags.get("description") or 
                  item.get("snippet", ""))
    
    # Clean description - remove common LinkedIn phrases
    if description:
        for phrase in ["View my profile on LinkedIn", "See my complete profile on LinkedIn", 
                      "View my professional profile on LinkedIn", "Find jobs at companies"]:
            description = description.replace(phrase, "").strip()
    
    # Extract location from description or title if possible
    location = extract_location(description, title)
    
    # Extract profile image (prefer twitter:image over og:image)
    profile_image_url = metatags.get("twitter:image") or metatags.get("og:image")
    
    # Find profile images from pagemap if not in metatags
    if not profile_image_url and "pagemap" in item:
        if "cse_image" in item["pagemap"]:
            profile_image_url = item["pagemap"]["cse_image"][0].get("src")
        elif "cse_thumbnail" in item["pagemap"]:
            profile_image_url = item["pagemap"]["cse_thumbnail"][0].get("src")
    
    # Try to extract any additional structured data if available
    structured_data = {}
    if "pagemap" in item:
        if "person" in item["pagemap"] and item["pagemap"]["person"]:
            structured_data = item["pagemap"]["person"][0]
        elif "jobposting" in item["pagemap"] and item["pagemap"]["jobposting"]:
            structured_data = item["pagemap"]["jobposting"][0]
    
    # Extract company if available from structured data
    company = structured_data.get("name") or structured_data.get("worksfor") or ""
    
    return Profile(
        linkedin_url=linkedin_url,
        title=title,
        first_name=name_parts["first_name"],
        last_name=name_parts["last_name"],
        description=description,
        profile_image_url=profile_image_url,
        followers_count=0,  # Initialize to 0, will be updated with real data if available
        location=location,
        company=company
    )


def extract_location(description: str, title: str) -> Optional[str]:
    """
    Extract location from description or title if possible.
    
    Args:
        description: Profile description
        title: Profile title
        
    Returns:
        Location string if found, None otherwise
    """
    # Common location patterns in LinkedIn profiles
    location_indicators = [
        " in ", " at ", " from ", "located in", "based in", 
        "working in", "living in", "residing in"
    ]
    
    # Try to find location in description
    for indicator in location_indicators:
        if indicator in description.lower():
            parts = description.split(indicator, 1)
            if len(parts) > 1:
                location_part = parts[1].split(".")[0].split(",")[0].strip()
                # Only return if it looks like a location (not too long)
                if 2 <= len(location_part.split()) <= 5:
                    return location_part
    
    # Check if location is in title format "Name - Title at Company (Location)"
    if " - " in title and "(" in title and title.endswith(")"):
        location = title.split("(")[-1].rstrip(")")
        if len(location.split()) <= 5:  # Simple check that it's not too complex
            return location
            
    return None


def parse_name(first_name: str, last_name: str, title: str) -> Dict[str, Optional[str]]:
    """
    Parse name components from available data.
    
    Args:
        first_name: First name from metatags
        last_name: Last name from metatags  
        title: Title string that might contain the name
        
    Returns:
        Dictionary with first_name and last_name
    """
    result = {
        "first_name": None,
        "last_name": None
    }
    
    # If we have both parts from metatags, use them
    if first_name and last_name:
        result["first_name"] = first_name
        result["last_name"] = last_name
        return result
    
    # Try to extract from title (e.g., "John Doe | LinkedIn" or "John Doe - Software Engineer")
    if title:
        # Split by common separators and take the first part as the name
        for separator in [" | ", " - ", " – ", ": "]:
            if separator in title:
                name_part = title.split(separator)[0].strip()
                name_components = name_part.split()
                
                if len(name_components) >= 2:
                    result["first_name"] = name_components[0]
                    result["last_name"] = " ".join(name_components[1:])
                    return result
    
    return result 