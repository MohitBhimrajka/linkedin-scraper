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
    
    # Extract name parts if available
    name_parts = parse_name(metatags.get("profile:first_name", ""), 
                           metatags.get("profile:last_name", ""),
                           title)
    
    # Extract description (prefer metatags)
    description = (metatags.get("og:description") or 
                  metatags.get("description") or 
                  item.get("snippet", ""))
    
    # Extract profile image (prefer twitter:image over og:image)
    profile_image_url = metatags.get("twitter:image") or metatags.get("og:image")
    
    # Find profile images from pagemap if not in metatags
    if not profile_image_url and "pagemap" in item:
        if "cse_image" in item["pagemap"]:
            profile_image_url = item["pagemap"]["cse_image"][0].get("src")
        elif "cse_thumbnail" in item["pagemap"]:
            profile_image_url = item["pagemap"]["cse_thumbnail"][0].get("src")
    
    return Profile(
        linkedin_url=linkedin_url,
        title=title,
        first_name=name_parts["first_name"],
        last_name=name_parts["last_name"],
        description=description,
        profile_image_url=profile_image_url,
        followers_count=0  # Initialize to 0, will be updated with real data if available
    )


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