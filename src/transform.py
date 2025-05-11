import re
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass
from src.logging_conf import logger
import pandas as pd


@dataclass
class Profile:
    linkedin_url: str
    title: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    description: Optional[str] = None
    profile_image_url: Optional[str] = None
    followers_count: Optional[int] = None
    contact_status: Optional[str] = "Not contacted"  # Default contact status
    connection_state: Optional[str] = "NOT_CONNECTED"  # Default connection state
    provider_id: Optional[str] = None  # Unipile-specific provider ID
    connection_msg: Optional[str] = ""  # Connection message
    comment_msg: Optional[str] = ""  # Comment message
    followup1: Optional[str] = ""  # First follow-up message
    followup2: Optional[str] = ""  # Second follow-up message
    followup3: Optional[str] = ""  # Third follow-up message
    inmail: Optional[str] = ""  # InMail message
    location: Optional[str] = None
    company: Optional[str] = None
    
    def __post_init__(self):
        """Ensure consistent default values for key fields"""
        # Handle missing or inconsistent contact_status
        if not self.contact_status or pd.isna(self.contact_status):
            self.contact_status = "Not contacted"
            
        # Handle missing or inconsistent connection_state
        if not self.connection_state or pd.isna(self.connection_state):
            self.connection_state = "NOT_CONNECTED"
            
        # Normalize connection_state (always uppercase)
        if isinstance(self.connection_state, str):
            self.connection_state = self.connection_state.upper()
            
        # Ensure consistent handling of new profiles
        if self.connection_state == "NOT_CONNECTED" and self.contact_status == "Profile Discovered":
            self.contact_status = "Not contacted"


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
    
    # Extract company if available from structured data or title
    company = ""
    if "pagemap" in item:
        if "person" in item["pagemap"] and item["pagemap"]["person"]:
            person_data = item["pagemap"]["person"][0]
            company = person_data.get("worksfor") or person_data.get("affiliation") # More person-specific fields
        elif "jobposting" in item["pagemap"] and item["pagemap"]["jobposting"]:
            job_data = item["pagemap"]["jobposting"][0]
            company = job_data.get("hiringorganization", {}).get("name") or job_data.get("name") # For job postings, company is often here
    
    # Fallback: Try to extract company from the title string (e.g., "Job Title at Company")
    if not company and title:
        # Common patterns: " at ", " @ ", " - " (if followed by company-like words)
        match_at = re.search(r'\s+at\s+([\w\s.&-]+?)(?:\s+-|\s+\||$)', title, re.IGNORECASE)
        match_hyphen_company = re.search(r'-\s+([\w\s.&-]+?)(?:\s+\||$)', title, re.IGNORECASE) # e.g. Director - Google
        
        if match_at:
            company_candidate = match_at.group(1).strip()
            # Avoid capturing very long strings or job descriptions if 'at' is used within them
            if len(company_candidate.split()) <= 4: # Heuristic: company names are usually short
                company = company_candidate
        elif match_hyphen_company and not name_parts.get("last_name", "").endswith(match_hyphen_company.group(1).strip()): # Ensure it's not part of the name
             company_candidate = match_hyphen_company.group(1).strip()
             if len(company_candidate.split()) <= 4:
                company = company_candidate
    
    # Clean up company name from common suffixes like " Inc.", ", LLC" if they are part of a larger non-company string
    if company:
        company = re.sub(r'(?i)(?:,\s*Inc\.|\s+Inc\.|\s*LLC|\s*Ltd\.|\s*GmbH)$', '', company).strip()
    
    return Profile(
        linkedin_url=linkedin_url,
        title=title,
        first_name=name_parts["first_name"],
        last_name=name_parts["last_name"],
        description=description,
        profile_image_url=profile_image_url,
        followers_count=0,  # Initialize to 0, will be updated with real data if available
        location=location,
        company=company.strip() or None,  # Ensure it's None if empty after stripping
        contact_status="Not contacted",  # Explicitly set default status
        connection_state="NOT_CONNECTED"  # Explicitly set default connection state
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
    # Try to find location in title first, as it's often more structured
    # Pattern: "Name - Title at Company (Location)" or "Name | Title | Location"
    if title:
        match_parentheses = re.search(r'\(([^)]+)\)$', title) # Location in parentheses at the end
        if match_parentheses:
            loc_candidate = match_parentheses.group(1).strip()
            # Basic check to avoid capturing job descriptions or non-locations
            if 1 < len(loc_candidate.split(',')) <= 3 or (len(loc_candidate.split()) <= 4 and not any(verb in loc_candidate.lower() for verb in ["hiring", "remote", "contract"])):
                return loc_candidate

        # Try splitting by "|" and check the last part if it looks like a location
        title_parts = [part.strip() for part in title.split('|')]
        if len(title_parts) > 1:
            potential_loc = title_parts[-1]
            # Heuristic: locations are often 1-3 words, or contain a comma (City, State/Country)
            if (1 <= len(potential_loc.split()) <= 4 and not any(c.isdigit() for c in potential_loc)) or ',' in potential_loc:
                # Further check to avoid job titles like "Greater Area Manager"
                if not any(kw.lower() in potential_loc.lower() for kw in ["manager", "director", "engineer", "specialist", "lead"]):
                     # Check if it's part of the "First Name Last Name | Title | Company | Location" pattern
                    if len(title_parts) >= 3 and potential_loc.lower() != title_parts[-2].lower() and potential_loc.lower() != title_parts[0].lower() :
                        return potential_loc

    # Try to find location in description (with improved pattern matching)
    if description:
        location_indicators = [
            " in ", " based in ", " located in ", " from " # more specific with spaces
        ]
        
        # Regex for common location patterns
        # Matches "City, State", "City, Country", "Region Area"
        location_regex = r'\b(?:[A-Z][\w\s-]+,\s*(?:[A-Z][\w\s-]+(?:\s*Area)?|[A-Z]{2,}))|(?:Greater\s+[A-Z][\w\s-]+(?:\s*City)?(?:\s*Area)?|[A-Z][\w\s-]+\s+Area)\b'
        
        match = re.search(location_regex, description)
        if match:
            return match.group(0).strip()

        for indicator in location_indicators:
            if indicator.lower() in description.lower(): # case-insensitive search
                parts = description.lower().split(indicator.lower(), 1)
                if len(parts) > 1:
                    # Take text after indicator, split by sentence end or common separators
                    location_part = re.split(r'[.,;!?)("]', parts[1], 1)[0].strip()
                    # Filter out very long strings or things that don't look like locations
                    if 0 < len(location_part) < 50 and not any(kw.lower() in location_part for kw in ["experience", "responsibilities", "skills"]):
                        # Capitalize words in the location
                        return ' '.join(word.capitalize() for word in location_part.split())
    
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