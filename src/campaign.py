import asyncio, pytz, datetime as dt
from typing import List, Dict, Optional
from src.unipile_client import UnipileClient, UnipileAuthError
from src.personalize import craft_messages
from src.transform import Profile
from src.logging_conf import logger
import gspread
import httpx
import concurrent.futures
from functools import partial
import os
import traceback

class CampaignStats: 
    generated=0
    sent=0 
    errors=0
    skipped=0

# Define specific API error handling
class UnipileApiError(Exception):
    pass

async def run_campaign(
    profiles: List[Profile], 
    followup_days=(3,7,14), 
    mode="Full",
    spreadsheet_id: Optional[str] = None, 
    sheet_name: Optional[str] = None,
    schedule_time: Optional[str] = None,
    should_generate_messages: bool = True,
    max_parallel: int = 10
) -> CampaignStats:
    """
    Generate personalized messages and/or launch a campaign.
    
    Args:
        profiles: List of LinkedIn profile data to process
        followup_days: Tuple of days to wait for each follow-up
        mode: "Generate only" or "Full" campaign
        spreadsheet_id: Google Sheet ID for updating statuses
        sheet_name: Specific worksheet name
        schedule_time: ISO timestamp for scheduling
        should_generate_messages: Whether to generate messages or use existing ones
        max_parallel: Maximum number of concurrent operations
    
    Returns:
        CampaignStats object with counts of generated/sent messages
    """
    stats = CampaignStats()
    
    # Connect to Google Sheets API if IDs are provided
    worksheet = None
    gc = None
    if spreadsheet_id and sheet_name:
        try:
            # Try multiple possible locations for service account file
            service_account_paths = [
                "service-account.json",  # Root directory
                "config/service_account.json",  # Config directory
                "../service-account.json"  # Parent directory
            ]
            
            for path in service_account_paths:
                try:
                    if os.path.exists(path):
                        gc = gspread.service_account(filename=path)
                        logger.info(f"Connected to Google Sheets using service account at {path}")
                        break
                except Exception:
                    continue
            
            # If no service account file was found, try OAuth
            if gc is None:
                gc = gspread.oauth()
                logger.info("Connected to Google Sheets using OAuth")
            
            # Open the spreadsheet and specific worksheet
            spread = gc.open_by_key(spreadsheet_id)
            worksheet = spread.worksheet(sheet_name)
            logger.info(f"Connected to Google Sheet '{sheet_name}'")
        except Exception as e:
            logger.error(f"Failed to connect to Google Sheets: {e}")
            logger.debug(f"Google Sheets connection error details: {traceback.format_exc()}")
    
    # Create a Unipile client
    client = UnipileClient()
    
    # Create tasks for batch processing profiles
    tasks = []
    semaphore = asyncio.Semaphore(max_parallel)
    
    async def process_profile(p, idx):
        async with semaphore:
            row_idx = None
            try:
                logger.info(f"Processing profile {idx+1}/{len(profiles)}: {p.linkedin_url}")
                
                # 1. If message generation is needed, do that first
                if should_generate_messages:
                    # Generate personalized messages based on the profile data
                    # Pass recent_post if available and used by craft_messages
                    recent_post_text = recent # Assuming 'recent' is fetched post text
                    msgs = await asyncio.to_thread(craft_messages, p, recent_post=recent_post_text)
                    
                    # Update the profile object with the generated messages
                    p.connection_msg = msgs.get("connection", "")
                    p.comment_msg = msgs.get("comment", "")
                    
                    followups = msgs.get("followups", [])
                    p.followup1 = followups[0] if len(followups) > 0 else ""
                    p.followup2 = followups[1] if len(followups) > 1 else ""
                    p.followup3 = followups[2] if len(followups) > 2 else ""
                    
                    # Assign InMail subject and body
                    inmail_subject = msgs.get("inmail_subject", "")
                    inmail_body = msgs.get("inmail_body", "")
                    if inmail_subject and inmail_body:
                        p.inmail = f"Subject: {inmail_subject}\n\n{inmail_body}"
                    elif inmail_body: # If only body is present
                        p.inmail = inmail_body
                    else: # If neither or only subject (less useful alone)
                        p.inmail = ""
                        
                    stats.generated += 1
                else:
                    # Use existing messages from the profile (ensure all fields are covered)
                    msgs = {
                        "connection": p.connection_msg or "",
                        "comment": p.comment_msg or "",
                        "followups": [
                            p.followup1 or "",
                            p.followup2 or "",
                            p.followup3 or ""
                        ],
                        # Reconstruct InMail subject/body if they were stored separately
                        # or ensure p.inmail is directly usable
                        "inmail_subject": "", # Or parse from p.inmail if needed
                        "inmail_body": p.inmail or ""
                    }
                
                # 2. Find the row in the Google Sheet (if applicable)
                if worksheet:
                    try:
                        # Find the row with this LinkedIn URL
                        # Ensure gspread calls are threaded if worksheet is used in async context
                        # cell = await asyncio.to_thread(worksheet.find, p.linkedin_url)
                        cell = worksheet.find(p.linkedin_url) # Assuming worksheet ops are made thread-safe elsewhere or run sync
                        row_idx = cell.row
                        
                        # Update the message cells in the sheet
                        batch_updates = [
                            {"range": f"I{row_idx}", "values": [[p.connection_msg]]},
                            {"range": f"J{row_idx}", "values": [[p.comment_msg]]},
                            {"range": f"K{row_idx}", "values": [[p.followup1 or "—"]]},
                            {"range": f"L{row_idx}", "values": [[p.followup2 or "—"]]},
                            {"range": f"M{row_idx}", "values": [[p.followup3 or "—"]]},
                            {"range": f"N{row_idx}", "values": [[p.inmail or "—"]]} # Column N for InMail
                        ]
                        worksheet.batch_update(batch_updates)
                        logger.info(f"Updated messages for row {row_idx}")
                    except Exception as e:
                        logger.warning(f"Could not update sheet for {p.linkedin_url}: {e}")
                
                # Skip API calls if in generate-only mode
                if mode == "Generate only":
                    stats.skipped += 1
                    return
                
                # 3. Get the LinkedIn provider_id for API calls
                # Fetch profile data to get the provider_id
                logger.info(f"Fetching profile data for {p.linkedin_url}")
                pdata = await client.get_profile(p.linkedin_url)
                
                # Check if pdata is a Profile object or a dictionary
                if hasattr(pdata, 'provider_id'):
                    # Direct attribute access for Profile objects
                    provider_id = pdata.provider_id
                elif isinstance(pdata, dict):
                    # Dictionary access for API responses
                    provider_id = pdata.get("provider_id")
                    if not provider_id:
                        # Try camelCase variant which might be in the API response
                        provider_id = pdata.get("providerId")
                else:
                    # Handle unexpected data type
                    raise TypeError(f"Expected Profile object or dictionary, got {type(pdata)}")
                    
                if not provider_id:
                    raise ValueError(f"Could not find provider_id in profile data for {p.linkedin_url}")
                    
                logger.info(f"Retrieved provider_id: {provider_id}")
                
                # 4. Always fetch posts to provide data for comments
                try:
                    logger.info(f"Fetching recent posts for {provider_id}")
                    posts_data = await client.recent_posts(provider_id=provider_id, limit=1) # Renamed to posts_data
                    if not isinstance(posts_data, list):
                        logger.warning(f"Unexpected posts data structure received: {type(posts_data)}")
                        posts_data = []
                    recent = posts_data[0]["text"] if posts_data and len(posts_data) > 0 and "text" in posts_data[0] else "" # Assign to 'recent'
                except IndexError:
                    logger.warning(f"No posts found for {provider_id}")
                    recent = ""
                    posts_data = []
                except KeyError as e:
                    logger.warning(f"Missing key in post data for {provider_id}: {e}")
                    recent = ""
                    posts_data = []
                except Exception as e:
                    # Non-fatal error for posts
                    logger.warning(f"Failed to fetch posts for {provider_id}: {e}")
                    recent = ""
                    posts_data = []
                
                # 5. Send invitation with connection message
                try:
                    logger.info(f"Sending invitation to {provider_id}" + (f" scheduled for {schedule_time}" if schedule_time else ""))
                    await client.send_invitation(
                        provider_id=provider_id, 
                        message=msgs["connection"],
                        send_at=schedule_time
                    )
                    
                    # Update stats and sheet
                    stats.sent += 1
                    if worksheet and row_idx:
                        now = dt.datetime.now(dt.UTC).isoformat()
                        status_updates = [
                            {"range": f"M{row_idx}", "values": [["Invited"]]},
                            {"range": f"N{row_idx}", "values": [[now]]},
                        ]
                        worksheet.batch_update(status_updates)
                except Exception as e:
                    # If sending fails but message generation succeeded
                    logger.error(f"Failed to send invitation to {provider_id}: {e}")
                    if worksheet and row_idx:
                        now = dt.datetime.now(dt.UTC).isoformat()
                        error_updates = [
                            {"range": f"M{row_idx}", "values": [["Error"]]},
                            {"range": f"N{row_idx}", "values": [[now]]},
                            {"range": f"O{row_idx}", "values": [[str(e)]]}
                        ]
                        worksheet.batch_update(error_updates)
                    stats.errors += 1
                
            except Exception as e:
                stats.errors += 1
                # Get detailed error information
                error_details = traceback.format_exc()
                error_type = type(e).__name__
                error_message = str(e) if str(e).strip() else "Unknown error occurred during campaign processing"
                
                # Log detailed error
                logger.error(f"Campaign error for {p.linkedin_url}: {error_type}: {error_message}")
                if error_details:
                    logger.debug(f"Error details for {p.linkedin_url}:\n{error_details}")
                
                # Try to update the worksheet with error information
                if worksheet and row_idx:
                    try:
                        now = dt.datetime.now(dt.UTC).isoformat()
                        simple_error = f"{error_type}: {error_message[:100]}..." if len(error_message) > 100 else f"{error_type}: {error_message}"
                        worksheet.batch_update([
                            {"range": f"M{row_idx}", "values": [["Error"]]},
                            {"range": f"N{row_idx}", "values": [[now]]},
                            {"range": f"O{row_idx}", "values": [[simple_error]]}
                        ])
                    except Exception as update_error:
                        logger.error(f"Failed to update error in sheet: {update_error}")
    
    # Create and gather all profile processing tasks
    tasks = [process_profile(p, i) for i, p in enumerate(profiles)]
    await asyncio.gather(*tasks)
    
    return stats 