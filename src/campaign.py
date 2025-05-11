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
from src.sheets import update_master_profile_action

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
    spreadsheet = None
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
            spreadsheet = gc.open_by_key(spreadsheet_id)
            worksheet = spreadsheet.worksheet(sheet_name)
            logger.info(f"Connected to Google Sheet '{sheet_name}'")
        except Exception as e:
            logger.error(f"Failed to connect to Google Sheets: {e}")
            logger.debug(f"Google Sheets connection error details: {traceback.format_exc()}")
            worksheet = None  # Ensure worksheet is None if connection fails
    
    # Create a Unipile client
    client = UnipileClient()
    
    try:
        # Create tasks for batch processing profiles
        tasks = []
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def process_profile(p, idx):
            async with semaphore:
                row_idx = None
                recent = ""  # Initialize recent here
                posts_data = []  # Initialize posts_data
                provider_id = None  # Initialize provider_id
                
                try:
                    logger.info(f"Processing profile {idx+1}/{len(profiles)}: {p.linkedin_url}")
                    
                    # 1. If message generation is needed, do that first
                    if should_generate_messages:
                        # Generate personalized messages based on the profile data
                        # Pass recent_post if available and used by craft_messages
                        recent_post_text = recent  # Now recent is initialized
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
                        
                        # Update Master_Profiles with message generation action
                        if spreadsheet:
                            update_master_profile_action(
                                spreadsheet, 
                                p.linkedin_url, 
                                "Messages generated",
                                None  # Don't have provider_id yet
                            )
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
                            # Use asyncio.to_thread for gspread operations
                            cell = await asyncio.to_thread(worksheet.find, p.linkedin_url)
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
                            await asyncio.to_thread(worksheet.batch_update, batch_updates)
                            logger.info(f"Updated messages for row {row_idx}")
                        except gspread.exceptions.CellNotFound:
                            logger.warning(f"Profile URL {p.linkedin_url} not found in sheet. Cannot update messages.")
                        except Exception as e:
                            logger.warning(f"Could not update sheet for {p.linkedin_url}: {e}")
                    
                    # Skip API calls if in generate-only mode
                    if mode == "Generate only":
                        stats.skipped += 1
                        logger.info(f"Generate only mode: Skipping API actions for {p.linkedin_url}")
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
                        posts_data = await client.recent_posts(provider_id=provider_id, limit=1)
                        
                        # Ensure posts_data is a list
                        if not isinstance(posts_data, list):
                            logger.warning(f"Unexpected posts data structure received: {type(posts_data)}")
                            posts_data = []
                            recent = ""
                        else:
                            # Get text from the first post if available
                            recent = ""
                            if posts_data and len(posts_data) > 0:
                                post = posts_data[0]
                                if isinstance(post, dict) and "text" in post:
                                    recent = post["text"]
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
                    
                    # 5. Before sending invitation, check current connection state from Unipile
                    if mode in ["Invite only", "Invite + Comment", "Full (invite, comment, follow-ups)"]:
                        try:
                            relations = await client.list_relations()
                            # Find if this provider_id already has a relation
                            for relation in relations:
                                if relation.get("provider_id") == provider_id:
                                    current_state = relation.get("state", "NOT_CONNECTED")
                                    # If already connected or pending, skip sending invite
                                    if current_state in ["CONNECTED", "PENDING"]:
                                        logger.info(f"Skipping invitation for {provider_id}: Already in state {current_state}")
                                        # Update the sheet status
                                        if worksheet and row_idx:
                                            now = dt.datetime.now(dt.UTC).isoformat()
                                            status_updates = [
                                                {"range": f"O{row_idx}", "values": [[f"Already {current_state}"]]},
                                                {"range": f"P{row_idx}", "values": [[now]]},
                                            ]
                                            await asyncio.to_thread(worksheet.batch_update, status_updates)
                                        
                                        # Update Master_Profiles
                                        if spreadsheet:
                                            update_master_profile_action(
                                                spreadsheet, 
                                                p.linkedin_url, 
                                                f"Already {current_state}",
                                                provider_id
                                            )
                                        
                                        stats.skipped += 1
                                        return
                        except Exception as e:
                            # If we can't check relations, log and continue
                            logger.warning(f"Failed to check relations for {provider_id}: {e}")
                    
                    # 6. Send invitation with connection message
                    if mode in ["Invite only", "Invite + Comment", "Full (invite, comment, follow-ups)"]:
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
                                    {"range": f"O{row_idx}", "values": [["Invited"]]},  # Column O for Contact Status
                                    {"range": f"P{row_idx}", "values": [[now]]},        # Column P for Last Action UTC
                                ]
                                await asyncio.to_thread(worksheet.batch_update, status_updates)
                            
                            # Update Master_Profiles
                            if spreadsheet:
                                update_master_profile_action(
                                    spreadsheet, 
                                    p.linkedin_url, 
                                    "Invited",
                                    provider_id
                                )
                            
                        except Exception as e:
                            logger.error(f"Failed to send invitation to {provider_id}: {str(e)}")
                            stats.errors += 1
                            if worksheet and row_idx:
                                await asyncio.to_thread(
                                    worksheet.update_cell, 
                                    row_idx, 
                                    15,  # Column O (15th column) for Contact Status
                                    f"Error: {str(e)[:100]}"
                                )
                    
                    # 7. If we have recent posts and the mode includes commenting
                    if recent and recent.strip() and posts_data and mode in ["Invite + Comment", "Full (invite, comment, follow-ups)"]:
                        try:
                            post_id = posts_data[0].get("id")
                            if post_id:
                                logger.info(f"Commenting on recent post for {provider_id}")
                                await client.comment_post(
                                    post_id=post_id,
                                    message=msgs["comment"],
                                    send_at=schedule_time
                                )
                                
                                # Update sheet
                                if worksheet and row_idx:
                                    await asyncio.to_thread(
                                        worksheet.update_cell, 
                                        row_idx, 
                                        15,  # Column O (15th column) for Contact Status 
                                        "Invited + Commented"
                                    )
                                    
                                    # Update Master_Profiles
                                    if spreadsheet:
                                        update_master_profile_action(
                                            spreadsheet, 
                                            p.linkedin_url, 
                                            "Invited + Commented",
                                            provider_id
                                        )
                        except Exception as e:
                            logger.warning(f"Failed to comment on post for {provider_id}: {str(e)}")
                            # Non-fatal error, don't increment error count
                    
                    # 8. If in message-only mode, send message directly
                    if mode == "Message only" and provider_id:
                        # TODO: Implement this once the Unipile API supports direct messaging
                        pass
                
                except Exception as e:
                    logger.error(f"Error processing profile {p.linkedin_url}: {str(e)}")
                    logger.debug(traceback.format_exc())
                    stats.errors += 1
                    if worksheet and row_idx:
                        try:
                            await asyncio.to_thread(
                                worksheet.update_cell, 
                                row_idx, 
                                15,  # Column O (15th column) for Contact Status
                                f"Error: {str(e)[:100]}" 
                            )
                        except Exception as update_error:
                            # If we can't even update the error status, just log it
                            logger.error(f"Failed to update error status: {str(update_error)}")
        
        # Create a task for each profile
        tasks = [process_profile(profile, i) for i, profile in enumerate(profiles)]
        
        # Run all tasks concurrently with the semaphore limiting concurrency
        await asyncio.gather(*tasks)
        
    except Exception as e:
        logger.error(f"Campaign error: {str(e)}")
        logger.debug(traceback.format_exc())
        stats.errors += 1
    finally:
        # Clean up
        try:
            await client.close()
        except:
            pass
    
    return stats 