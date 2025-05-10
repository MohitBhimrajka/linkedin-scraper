import asyncio, pytz, datetime as dt
from typing import List, Dict, Optional
from src.unipile_client import UnipileClient
from src.personalize import craft_messages
from src.transform import Profile
from src.logging_conf import logger
import gspread

class CampaignStats: 
    generated=0
    sent=0 
    errors=0
    skipped=0

async def run_campaign(
    profiles: List[Profile], 
    followup_days=(3,7,14), 
    mode="Full",
    spreadsheet_id: Optional[str] = None, 
    sheet_name: Optional[str] = None
) -> CampaignStats:
    """
    Run a LinkedIn outreach campaign on the specified profiles.
    
    Args:
        profiles: List of LinkedIn profiles to target
        followup_days: Tuple of days to wait before sending follow-up messages
        mode: Campaign mode - "Generate only", "Invite only", "Invite + Comment", or "Full"
        spreadsheet_id: Google Sheet ID where profiles are stored
        sheet_name: Sheet name where profiles are stored
    
    Returns:
        CampaignStats object with counts of actions taken
    """
    client = UnipileClient()  # uses env vars
    stats = CampaignStats()
    utc = pytz.UTC
    
    # Find the sheet if IDs are provided to update campaign messages
    worksheet = None
    all_data = []
    if spreadsheet_id and sheet_name:
        try:
            from src.sheets import get_spreadsheet
            spreadsheet = get_spreadsheet(spreadsheet_id)
            worksheet = spreadsheet.worksheet(sheet_name)
            # Get all data to find row indexes
            all_data = worksheet.get_all_records()
        except Exception as e:
            logger.error(f"Failed to access sheet: {e}")
            worksheet = None
    
    for i, p in enumerate(profiles):
        # Initialize row_idx outside the try block
        row_idx = None
        
        try:
            # 1 fetch enriched data (always do this regardless of mode)
            pdata = await client.get_profile(p.linkedin_url)
            # Get the provider_id from the API response
            provider_id = pdata["provider_id"]  # Changed from 'id' to 'provider_id'
            
            posts = await client.recent_posts(provider_id, limit=1) if mode != "Generate only" else []
            recent = posts[0]["text"] if posts else ""
            msgs = craft_messages(pdata, recent)
            stats.generated += 1
            
            # Find and update the corresponding row in the sheet
            if worksheet:
                try:
                    # Find row by LinkedIn URL (add 2 because of header and 1-based indexing)
                    for idx, row in enumerate(all_data):
                        if row.get("LinkedIn URL") == p.linkedin_url:
                            row_idx = idx + 2  # +2 for header and 1-based indexing
                            break
                    
                    if row_idx:
                        now = dt.datetime.now(dt.UTC).replace(tzinfo=utc).isoformat()
                        # Update the message columns
                        worksheet.update(f"G{row_idx}", msgs["connection"])  # Connection Msg
                        worksheet.update(f"H{row_idx}", msgs["comment"])     # Comment Msg
                        worksheet.update(f"I{row_idx}", msgs["followups"][0] if len(msgs["followups"]) > 0 else "")  # F/U-1
                        worksheet.update(f"J{row_idx}", msgs["followups"][1] if len(msgs["followups"]) > 1 else "")  # F/U-2
                        worksheet.update(f"K{row_idx}", msgs["followups"][2] if len(msgs["followups"]) > 2 else "")  # F/U-3
                        worksheet.update(f"L{row_idx}", f"Subject: {msgs['inmail_subject']}\nBody: {msgs['inmail_body']}")  # InMail
                        
                        # Update status
                        worksheet.update(f"M{row_idx}", "GENERATED")  # Contact Status
                        worksheet.update(f"N{row_idx}", now)          # Last Action UTC
                except Exception as e:
                    logger.error(f"Failed to update sheet for {p.linkedin_url}: {e}")
            
            # Skip API calls if in generate-only mode
            if mode == "Generate only":
                stats.skipped += 1
                continue
            
            # 2 send invitation (for all modes except "Generate only")
            await client.send_invitation(provider_id, msgs["connection"])  # Changed from profile_id to provider_id
            
            # Update status in sheet
            if worksheet and row_idx:
                now = dt.datetime.now(dt.UTC).replace(tzinfo=utc).isoformat()
                worksheet.update(f"M{row_idx}", "INVITED")  # Contact Status
                worksheet.update(f"N{row_idx}", now)        # Last Action UTC
            
            # 3 comment (for "Invite + Comment" and "Full" modes)
            if (mode == "Invite + Comment" or mode == "Full") and posts:
                await client.comment_post(posts[0]["id"], msgs["comment"])  # Using post id which should be correct
                
                # Update status in sheet
                if worksheet and row_idx:
                    now = dt.datetime.now(dt.UTC).replace(tzinfo=utc).isoformat()
                    worksheet.update(f"M{row_idx}", "COMMENTED")  # Contact Status
                    worksheet.update(f"N{row_idx}", now)          # Last Action UTC

            # 4 store follow-ups in sheet (don't schedule yet)
            # Real conversation ID only exists after invite acceptance
            # Instead of scheduling via API now, we'll store messages in the sheet
            # and implement a separate process to check invitation status and send follow-ups later
            
            # We still calculate intended follow-up dates for reference
            now = dt.datetime.now(dt.UTC).replace(tzinfo=utc)
            followup_dates = []
            for j in range(len(msgs["followups"])):  # Changed i to j to avoid variable reuse
                followup_dates.append((now + dt.timedelta(days=followup_days[j])).isoformat())
            
            # Mark follow-ups as ready if in Full mode
            if mode == "Full" and worksheet and row_idx:
                worksheet.update(f"M{row_idx}", "FOLLOW-UPS READY")  # Contact Status
                worksheet.update(f"N{row_idx}", now.isoformat())      # Last Action UTC
                
                # In a real implementation, we would store the user ID and the follow-up dates
                # in a separate table or queue for a background worker to process
            
            stats.sent += 1
        except Exception as e:
            stats.errors += 1
            logger.error(f"Campaign error for {p.linkedin_url}: {e}")
            
            # Update status in sheet to show error - only if worksheet and row_idx are valid
            if worksheet and row_idx:
                try:
                    now = dt.datetime.now(dt.UTC).replace(tzinfo=utc).isoformat()
                    worksheet.update(f"M{row_idx}", "ERROR")  # Contact Status
                    worksheet.update(f"N{row_idx}", now)      # Last Action UTC
                    worksheet.update(f"O{row_idx}", str(e))   # Store error message in extra column
                except Exception as sheet_error:
                    logger.error(f"Failed to update error status in sheet: {sheet_error}")
    
    # Close the client connection
    await client.close()
    
    return stats 