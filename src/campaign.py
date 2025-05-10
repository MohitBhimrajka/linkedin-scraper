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
    sheet_name: Optional[str] = None,
    schedule_time: Optional[str] = None
) -> CampaignStats:
    """
    Run a LinkedIn outreach campaign on the specified profiles.
    
    Args:
        profiles: List of LinkedIn profiles to target
        followup_days: Tuple of days to wait before sending follow-up messages
        mode: Campaign mode - "Generate only", "Invite only", "Invite + Comment", or "Full"
        spreadsheet_id: Google Sheet ID where profiles are stored
        sheet_name: Sheet name where profiles are stored
        schedule_time: Optional ISO-format datetime string to schedule message sending
    
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
            
            # Always fetch posts to provide data for comments, regardless of mode
            posts = await client.recent_posts(provider_id, limit=1)
            recent = posts[0]["text"] if posts else ""
            
            # Normalize follower count field for consistency
            pdata["followers_count"] = pdata.get("followersCount", 0)
            
            # Store follower count in the profile object for sheets
            p.followers_count = pdata["followers_count"]
            
            msgs = craft_messages(pdata, recent)
            
            # Ensure we always have non-empty values for all message types
            if not msgs["comment"]:
                headline = pdata.get("headline", "your industry")
                msgs["comment"] = f"Really enjoyed reading about your work in {headline.split('|')[0].strip()}!"
                
            # Make every column non-empty regardless of mode
            for m in ("connection", "comment", "inmail_subject", "inmail_body"):
                if not msgs.get(m):   # safety net
                    msgs[m] = "—"     # an em-dash placeholder
            
            # Ensure followups are never empty
            if not msgs.get("followups") or not any(msgs["followups"]):
                msgs["followups"] = [
                    "Hope you're having a great week! Wanted to share this article that relates to your work.",
                    "Just came across this resource that might interest you given your background.",
                    "Would value your perspective on a project I'm working on - any chance we could chat briefly?"
                ]
            
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
                        
                        # Prepare all updates as a batch to minimize API calls
                        cell_updates = [
                            {"range": f"G{row_idx}", "values": [[msgs["connection"]]]},
                            {"range": f"H{row_idx}", "values": [[msgs["comment"]]]},
                            {"range": f"I{row_idx}", "values": [[msgs["followups"][0] if len(msgs["followups"]) > 0 else "—"]]},
                            {"range": f"J{row_idx}", "values": [[msgs["followups"][1] if len(msgs["followups"]) > 1 else "—"]]},
                            {"range": f"K{row_idx}", "values": [[msgs["followups"][2] if len(msgs["followups"]) > 2 else "—"]]},
                            {"range": f"L{row_idx}", "values": [[f"Subject: {msgs['inmail_subject']}\nBody: {msgs['inmail_body']}"]]},
                            {"range": f"M{row_idx}", "values": [["GENERATED"]]},
                            {"range": f"N{row_idx}", "values": [[now]]},
                            {"range": f"P{row_idx}", "values": [[str(p.followers_count)]]}  # Store followers count in a new column
                        ]
                        
                        # Execute the batch update
                        worksheet.batch_update(cell_updates)
                        logger.info(f"Updated messages for {p.linkedin_url} in sheet")
                        
                except Exception as e:
                    logger.error(f"Failed to update sheet for {p.linkedin_url}: {str(e)}")
            
            # Skip API calls if in generate-only mode
            if mode == "Generate only":
                stats.skipped += 1
                continue
            
            # 2 send invitation (for all modes except "Generate only")
            # Use the invitation API to send connection request, with optional scheduling
            await client.send_invitation(
                provider_id=provider_id, 
                message=msgs["connection"],
                send_at=schedule_time
            )
            
            # Update status in sheet
            if worksheet and row_idx:
                status_value = "SCHEDULED" if schedule_time else "INVITED"
                now = dt.datetime.now(dt.UTC).replace(tzinfo=utc).isoformat()
                worksheet.batch_update([
                    {"range": f"M{row_idx}", "values": [[status_value]]},
                    {"range": f"N{row_idx}", "values": [[now]]}
                ])
            
            # 3 comment (for "Invite + Comment" and "Full" modes)
            if (mode == "Invite + Comment" or mode == "Full") and posts:
                await client.comment_post(
                    post_id=posts[0]["id"], 
                    message=msgs["comment"],
                    send_at=schedule_time
                )
                
                # Update status in sheet
                if worksheet and row_idx:
                    status_value = "SCHEDULED" if schedule_time else "COMMENTED"
                    now = dt.datetime.now(dt.UTC).replace(tzinfo=utc).isoformat()
                    worksheet.batch_update([
                        {"range": f"M{row_idx}", "values": [[status_value]]},
                        {"range": f"N{row_idx}", "values": [[now]]}
                    ])

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
                worksheet.batch_update([
                    {"range": f"M{row_idx}", "values": [["FOLLOW-UPS READY"]]},
                    {"range": f"N{row_idx}", "values": [[now.isoformat()]]}
                ])
                
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
                    worksheet.batch_update([
                        {"range": f"M{row_idx}", "values": [["ERROR"]]},
                        {"range": f"N{row_idx}", "values": [[now]]},
                        {"range": f"O{row_idx}", "values": [[str(e)]]}
                    ])
                except Exception as sheet_error:
                    logger.error(f"Failed to update error status in sheet: {sheet_error}")
    
    # Close the client connection
    await client.close()
    
    return stats 