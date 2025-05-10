import asyncio
from typing import Optional
from src.unipile_client import UnipileClient
from src.sheets import get_spreadsheet
from src.logging_conf import logger


async def sync_status(sheet_id: str, sheet_name: str, specific_rows: Optional[list] = None) -> dict:
    """
    Sync relationship status data from Unipile to the Google Sheet.
    
    Args:
        sheet_id: Google Sheet ID
        sheet_name: Name of the worksheet
        specific_rows: Optional list of row indices to update (2-based for sheet rows)
    
    Returns:
        Dict with statistics about the sync operation
    """
    stats = {
        "processed": 0,
        "updated": 0,
        "errors": 0,
        "not_found": 0,
        "api_errors": 0
    }
    
    try:
        # Get the worksheet
        ws = get_spreadsheet(sheet_id).worksheet(sheet_name)
        
        # Get all data as records
        rows = ws.get_all_records()
        if not rows:
            logger.warning(f"No data found in sheet {sheet_name}")
            return stats
        
        # Initialize Unipile client
        cli = UnipileClient()
        
        # Pull data once to avoid N² API calls
        invites = {}
        relations = {}
        conversations = {}
        
        # Try to get invitations - don't fail if this API call doesn't work
        try:
            invites = {i.get("provider_id", ""): i for i in await cli.list_sent_invitations()}
            logger.info(f"Loaded {len(invites)} sent invitations")
        except Exception as e:
            logger.error(f"Error fetching invitations: {str(e)}")
            stats["api_errors"] += 1
            # Continue with empty invites
        
        # Try to get relations - don't fail if this API call doesn't work
        try:
            relations = {r.get("provider_id", ""): r for r in await cli.list_relations()}
            logger.info(f"Loaded {len(relations)} relations")
        except Exception as e:
            logger.error(f"Error fetching relations: {str(e)}")
            stats["api_errors"] += 1
            # Continue with empty relations
        
        # Try to get conversations - don't fail if this API call doesn't work
        try:
            conversations = {c.get("provider_id", ""): c for c in await cli.list_conversations()}
            logger.info(f"Loaded {len(conversations)} conversations")
        except Exception as e:
            logger.error(f"Error fetching conversations: {str(e)}")
            stats["api_errors"] += 1
            # Continue with empty conversations
        
        # Process each row
        row_indices = specific_rows if specific_rows else range(len(rows))
        for idx in row_indices:
            sheet_row = idx + 2  # +1 for 0-index to 1-index, +1 for header row
            
            try:
                row = rows[idx]
                stats["processed"] += 1
                
                # Get provider_id from either Provider ID column or by looking up URL in invites
                linkedin_url = row.get("LinkedIn URL", "")
                provider_id = row.get("Provider ID", "")  
                
                # If we don't have provider_id but have URL, try to find in Unipile data
                if not provider_id and linkedin_url:
                    # Extract identifier from URL
                    ident = linkedin_url.rstrip('/').split('/')[-1]
                    
                    # Check if this URL's identifier matches any provider_id we have
                    for pid, invite_data in invites.items():
                        if ident in pid:
                            provider_id = pid
                            break
                    
                    # If still not found, check in relations data
                    if not provider_id:
                        for pid, relation_data in relations.items():
                            if ident in pid:
                                provider_id = pid
                                break
                
                if not provider_id:
                    stats["not_found"] += 1
                    continue
                
                # Get data for this profile
                invite_data = invites.get(provider_id, {})
                relation_data = relations.get(provider_id, {})
                conversation_data = conversations.get(provider_id, {})
                
                # Prepare updates
                updates = {}
                
                # Add Invite ID (column O)
                invite_id = invite_data.get("id", "")
                if invite_id:
                    updates["O"] = invite_id
                
                # Add Connection State (column P)
                connection_state = relation_data.get("state", "NOT_CONNECTED")
                if connection_state:
                    updates["P"] = connection_state
                
                # Add Follower Count (column Q)
                follower_count = invite_data.get("follower_count", 0)
                if follower_count:
                    updates["Q"] = follower_count
                
                # Add Unread Count (column R)
                unread_count = conversation_data.get("unread_count", 0)
                updates["R"] = unread_count  # Always set, even if 0
                
                # Add Last Message UTC (column S)
                last_message_at = conversation_data.get("last_message_at", "")
                if last_message_at:
                    updates["S"] = last_message_at
                
                # Apply updates if we have any
                if updates:
                    # Update in batches of 5 to avoid API rate limits
                    batch_updates = []
                    for col, value in updates.items():
                        batch_updates.append({
                            "range": f"{col}{sheet_row}", 
                            "values": [[value]]
                        })
                    
                    # Apply batch update
                    if batch_updates:
                        ws.batch_update(batch_updates)
                    stats["updated"] += 1
                
            except Exception as e:
                logger.error(f"Error processing row {sheet_row}: {str(e)}")
                stats["errors"] += 1
        
        # Clean up
        await cli.close()
        return stats
    
    except Exception as e:
        logger.error(f"Error in sync_status: {str(e)}")
        try:
            await cli.close()
        except:
            pass
        raise 