import asyncio
from typing import Optional
from src.unipile_client import UnipileClient
from src.sheets import get_spreadsheet, create_or_get_master_profiles_sheet
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
        "api_errors": 0,
        "master_profiles_updated": 0
    }
    
    try:
        # Get the spreadsheet
        spreadsheet = get_spreadsheet(sheet_id)
        
        # Get the worksheet
        ws = spreadsheet.worksheet(sheet_name)
        
        # Get the Master_Profiles sheet
        master_sheet = create_or_get_master_profiles_sheet(spreadsheet)
        
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
        
        # First, process the Master_Profiles sheet to update all profiles with the latest Unipile data
        try:
            master_profiles = master_sheet.get_all_records()
            
            for i, profile in enumerate(master_profiles):
                master_row = i + 2  # +1 for 0-index to 1-index, +1 for header row
                
                # Get LinkedIn URL and Provider ID
                linkedin_url = profile.get("LinkedIn URL", "")
                provider_id = profile.get("Provider ID", "")
                
                # Skip if no identifiers
                if not linkedin_url and not provider_id:
                    continue
                
                # Try to match by provider_id first, then by extracting ID from LinkedIn URL
                matched_provider_id = None
                if provider_id:
                    # Direct match by provider_id
                    matched_provider_id = provider_id
                elif linkedin_url:
                    # Extract identifier from URL
                    ident = linkedin_url.rstrip('/').split('/')[-1]
                    
                    # Check if this URL's identifier matches any provider_id we have
                    for pid in invites.keys():
                        if ident in pid:
                            matched_provider_id = pid
                            break
                    
                    # If still not found, check in relations data
                    if not matched_provider_id:
                        for pid in relations.keys():
                            if ident in pid:
                                matched_provider_id = pid
                                break
                
                # If we found a match, update the profile
                if matched_provider_id:
                    # Get data for this profile
                    invite_data = invites.get(matched_provider_id, {})
                    relation_data = relations.get(matched_provider_id, {})
                    conversation_data = conversations.get(matched_provider_id, {})
                    
                    # Prepare updates
                    updates = {}
                    
                    # Update Provider ID if missing
                    if not provider_id and matched_provider_id:
                        updates["B"] = matched_provider_id
                    
                    # Add Connection State (column L)
                    connection_state = relation_data.get("state", "NOT_CONNECTED")
                    if connection_state:
                        updates["L"] = connection_state
                    
                    # Add Unipile Last Interaction UTC (column M)
                    last_message_at = conversation_data.get("last_message_at", "")
                    if last_message_at:
                        updates["M"] = last_message_at
                    
                    # Extract and add interaction type (column R)
                    last_interaction_type = ""
                    if conversation_data:
                        # Determine interaction type based on conversation data
                        if conversation_data.get("last_message_direction") == "inbound":
                            last_interaction_type = "Message Received"
                        elif conversation_data.get("last_message_direction") == "outbound":
                            last_interaction_type = "Message Sent"
                        elif relation_data.get("state") == "CONNECTED" and relation_data.get("updated_at"):
                            last_interaction_type = "Connection Accepted"
                        elif relation_data.get("state") == "PENDING" and relation_data.get("updated_at"):
                            last_interaction_type = "Invitation Sent"
                        
                        if last_interaction_type:
                            updates["R"] = last_interaction_type
                    
                    # Extract and add last interaction snippet (column S)
                    last_interaction_snippet = ""
                    if conversation_data and conversation_data.get("last_message_text"):
                        last_interaction_snippet = conversation_data.get("last_message_text", "")
                        # Truncate if too long
                        if len(last_interaction_snippet) > 150:
                            last_interaction_snippet = last_interaction_snippet[:147] + "..."
                        updates["S"] = last_interaction_snippet
                    
                    # Apply updates if we have any
                    if updates:
                        # Prepare batch updates
                        batch_updates = []
                        for col, value in updates.items():
                            batch_updates.append({
                                "range": f"{col}{master_row}", 
                                "values": [[value]]
                            })
                        
                        # Apply batch update
                        if batch_updates:
                            master_sheet.batch_update(batch_updates)
                            stats["master_profiles_updated"] += 1
            
            logger.info(f"Updated {stats['master_profiles_updated']} profiles in Master_Profiles sheet")
        except Exception as e:
            logger.error(f"Error updating Master_Profiles sheet: {str(e)}")
            # Continue with the original sync function
        
        # Process each row in the original sheet
        row_indices = specific_rows if specific_rows else range(len(rows))
        for idx in row_indices:
            sheet_row = idx + 2  # +1 for 0-index to 1-index, +1 for header row
            
            try:
                row = rows[idx]
                stats["processed"] += 1
                
                # Get provider_id from either Provider ID column or by looking up URL in invites
                linkedin_url = row.get("LinkedIn URL", "")
                
                # Skip row if LinkedIn URL is missing entirely
                if not linkedin_url:
                    logger.warning(f"Row {sheet_row} is missing LinkedIn URL, skipping")
                    stats["not_found"] += 1
                    continue
                
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
                
                # Add Invite ID (column S)
                invite_id = invite_data.get("id", "")
                if invite_id:
                    updates["S"] = invite_id
                
                # Add Connection State (column T)
                connection_state = relation_data.get("state", "NOT_CONNECTED")
                if connection_state:
                    updates["T"] = connection_state
                
                # Add Follower Count (column U)
                follower_count = invite_data.get("follower_count", 0)
                if follower_count:
                    updates["U"] = follower_count
                
                # Add Unread Count (column V)
                unread_count = conversation_data.get("unread_count", 0)
                updates["V"] = unread_count  # Always set, even if 0
                
                # Add Last Message UTC (column W)
                last_message_at = conversation_data.get("last_message_at", "")
                if last_message_at:
                    updates["W"] = last_message_at
                
                # Add Last Interaction Type (column Z)
                last_interaction_type = ""
                if conversation_data:
                    # Determine interaction type based on conversation data
                    if conversation_data.get("last_message_direction") == "inbound":
                        last_interaction_type = "Message Received"
                    elif conversation_data.get("last_message_direction") == "outbound":
                        last_interaction_type = "Message Sent"
                    elif relation_data.get("state") == "CONNECTED" and relation_data.get("updated_at"):
                        last_interaction_type = "Connection Accepted"
                    elif relation_data.get("state") == "PENDING" and relation_data.get("updated_at"):
                        last_interaction_type = "Invitation Sent"
                    
                    if last_interaction_type:
                        updates["Z"] = last_interaction_type
                
                # Add Last Interaction Snippet (column AA)
                last_interaction_snippet = ""
                if conversation_data and conversation_data.get("last_message_text"):
                    last_interaction_snippet = conversation_data.get("last_message_text", "")
                    # Truncate if too long
                    if len(last_interaction_snippet) > 150:
                        last_interaction_snippet = last_interaction_snippet[:147] + "..."
                    updates["AA"] = last_interaction_snippet
                
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
        
        return stats
    except Exception as e:
        logger.error(f"General error in sync_status: {str(e)}")
        raise 