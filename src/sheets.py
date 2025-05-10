import os
import re
import time
import json
from typing import List, Dict, Optional, Set
import gspread
from google.oauth2.service_account import Credentials
from gspread.exceptions import APIError, SpreadsheetNotFound, WorksheetNotFound
from src.transform import Profile
from src.logging_conf import logger
from google import genai
from google.genai import types
import traceback
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type


def get_google_sheets_client():
    """
    Authenticate with Google Sheets API.
    
    Tries service account first, then fallbacks to OAuth if available.
    
    Returns:
        Authenticated gspread client
    """
    # Check for service account credentials
    try:
        scopes = ['https://www.googleapis.com/auth/spreadsheets']
        
        # Try to use service account JSON if it exists
        if os.path.exists('service-account.json'):
            credentials = Credentials.from_service_account_file(
                'service-account.json', scopes=scopes
            )
            client = gspread.authorize(credentials)
            logger.info("Authenticated with Google Sheets using service account")
            return client
    except Exception as e:
        logger.warning(f"Service account auth failed: {str(e)}")
    
    # Fallback to OAuth (requires user interaction)
    try:
        client = gspread.oauth()
        logger.info("Authenticated with Google Sheets using OAuth")
        return client
    except Exception as e:
        logger.error(f"OAuth auth failed: {str(e)}")
        raise ValueError("Failed to authenticate with Google Sheets. Check your credentials.")


def generate_sheet_names_with_gemini(icp_list: List[Dict]) -> Dict[int, str]:
    """
    Use Gemini to generate professional sheet names for each ICP.
    
    Args:
        icp_list: List of ICP dictionaries with 'original' field
        
    Returns:
        Dictionary mapping ICP index to generated sheet name
    """
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    
    if not gemini_api_key:
        logger.warning("GEMINI_API_KEY not found in environment variables, falling back to default naming")
        return {}
    
    client = genai.Client(api_key=gemini_api_key)
    model = "gemini-2.5-flash-preview-04-17"  # Use the same model as in optimize_query_with_gemini
    
    # Extract just the original ICP descriptions
    icp_descriptions = [{"index": i, "description": icp["original"]} for i, icp in enumerate(icp_list)]
    
    prompt = f"""
    You are a professional data analyst who excels at creating concise, meaningful names for data categories.

    I have a list of Ideal Customer Profile (ICP) descriptions that need concise sheet names for Google Sheets.
    Each sheet name should:
    - Be 2-4 words maximum
    - Be professional and descriptive
    - Focus on job roles, industries, or key targeting criteria
    - Avoid special characters like slash, asterisk, question mark, colon, bracket
    - Be 30 characters or less

    Here are the ICPs:
    {json.dumps(icp_descriptions, indent=2)}

    Return ONLY a valid JSON object with this exact format:
    {{
      "sheet_names": [
        {{
          "index": 0,
          "name": "Concise Sheet Name 1"
        }},
        {{
          "index": 1,
          "name": "Concise Sheet Name 2"
        }}
      ]
    }}
    """
    
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]
    
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )
    
    # Add retry logic
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_content_config,
            )
            
            response_text = response.text
            
            # Extract JSON from the response (in case there's other text)
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                # Verify the expected structure
                if "sheet_names" in data and isinstance(data["sheet_names"], list):
                    # Convert to dictionary mapping index to name
                    result = {}
                    for item in data["sheet_names"]:
                        if "index" in item and "name" in item:
                            result[item["index"]] = item["name"]
                    
                    logger.info(f"Generated {len(result)} sheet names with Gemini")
                    return result
            
            logger.warning(f"Attempt {attempt+1}: Invalid JSON structure from Gemini")
            
        except Exception as e:
            logger.warning(f"Attempt {attempt+1}: Error getting sheet names from Gemini: {str(e)}")
            traceback.print_exc()
        
        # Only sleep if this is not the last attempt
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    
    logger.warning("Failed to get valid sheet names from Gemini after multiple attempts")
    return {}


def generate_sheet_name(icp_text: str, index: int, gemini_names: Dict[int, str] = None) -> str:
    """
    Generate a descriptive sheet name from the ICP text.
    
    Args:
        icp_text: Original ICP text
        index: ICP index (for uniqueness)
        gemini_names: Optional dictionary of Gemini-generated names by index
        
    Returns:
        A short descriptive name for the sheet (max 30 chars)
    """
    # First try to use Gemini-generated name if available
    if gemini_names and index in gemini_names and gemini_names[index]:
        sheet_name = gemini_names[index]
        # Quick validation and cleanup
        sheet_name = re.sub(r'[\\/*[\]?:]', '', sheet_name)
        if len(sheet_name) > 30:
            sheet_name = sheet_name[:27] + "..."
        logger.info(f"Using Gemini-generated sheet name: {sheet_name}")
        return sheet_name
    
    # Extract key phrases by removing common words and special characters
    text = icp_text.lower()
    
    # Try to extract roles (expanded list with more options)
    roles = re.findall(r'(ceo|cto|coo|cfo|vp|chief|director|head|lead|manager|executive|founder|president|principal)', text)
    
    # Try to extract industries (expanded list)
    industries = re.findall(r'(saas|tech|fintech|health|retail|ecommerce|software|manufacturing|marketing|finance|pharma|insurance|legal|hospitality|education)', text)
    
    # Try to extract locations 
    locations = re.findall(r'(san francisco|new york|london|berlin|europe|usa|united states|america|asia|canada|australia|india|uk|france|germany)', text)
    
    # Extract company characteristics
    company_chars = re.findall(r'(startup|enterprise|series [abcde]|fortune 500|small business|mid-market|global)', text)
    
    # Combine the most important elements
    parts = []
    
    # Primary role
    if roles:
        parts.append(roles[0].title())
    
    # Industry or sector
    if industries:
        parts.append(industries[0].title())
    
    # Add company characteristic if we have space
    if company_chars and len(parts) < 2:
        parts.append(company_chars[0].title())
        
    # Location only if we don't have enough context
    if locations and len(parts) < 2:
        parts.append(locations[0].title())
    
    # If we couldn't extract meaningful parts, use key words from the text
    if not parts:
        # Extract the first 2-3 significant words
        words = re.sub(r'[^\w\s]', ' ', text).split()
        significant_words = [w for w in words if len(w) > 3 and w not in ('with', 'from', 'that', 'have', 'this', 'are', 'for', 'the', 'and')]
        
        if significant_words:
            # Take up to 3 significant words
            parts = [w.title() for w in significant_words[:3]]
        else:
            # Last resort, use the first few words
            parts = [w.title() for w in words[:2] if w]
    
    # Create the sheet name
    if parts:
        sheet_name = " ".join(parts)
    else:
        # If all else fails, use a generic name with timestamp for uniqueness
        timestamp = int(time.time()) % 10000
        sheet_name = f"ICP {index + 1}"
    
    # Always add the index for uniqueness in case of similar ICPs
    if index > 0 and not re.search(r'\d+$', sheet_name):
        sheet_name = f"{sheet_name} {index + 1}"
    
    # Ensure name isn't too long (Sheets has a 100 char limit, but we'll keep it shorter)
    if len(sheet_name) > 30:
        sheet_name = sheet_name[:27] + "..."
    
    # Replace characters that might cause issues in sheet names
    sheet_name = re.sub(r'[\\/*[\]?:]', '', sheet_name)
    
    return sheet_name


def ensure_unique_sheet_name(spreadsheet, base_name: str) -> str:
    """
    Ensure sheet name is unique by appending a number if necessary.
    
    Args:
        spreadsheet: Google Spreadsheet object
        base_name: Desired sheet name
        
    Returns:
        A unique sheet name
    """
    try:
        existing_sheets = [ws.title for ws in spreadsheet.worksheets()]
    except Exception as e:
        logger.error(f"Failed to get worksheets: {str(e)}")
        existing_sheets = []
    
    name = base_name
    suffix = 1
    
    # Keep incrementing suffix until we find a unique name
    while name in existing_sheets:
        name = f"{base_name} ({suffix})"
        suffix += 1
        
        # Ensure we don't exceed sheet name length limits
        if len(name) > 30:
            name = f"{base_name[:25]}_{suffix}"
    
    return name


def create_or_get_worksheet(spreadsheet, sheet_name: str) -> gspread.Worksheet:
    """
    Create a new worksheet or get existing one.
    
    Args:
        spreadsheet: Google Spreadsheet object
        sheet_name: Name for the worksheet
        
    Returns:
        Worksheet object
    """
    # Make sheet name unique
    unique_sheet_name = ensure_unique_sheet_name(spreadsheet, sheet_name)
    
    # Retry logic for worksheet operations (in case of API rate limits)
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Check if worksheet already exists
            try:
                worksheet = spreadsheet.worksheet(unique_sheet_name)
                logger.info(f"Using existing worksheet: {unique_sheet_name}")
                
                # Clear existing content except headers
                try:
                    # Get the current row count
                    all_values = worksheet.get_all_values()
                    
                    # Check if we need to upgrade headers to new format with status columns
                    if all_values and len(all_values[0]) < 19:
                        logger.info(f"Upgrading worksheet {unique_sheet_name} from {len(all_values[0])} to 19 columns")
                        
                        # Define the full set of headers
                        headers = [
                            "LinkedIn URL", 
                            "Title", 
                            "First Name", 
                            "Last Name", 
                            "Company",
                            "Location",
                            "Description", 
                            "Profile Image URL",
                            "Connection Msg",
                            "Comment Msg",
                            "FU-1",
                            "FU-2",
                            "FU-3",
                            "InMail",
                            "Contact Status",
                            "Last Action UTC",
                            "Error Msg",
                            # New relationship state columns
                            "Invite ID",
                            "Connection State",
                            "Follower Cnt",
                            "Unread Cnt",
                            "Last Msg UTC"
                        ]
                        
                        # Update headers (A-V = 21 columns)
                        worksheet.update("A1:V1", [headers])
                        
                        # Format headers
                        worksheet.format("A1:V1", {
                            "textFormat": {"bold": True},
                            "horizontalAlignment": "CENTER",
                            "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}
                        })
                        
                        # Set default values for status columns for all existing rows
                        if len(all_values) > 1:
                            # Create default values for Contact Status and Connection State
                            contact_status_updates = []
                            connection_state_updates = []
                            
                            for row_num in range(2, len(all_values) + 1):  # Start from row 2 (skip header)
                                contact_status_updates.append({
                                    "range": f"M{row_num}",
                                    "values": [["Not contacted"]]
                                })
                                connection_state_updates.append({
                                    "range": f"Q{row_num}",
                                    "values": [["NOT_CONNECTED"]]
                                })
                            
                            # Apply updates in batches to avoid API limits
                            if contact_status_updates:
                                for i in range(0, len(contact_status_updates), 20):
                                    batch = contact_status_updates[i:i+20]
                                    batch_update_with_retry(worksheet, batch)
                            
                            if connection_state_updates:
                                for i in range(0, len(connection_state_updates), 20):
                                    batch = connection_state_updates[i:i+20]
                                    batch_update_with_retry(worksheet, batch)
                            
                            logger.info(f"Set default status values for {len(all_values)-1} rows")
                        
                        # Auto-resize columns
                        try:
                            columns_auto_resize_with_retry(worksheet, 0, 20)  # Resize columns A-V (0-20)
                        except Exception as resize_error:
                            logger.warning(f"Failed to auto-resize columns: {str(resize_error)}")
                    
                    if len(all_values) > 1:  # If there are rows beyond the header
                        # Clear everything after row 1
                        worksheet.batch_clear([f"A2:Z{len(all_values)}"])
                        logger.info(f"Cleared existing content in worksheet: {unique_sheet_name}")
                except Exception as e:
                    logger.warning(f"Failed to clear existing worksheet content: {str(e)}")
                
                return worksheet
            except WorksheetNotFound:
                # Create a new worksheet
                worksheet = spreadsheet.add_worksheet(title=unique_sheet_name, rows=1000, cols=20)
                logger.info(f"Created new worksheet: {unique_sheet_name}")
                
                # Add headers
                headers = [
                    "LinkedIn URL", 
                    "Title", 
                    "First Name", 
                    "Last Name", 
                    "Company",
                    "Location",
                    "Description", 
                    "Profile Image URL",
                    "Connection Msg",
                    "Comment Msg",
                    "FU-1",
                    "FU-2",
                    "FU-3",
                    "InMail",
                    "Contact Status",
                    "Last Action UTC",
                    "Error Msg",
                    # New relationship state columns
                    "Invite ID",
                    "Connection State",
                    "Follower Cnt",
                    "Unread Cnt",
                    "Last Msg UTC"
                ]
                append_rows_with_retry(worksheet, [headers], value_input_option='RAW')
                
                # Format headers (make bold, freeze row, center alignment)
                worksheet.format("A1:V1", {
                    "textFormat": {"bold": True},
                    "horizontalAlignment": "CENTER",
                    "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}
                })
                worksheet.freeze(rows=1)
                
                # Auto-resize columns to fit content - this replaces set_column_width
                try:
                    columns_auto_resize_with_retry(worksheet, 0, 21)  # Resize columns A-V (includes new columns)
                except Exception as e:
                    logger.warning(f"Failed to auto-resize columns: {str(e)}")
                
                return worksheet
        
        except APIError as e:
            if attempt < max_retries - 1:
                logger.warning(f"API error when creating/getting worksheet (attempt {attempt+1}): {str(e)}")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to create/get worksheet after {max_retries} attempts: {str(e)}")
                raise
    
    # This should never be reached due to the raise in the loop, but just in case
    raise RuntimeError("Failed to create or get worksheet")


def append_rows_to_worksheet(worksheet, profiles: List[Profile]):
    """
    Append profile data to a specific worksheet.
    
    Args:
        worksheet: Target worksheet
        profiles: List of Profile objects to append
    """
    if not profiles:
        logger.info("No profiles to append to worksheet")
        return
    
    # Convert profiles to rows with the required column order:
    # LinkedIn URL, Title, First Name, Last Name, Description, Profile Image URL
    rows = []
    for profile in profiles:
        # Truncate long descriptions and other fields to prevent cells from expanding too wide
        description = profile.description or ""
        if len(description) > 300:
            description = description[:297] + "..."
            
        linkedin_url = profile.linkedin_url
        if len(linkedin_url) > 100:
            linkedin_url = linkedin_url[:97] + "..."
            
        profile_image_url = profile.profile_image_url or ""
        if len(profile_image_url) > 100:
            profile_image_url = profile_image_url[:97] + "..."
            
        title = profile.title or ""
        if len(title) > 100:
            title = title[:97] + "..."
            
        # Get company and location if available
        company = profile.company if hasattr(profile, 'company') else ""
        location = profile.location if hasattr(profile, 'location') else ""
        
        # Set default values for connection_state and contact_status if not provided
        connection_state = profile.connection_state if hasattr(profile, 'connection_state') and profile.connection_state else "NOT_CONNECTED"
        contact_status = profile.contact_status if hasattr(profile, 'contact_status') and profile.contact_status else "Not contacted"
        
        rows.append([
            linkedin_url,
            title,
            profile.first_name or "",
            profile.last_name or "",
            company,           # Add company
            location,          # Add location
            description,
            profile_image_url,
            profile.connection_msg if hasattr(profile, 'connection_msg') else "",  # Connection Msg
            profile.comment_msg if hasattr(profile, 'comment_msg') else "",  # Comment Msg
            profile.followup1 if hasattr(profile, 'followup1') else "",  # F/U-1
            profile.followup2 if hasattr(profile, 'followup2') else "",  # F/U-2
            profile.followup3 if hasattr(profile, 'followup3') else "",  # F/U-3
            profile.inmail if hasattr(profile, 'inmail') else "",  # InMail
            contact_status,  # Contact Status
            "",  # Last Action UTC
            "",  # Error Msg
            "",  # Invite ID
            connection_state,  # Connection State
            str(profile.followers_count) if profile.followers_count else "",  # Followers Count
            "0",  # Unread Count (default to 0)
            ""   # Last Msg UTC
        ])
    
    # Retry logic for append operations (in case of API rate limits)
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Append the rows to the worksheet
            append_rows_with_retry(worksheet, rows)
            logger.info(f"Successfully appended {len(rows)} profiles to worksheet")
            
            # Format all cells for better readability
            if rows:
                last_row = worksheet.row_count
                first_new_row = last_row - len(rows) + 1
                
                # Check if we're exceeding 1000 rows (Google Sheets grid limit)
                if first_new_row > 1:  # Make sure we're not formatting the header row
                    try:
                        # Make sure we don't exceed sheet limits (1000 rows max in most sheets)
                        max_format_row = min(last_row, 1000)  # Google Sheets has a limit of 1000 rows by default
                        
                        if first_new_row <= max_format_row:
                            # Only format rows within the limit
                            format_last_row = min(last_row, max_format_row)
                            
                            # Add formatting for better readability
                            worksheet.format(f"A{first_new_row}:P{format_last_row}", {
                                "wrapStrategy": "WRAP",  # Change from CLIP to WRAP
                                "verticalAlignment": "TOP",
                                "padding": {"top": 2, "bottom": 2}
                            })
                    except Exception as e:
                        logger.warning(f"Failed to format cells: {str(e)}")
            
            # Auto-resize columns after adding data
            try:
                columns_auto_resize_with_retry(worksheet, 0, 21)  # Resize columns A-V (includes new columns)
            except Exception as e:
                logger.warning(f"Failed to auto-resize columns: {str(e)}")
                
            break
        
        except APIError as e:
            if attempt < max_retries - 1:
                logger.warning(f"API error when appending rows (attempt {attempt+1}): {str(e)}")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to append rows after {max_retries} attempts: {str(e)}")
                raise
    else:
        raise RuntimeError("Failed to append rows to worksheet")


def create_summary_sheet(spreadsheet, icp_results: List[Dict]):
    """
    Create or update a master sheet showing all ICPs and their result counts.
    
    Args:
        spreadsheet: Google Spreadsheet object
        icp_results: List of ICP data dictionaries
    """
    try:
        # Create or get the master sheet
        try:
            master_sheet = spreadsheet.worksheet("Master_Sheet")
            # Get existing rows to append to them
            existing_values = master_sheet.get_all_values()
            existing_rows = len(existing_values)
            
            # Check if headers exist
            if existing_rows == 0 or not existing_values[0] or len(existing_values[0]) < 6:
                # Add headers if missing
                headers = ["Date", "ICP #", "Sheet Name", "Found Profiles", "Original Query", "Sheet Link"]
                master_sheet.update("A1:F1", [headers])
                master_sheet.format("A1:F1", {
                    "textFormat": {"bold": True},
                    "horizontalAlignment": "CENTER",
                    "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}
                })
                master_sheet.freeze(rows=1)
                
                # Auto-resize columns for Master Sheet
                try:
                    master_sheet.columns_auto_resize(0, 5)  # Resize columns A-F
                except Exception as e:
                    logger.warning(f"Failed to auto-resize columns for Master Sheet: {str(e)}")
                
                start_row = 2  # Start at row 2 if empty or headers just added
            else:
                # Use the next row after existing data without skipping rows
                start_row = existing_rows + 1
                
        except WorksheetNotFound:
            master_sheet = spreadsheet.add_worksheet(title="Master_Sheet", rows=1000, cols=10)
            # Add headers
            headers = ["Date", "ICP #", "Sheet Name", "Found Profiles", "Original Query", "Sheet Link"]
            master_sheet.update("A1:F1", [headers])
            master_sheet.format("A1:F1", {
                "textFormat": {"bold": True},
                "horizontalAlignment": "CENTER",
                "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}
            })
            master_sheet.freeze(rows=1)
            
            # Auto-resize columns for Master Sheet
            try:
                master_sheet.columns_auto_resize(0, 5)  # Resize columns A-F
            except Exception as e:
                logger.warning(f"Failed to auto-resize columns for Master Sheet: {str(e)}")
            
            start_row = 2
        
        # Add timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Add each ICP's data
        row_data = []
        sheet_id = os.environ.get("GOOGLE_SHEET_ID")
        base_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit#gid="
        
        for i, icp_data in enumerate(icp_results):
            sheet_name = generate_sheet_name(icp_data["original"], i)
            
            # Get the actual sheet name in case it was made unique
            try:
                actual_sheet_name = ensure_unique_sheet_name(spreadsheet, sheet_name)
            except Exception:
                actual_sheet_name = sheet_name
            
            result_count = icp_data.get("result_count", 0)
            original_query = icp_data["original"]
            
            # Truncate long queries
            if len(original_query) > 100:
                original_query = original_query[:97] + "..."
            
            # Create a link to the sheet
            try:
                worksheet = spreadsheet.worksheet(actual_sheet_name)
                sheet_gid = worksheet.id
                sheet_link = f"{base_url}{sheet_gid}"
            except Exception:
                sheet_link = ""
            
            row_data.append([
                timestamp,
                i + 1,
                actual_sheet_name,
                result_count,
                original_query,
                sheet_link
            ])
        
        # Make sure we don't exceed sheet limits
        max_rows = 999  # Google Sheets has a limit of 1000 rows by default
        
        # Update the master sheet with the data
        if row_data:
            # Check if adding these rows would exceed limits
            end_row = start_row + len(row_data) - 1
            if end_row > max_rows:
                # Truncate the data to fit within limits
                rows_to_add = max_rows - start_row + 1
                if rows_to_add > 0:
                    logger.warning(f"Truncating data to fit within sheet limits: adding {rows_to_add} of {len(row_data)} rows")
                    row_data = row_data[:rows_to_add]
                else:
                    logger.warning("Cannot add more rows to Master_Sheet, it's at capacity")
                    return
            
            # Update the data starting at the next row after existing data
            master_sheet.update(f"A{start_row}:F{start_row + len(row_data) - 1}", row_data)
        
        # Format the data rows
        if row_data:
            master_sheet.format(f"A{start_row}:F{start_row + len(row_data) - 1}", {
                "wrapStrategy": "CLIP",
                "verticalAlignment": "TOP"
            })
            
            # Make sheet links clickable
            for row_num in range(start_row, start_row + len(row_data)):
                try:
                    master_sheet.update_cell(row_num, 6, f'=HYPERLINK("{row_data[row_num-start_row][5]}", "Open Sheet")')
                except Exception as e:
                    logger.warning(f"Failed to set hyperlink in cell F{row_num}: {str(e)}")
        
        # Auto-resize columns to show all content properly
        try:
            master_sheet.columns_auto_resize(0, 5)  # Resize columns A-F
        except Exception as e:
            logger.warning(f"Failed to auto-resize columns in master sheet: {str(e)}")
        
        logger.info(f"Updated Master_Sheet with {len(row_data)} new entries starting at row {start_row}")
    
    except Exception as e:
        logger.error(f"Failed to update Master_Sheet: {str(e)}")
        # Continue without master sheet - non-critical


def check_spreadsheet_access(spreadsheet):
    """
    Check if we have sufficient permissions for the spreadsheet.
    
    Args:
        spreadsheet: Google Spreadsheet object
        
    Returns:
        True if we have edit access, False otherwise
    """
    try:
        # Try to get the spreadsheet properties (requires read access)
        properties = spreadsheet.fetch_sheet_metadata()
        
        # Try to add a test worksheet (requires write access)
        test_title = f"test_access_{int(time.time())}"
        test_sheet = spreadsheet.add_worksheet(title=test_title, rows=1, cols=1)
        
        # If we got here, we have edit access - clean up the test sheet
        spreadsheet.del_worksheet(test_sheet)
        return True
    
    except Exception as e:
        logger.error(f"Insufficient permissions for spreadsheet: {str(e)}")
        return False


def get_spreadsheet(sheet_id: str):
    """
    Get the spreadsheet object with error handling.
    
    Args:
        sheet_id: Google Sheet ID
        
    Returns:
        Spreadsheet object
    """
    try:
        client = get_google_sheets_client()
        spreadsheet = client.open_by_key(sheet_id)
        
        # Check if we have sufficient permissions
        if not check_spreadsheet_access(spreadsheet):
            raise ValueError("Insufficient permissions for the Google Sheet. Make sure you have edit access.")
        
        return spreadsheet
    
    except SpreadsheetNotFound:
        raise ValueError(f"Spreadsheet with ID '{sheet_id}' not found. Check your GOOGLE_SHEET_ID.")
    
    except Exception as e:
        logger.error(f"Failed to open spreadsheet: {str(e)}")
        raise ValueError(f"Error accessing Google Sheet: {str(e)}")


def append_rows(profiles: List[Profile], main_sheet: bool = False):
    """
    Append profile data to Google Sheets (main sheet).
    
    Args:
        profiles: List of Profile objects to append
        main_sheet: If True, use the main results sheet. Otherwise use "All Results" sheet.
    """
    if not profiles:
        logger.info("No profiles to append to Google Sheets")
        return
    
    sheet_id = os.environ.get("GOOGLE_SHEET_ID")
    if not sheet_id:
        raise ValueError("Missing GOOGLE_SHEET_ID environment variable")
    
    try:
        # Get the spreadsheet
        spreadsheet = get_spreadsheet(sheet_id)
        
        # Get/Create All Results sheet
        sheet_name = "Main Results" if main_sheet else "All Results"
        all_results_sheet = create_or_get_worksheet(spreadsheet, sheet_name)
        
        # Append the profiles
        append_rows_to_worksheet(all_results_sheet, profiles)
        
        logger.info(f"Successfully appended {len(profiles)} profiles to {sheet_name} sheet")
        
    except Exception as e:
        logger.error(f"Error appending to Google Sheets: {str(e)}")
        raise


def append_icp_results(icp_results: List[Dict]):
    """
    Create separate sheets for each ICP and append results.
    
    Args:
        icp_results: List of dictionaries with ICP data and results
            Each dict should have:
            - "original": Original ICP text
            - "optimized": Optimized query
            - "results": Raw search results
    """
    if not icp_results:
        logger.info("No ICP results to append")
        return
    
    sheet_id = os.environ.get("GOOGLE_SHEET_ID")
    if not sheet_id:
        raise ValueError("Missing GOOGLE_SHEET_ID environment variable")
    
    try:
        # Get the spreadsheet
        spreadsheet = get_spreadsheet(sheet_id)
        
        # Generate sheet names with Gemini
        gemini_names = generate_sheet_names_with_gemini(icp_results)
        
        # Track processed LinkedIn URLs to avoid duplicates within each ICP
        for i, icp_data in enumerate(icp_results):
            # Generate sheet name based on ICP content
            sheet_name = generate_sheet_name(icp_data["original"], i, gemini_names)
            
            # Get or create worksheet with unique name
            worksheet = create_or_get_worksheet(spreadsheet, sheet_name)
            
            # Normalize results for this ICP
            from src.transform import normalize_results
            icp_profiles = normalize_results(icp_data.get("results", []))
            
            # Append profiles to the worksheet
            append_rows_to_worksheet(worksheet, icp_profiles)
            
            # We're removing the ICP information from columns to avoid headers issues
            
        # Create a summary sheet
        create_summary_sheet(spreadsheet, icp_results)
        
        logger.info(f"Successfully created and populated {len(icp_results)} ICP-specific sheets")
        
    except Exception as e:
        logger.error(f"Error creating ICP-specific sheets: {str(e)}")
        raise

# Add retry decorators for common gspread operations that may hit quota limits
@retry(
    retry=retry_if_exception_type(APIError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    reraise=True
)
def batch_update_with_retry(worksheet, data):
    """
    Wrapper for worksheet.batch_update with exponential backoff retry logic.
    
    Args:
        worksheet: gspread worksheet
        data: Data to update in batch
        
    Returns:
        Result from worksheet.batch_update
    """
    try:
        return worksheet.batch_update(data)
    except APIError as e:
        if "quota" in str(e).lower() or "429" in str(e):
            logger.warning(f"Google Sheets quota exceeded. Retrying with backoff: {str(e)}")
        raise

@retry(
    retry=retry_if_exception_type(APIError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    reraise=True
)
def append_rows_with_retry(worksheet, rows, value_input_option='RAW'):
    """
    Wrapper for worksheet.append_rows with exponential backoff retry logic.
    
    Args:
        worksheet: gspread worksheet
        rows: Rows to append
        value_input_option: Input option (default 'RAW')
        
    Returns:
        Result from worksheet.append_rows
    """
    try:
        return worksheet.append_rows(rows, value_input_option=value_input_option)
    except APIError as e:
        if "quota" in str(e).lower() or "429" in str(e):
            logger.warning(f"Google Sheets quota exceeded. Retrying with backoff: {str(e)}")
        raise

@retry(
    retry=retry_if_exception_type(APIError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True
)
def columns_auto_resize_with_retry(worksheet, start_column, end_column):
    """
    Wrapper for worksheet.columns_auto_resize with retry logic.
    
    Args:
        worksheet: gspread worksheet
        start_column: Start column index (0-based)
        end_column: End column index (0-based)
        
    Returns:
        Result from worksheet.columns_auto_resize
    """
    try:
        return worksheet.columns_auto_resize(start_column, end_column)
    except Exception as e:
        # Handle both APIError and attribute errors (older gspread versions)
        if isinstance(e, AttributeError):
            logger.warning(f"columns_auto_resize not available in this version of gspread: {str(e)}")
            return None
        elif "quota" in str(e).lower() or "429" in str(e):
            logger.warning(f"Google Sheets quota exceeded. Retrying with backoff: {str(e)}")
            raise
        else:
            # For other errors, log but don't retry
            logger.warning(f"Error in columns_auto_resize: {str(e)}")
            return None 