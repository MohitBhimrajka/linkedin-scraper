import os
import asyncio
import streamlit as st
import pandas as pd
import yaml
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from gspread.exceptions import WorksheetNotFound, GSpreadException
from datetime import datetime
import traceback
import re

# Import from our scraper
from src.runner import process_query, load_queries
from src.throttle import Batcher
from src.transform import normalize_results, Profile
from src.sheets import append_rows, append_icp_results, get_spreadsheet, get_google_sheets_client, generate_sheet_name, create_or_get_master_profiles_sheet
from src.logging_conf import logger
from src.campaign import run_campaign
from src.sync import sync_status


# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="LinkedIn Lead Generator",
    page_icon="🔎",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding: 0px 16px;
        font-size: 16px;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        margin-bottom: 0.5rem;
    }
    .stExpander {
        border: 1px solid #f0f2f6;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
    .success-message {
        padding: 8px;
        background-color: #d1e7dd;
        color: #0a3622;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .info-message {
        padding: 8px;
        background-color: #cfe2ff;
        color: #084298;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


def optimize_query_with_gemini(query_text, negative_keywords_str=""):
    """Use Gemini API to optimize a search query into multiple variations."""
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    
    if not gemini_api_key:
        st.error("❌ GEMINI_API_KEY not found in environment variables")
        return [query_text]  # Return original query in a list
    
    # Implement a simple rate limiter for Gemini API
    # Using a class attribute to track the last call time across function calls
    if not hasattr(optimize_query_with_gemini, "last_call_time"):
        optimize_query_with_gemini.last_call_time = 0
    
    # Minimum delay between calls (in seconds)
    min_delay = 2.0
    current_time = time.time()
    time_since_last_call = current_time - optimize_query_with_gemini.last_call_time
    
    if time_since_last_call < min_delay:
        # Need to wait to avoid rate limiting
        sleep_time = min_delay - time_since_last_call
        logger.info(f"Rate limiting Gemini API calls, sleeping for {sleep_time:.2f}s")
        time.sleep(sleep_time)
    
    # Update last call time
    optimize_query_with_gemini.last_call_time = time.time()
    
    client = genai.Client(api_key=gemini_api_key)
    model = "gemini-2.5-flash-preview-04-17"
    
    # Process negative keywords if provided
    negative_keywords_list = [kw.strip() for kw in negative_keywords_str.split(',') if kw.strip()]
    negative_keywords_for_prompt = " ".join([f"-{kw}" for kw in negative_keywords_list])
    
    prompt = f"""
    You are a world-class expert in LinkedIn search optimization and Google dorking techniques, specializing in crafting highly precise search queries to find the exact LinkedIn profiles matching specific Ideal Customer Profile (ICP) criteria.

    Your task is to transform the given "Original ICP criteria" into a JSON list of 3 to 5 DIVERSE but RELATED, HIGHLY TARGETED, OPTIMIZED Google search strings. Each string in the list will be used with a Google Custom Search Engine (CSE) configured to search within LinkedIn. The goal is to maximize the chances of finding relevant profiles by exploring different angles of the ICP.

    **Advanced Optimization Guidelines (apply to EACH generated query string):**
    1.  **Precision Over Recall:** Optimize for finding EXACTLY the right profiles, even if it means fewer results. It's better to find 10 perfect matches than 100 mediocre ones.
    2.  **Advanced Google Search Operators:**
        * Use `AND` (often implicit, but explicit for clarity when needed)
        * Use `OR` with parentheses for alternatives: `("VP Sales" OR "Sales Director")`
        * Use exclusions with `-` prefix: `-intern -assistant -junior`
        * Use exact phrases with quotes for multi-word concepts: `"product management"`
        * Use wildcards sparingly: `"head * marketing"` 
        * Use AROUND(n) for proximity searches: `marketing AROUND(3) director`
        * For company targeting, use company name variations: `("Company A" OR "CompanyA" OR "Company-A")`
    3.  **LinkedIn-Specific Keyword Optimization:**
        * **Job Titles:** Be comprehensive with exact titles AND functional equivalents
          * Example: `("Chief Revenue Officer" OR "CRO" OR "VP Sales" OR "Head of Revenue" OR "Revenue Leader")`
        * **Seniority:** Always be specific about seniority when it matters
          * Example: `(executive OR C-level OR CXO OR chief OR head OR director OR VP OR "vice president")`
        * **Industry Terms:** Use industry-specific terminology that professionals would list in their profiles
          * Example for FinTech: `("payment processing" OR "digital banking" OR "financial technology" OR fintech OR "open banking")`
        * **Location:** Include regional terms, abbreviations, and major cities
          * Example: `("San Francisco Bay Area" OR SF OR "Bay Area" OR Oakland OR Berkeley OR "Silicon Valley" OR SV)`
        * **Company Size/Stage:** Use terms professionals use in profiles
          * Example: `("Series B" OR "Series C" OR "growth stage" OR scale-up OR scaleup OR "high growth")`
    4.  **Logical Structure:** Use nested parentheses to create a clear logical structure, with most important criteria first.
    5.  **Balance AND/OR Logic:** Make sure your query combines broad enough terms to find good matches while being specific enough to exclude irrelevant profiles.
    6.  **Strategic Exclusions:** Based on the ICP, actively consider terms that should be excluded to avoid irrelevant profiles. For instance, if targeting "Sales Directors", consider excluding "-assistant", "-coordinator", "-intern", "-entry level". Incorporate these as negative keywords (e.g., `-term`).

    **Output Requirements:**
    * Return ONLY a valid JSON array of objects. Each object should have "query" and "confidence" (1-5, 5 is highest) keys.
    * Example: `[{"query": "query one", "confidence": 5}, {"query": "query two", "confidence": 4}]`
    * Do NOT include explanations, labels, or any text before/after the JSON array.
    * Do NOT include site:linkedin.com/in/ as the CSE is already configured to search only within LinkedIn.

    **Examples of Excellent Optimized Query Lists:**
    
    ORIGINAL: "Product managers at enterprise SaaS companies in Boston"
    OPTIMIZED_JSON_LIST: [
        {{"query": "(\\"product manager\\" OR \\"product owner\\") AND (\\"enterprise software\\" OR \\"B2B SaaS\\") AND (Boston OR \\"MA\\") -junior", "confidence": 5}},
        {{"query": "(\\"senior product manager\\" OR \\"group product manager\\") AND SaaS AND (\\"Boston Area\\" OR \\"Cambridge MA\\") -intern -associate", "confidence": 4}},
        {{"query": "(\\"head of product\\" OR \\"product lead\\") AND (\\"enterprise SaaS\\" OR \\"cloud software\\") AND (Massachusetts OR \\"New England\\") -consultant", "confidence": 4}}
    ]
    
    ORIGINAL: "Finance executives at Series B startups in Europe"
    OPTIMIZED_JSON_LIST: [
        {{"query": "(\\"CFO\\" OR \\"Chief Financial Officer\\" OR \\"VP Finance\\" OR \\"Head of Finance\\" OR \\"Finance Director\\") AND (\\"Series B\\" OR \\"Series C\\" OR \\"growth stage\\" OR \\"venture backed\\" OR \\"venture funded\\") AND (\\"Europe\\" OR \\"European Union\\" OR EU OR London OR Paris OR Berlin OR Amsterdam OR Madrid OR \\"Nordic\\" OR DACH)", "confidence": 5}},
        {{"query": "(\\"Finance Executive\\" OR \\"Finance Leader\\") AND (startup OR scaleup) AND (UK OR Germany OR France OR Netherlands OR Spain) AND (fintech OR \\"financial services\\") -analyst", "confidence": 4}},
        {{"query": "(\\"Head of Finance\\" OR \\"Finance Director\\") AND (\\"venture capital\\" OR \\"private equity backed\\") AND (Europe) AND (B2B OR B2C)", "confidence": 3}}
    ]

    Original ICP criteria:
    {query_text}

    User-provided negative keywords to EXCLUDE (append these to each generated query):
    {negative_keywords_for_prompt if negative_keywords_for_prompt else "None"}

    Optimized Google Search String List (JSON Array of 3-5 diverse queries):
    """
    
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]
    
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",  # Expecting JSON text
    )
    
    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        
        response_text = response.text.strip()
        
        # Attempt to parse the response as JSON
        try:
            # Extract JSON array if found within response text
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                query_variations = json.loads(json_str)
                
                # Handle both the new format (objects with query and confidence) and old format (just strings)
                if isinstance(query_variations, list):
                    if all(isinstance(q, dict) and "query" in q for q in query_variations):
                        # New format with query and confidence
                        queries_only = [item["query"] for item in query_variations]
                        logger.info(f"Generated {len(queries_only)} query variations with Gemini (with confidence scores)")
                    elif all(isinstance(q, str) for q in query_variations):
                        # Old format with just strings
                        queries_only = query_variations
                        logger.info(f"Generated {len(queries_only)} query variations with Gemini (without confidence scores)")
                    else:
                        logger.warning("Unexpected format in Gemini response, using original query")
                        queries_only = [query_text]
                else:
                    logger.warning("Gemini response was not a list, using original query")
                    queries_only = [query_text]
                
                # Ensure user-provided negative keywords are appended if Gemini didn't include them
                final_variations = []
                for q_var in queries_only:
                    current_q = q_var
                    for neg_kw in negative_keywords_list:
                        if f"-{neg_kw.lower()}" not in current_q.lower() and f"-\"{neg_kw.lower()}\"" not in current_q.lower():
                            current_q += f" -{neg_kw}"  # Append as simple negative keyword
                    final_variations.append(current_q)
                
                return final_variations
            else:
                logger.warning("No JSON array found in Gemini response, using original query")
        except json.JSONDecodeError:
            logger.warning("Failed to parse Gemini response as JSON, using original query")
        
        # If we couldn't parse or the response wasn't valid, return the original query as a single-item list
        return [query_text]
        
    except Exception as e:
        st.error(f"❌ Error optimizing ICP criteria with Gemini: {str(e)}")
        return [query_text]  # Return original query in a list in case of error


def save_queries_to_yaml(queries):
    """Save the list of ICP queries to the YAML file."""
    # Handle both legacy format (list of strings) and new format (list of dicts)
    query_data = {"queries": []}
    
    for q in queries:
        if isinstance(q, str):
            # Legacy format, just a string description
            query_data["queries"].append({
                "description": q,
                "negative_keywords": ""
            })
        elif isinstance(q, dict):
            # New format with description and negative_keywords
            query_data["queries"].append(q)
    
    with open("config/queries.yaml", "w") as f:
        yaml.dump(query_data, f, default_flow_style=False)
    logger.info(f"Saved {len(queries)} ICPs to config/queries.yaml")


async def optimize_all_icps(icps):
    """Optimize all ICPs using Gemini."""
    optimized_icps = []
    
    for icp in icps:
        optimized_icp = optimize_query_with_gemini(icp)
        optimized_icps.append({
            "original": icp,
            "optimized": optimized_icp
        })
    
    return optimized_icps


async def run_scraper(optimized_icps, limit):
    """Run the LinkedIn scraper with the provided optimized ICP criteria."""
    # Check for required environment variables
    required_vars = ["GOOGLE_API_KEY", "CX_ID", "GOOGLE_SHEET_ID"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        st.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        return None, optimized_icps
    
    # Extract original ICP queries (not optimized yet)
    original_queries = [icp["original"] for icp in optimized_icps]
    logger.info(f"Starting LinkedIn profile collection with {len(original_queries)} ICPs (limit: {limit})")
    
    # Create Batcher for rate limiting and parallelism
    batcher = Batcher(max_in_flight=5, delay=3)
    
    # Process each ICP criteria concurrently using the enhanced function
    all_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process all ICPs with query variations
    all_icp_data = await _process_all_icps_concurrently(
        original_queries, 
        limit, 
        batcher, 
        progress_bar, 
        status_text
    )
    
    # Process and collect all results from all variations
    for icp_data in all_icp_data:
        all_results.extend(icp_data["results"])
    
    # Update optimized_icps with the new data structure from _process_all_icps_concurrently
    for i, icp_data in enumerate(all_icp_data):
        optimized_icps[i] = {
            "original": icp_data["original"],
            "optimized": icp_data["optimized_variations"][0] if icp_data["optimized_variations"] else icp_data["original"],  # Use first variation as primary
            "optimized_variations": icp_data["optimized_variations"],
            "results": icp_data["results"],
            "result_count": icp_data["result_count"]
        }
    
    # Normalize and deduplicate results
    unique_profiles = normalize_results(all_results)
    
    # Append to Google Sheets - both to main sheet and individual ICP sheets
    if unique_profiles:
        logger.info(f"Appending {len(unique_profiles)} profiles to Google Sheets...")
        status_text.text(f"Appending {len(unique_profiles)} profiles to Google Sheets...")
        
        # Append to main sheet
        append_rows(unique_profiles, main_sheet=True)
        
        # Create separate sheets for each ICP
        append_icp_results(optimized_icps)
        
        status_text.text("✅ LinkedIn profile collection completed successfully")
        
        # Convert profiles to DataFrame for display
        profiles_data = []
        for profile in unique_profiles:
            profiles_data.append({
                "LinkedIn URL": profile.linkedin_url,
                "Title": profile.title or "",
                "Name": f"{profile.first_name or ''} {profile.last_name or ''}".strip(),
                "Company": profile.company if hasattr(profile, "company") else "",
                "Location": profile.location if hasattr(profile, "location") else "",
                "Description": profile.description or ""
            })
        
        return pd.DataFrame(profiles_data), optimized_icps
    else:
        status_text.text("⚠️ No profiles found to append")
        return None, optimized_icps


async def cleanup_spreadsheet(show_progress=True):
    """
    Clean up the spreadsheet by removing all sheets except Master_Sheet and clearing runs in Master_Sheet.
    
    Args:
        show_progress: Whether to show progress in Streamlit UI
    """
    sheet_id = os.environ.get("GOOGLE_SHEET_ID")
    if not sheet_id:
        st.error("❌ Missing GOOGLE_SHEET_ID environment variable")
        return False
    
    try:
        # Get the spreadsheet
        spreadsheet = get_spreadsheet(sheet_id)
        
        # Get all worksheets - create a copy of the list to safely iterate
        worksheets = list(spreadsheet.worksheets())
        
        # Progress tracking
        if show_progress:
            progress_text = st.empty()
            progress_bar = st.progress(0)
            progress_text.text("Cleaning up spreadsheet...")
        
        # Counter for deleted sheets
        deleted_count = 0
        
        # Process each worksheet
        for i, worksheet in enumerate(worksheets):
            sheet_title = worksheet.title
            
            if show_progress:
                progress_bar.progress((i + 1) / len(worksheets))
            
            if sheet_title == "Master_Sheet" or sheet_title == "Master_Profiles":
                # Clear all data in Master_Sheet except headers
                try:
                    # Get all values to determine the size
                    all_values = worksheet.get_all_values()
                    if len(all_values) > 1:  # If there are rows beyond the header
                        # Clear everything after row 1
                        worksheet.batch_clear([f"A2:Z{len(all_values)}"])
                        if show_progress:
                            progress_text.text(f"Cleared all data from {sheet_title}")
                except Exception as e:
                    if show_progress:
                        st.warning(f"Could not clear {sheet_title}: {str(e)}")
                    logger.warning(f"Failed to clear {sheet_title}: {str(e)}")
            elif sheet_title != "Sheet1":  # Skip the default Sheet1
                # Delete all other sheets
                try:
                    # Check if this is the current Master_Sheet before deleting
                    spreadsheet.del_worksheet(worksheet)
                    deleted_count += 1
                    if show_progress:
                        progress_text.text(f"Deleted sheet: {sheet_title}")
                except Exception as e:
                    if show_progress:
                        st.warning(f"Could not delete sheet {sheet_title}: {str(e)}")
                    logger.warning(f"Failed to delete sheet {sheet_title}: {str(e)}")
        
        # Complete progress
        if show_progress:
            progress_bar.progress(1.0)
            progress_text.text(f"✅ Cleanup complete! Deleted {deleted_count} sheets")
            time.sleep(1)  # Show the completion message briefly
        
        logger.info(f"Spreadsheet cleanup completed: deleted {deleted_count} sheets")
        return True
    
    except Exception as e:
        error_msg = f"Failed to clean up spreadsheet: {str(e)}"
        if show_progress:
            st.error(f"❌ {error_msg}")
        logger.error(error_msg)
        return False


async def fix_master_sheet_formatting():
    """
    Fix the Master_Sheet by ensuring it has proper headers and no blank rows.
    """
    sheet_id = os.environ.get("GOOGLE_SHEET_ID")
    if not sheet_id:
        st.error("❌ Missing GOOGLE_SHEET_ID environment variable")
        return False
    
    try:
        # Get the spreadsheet
        spreadsheet = get_spreadsheet(sheet_id)
        
        try:
            # Check if Master_Sheet exists
            master_sheet = spreadsheet.worksheet("Master_Sheet")
            
            # Get all existing values
            existing_values = master_sheet.get_all_values()
            
            if not existing_values:
                # Sheet is completely empty, add headers
                headers = ["Date", "ICP #", "Sheet Name", "Found Profiles", "Original Query", "Sheet Link"]
                master_sheet.update("A1:F1", [headers])
                master_sheet.format("A1:F1", {
                    "textFormat": {"bold": True},
                    "horizontalAlignment": "CENTER",
                    "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}
                })
                master_sheet.freeze(rows=1)
                st.success("✅ Added headers to empty Master_Sheet")
                return True
            
            # Check if headers are missing or incorrect
            if len(existing_values[0]) < 6 or existing_values[0][0] != "Date" or existing_values[0][5] != "Sheet Link":
                # Headers missing or incorrect, add them
                headers = ["Date", "ICP #", "Sheet Name", "Found Profiles", "Original Query", "Sheet Link"]
                master_sheet.update("A1:F1", [headers])
                master_sheet.format("A1:F1", {
                    "textFormat": {"bold": True},
                    "horizontalAlignment": "CENTER",
                    "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}
                })
                master_sheet.freeze(rows=1)
                st.success("✅ Fixed headers in Master_Sheet")
            
            # Check for and remove blank rows
            if len(existing_values) > 1:
                non_blank_rows = [row for row in existing_values[1:] if any(cell.strip() if isinstance(cell, str) else cell for cell in row)]
                
                if len(non_blank_rows) < len(existing_values) - 1:
                    # Clear sheet and rewrite without blank rows
                    master_sheet.batch_clear([f"A2:F{len(existing_values)}"])
                    
                    if non_blank_rows:
                        master_sheet.update(f"A2:F{len(non_blank_rows)+1}", non_blank_rows)
                        
                        # Fix hyperlinks
                        for i, row in enumerate(non_blank_rows):
                            row_num = i + 2  # +2 because row 1 is headers and we're 0-indexed
                            if len(row) >= 6 and row[5] and "http" in str(row[5]):
                                link = row[5]
                                if not link.startswith("=HYPERLINK"):
                                    try:
                                        master_sheet.update_cell(row_num, 6, f'=HYPERLINK("{link}", "Open Sheet")')
                                    except Exception:
                                        pass
                    
                    st.success(f"✅ Removed {len(existing_values) - 1 - len(non_blank_rows)} blank rows from Master_Sheet")
            
            # Make sure columns are properly sized
            try:
                master_sheet.columns_auto_resize(0, 5)  # Resize columns A-F
            except Exception:
                pass
                
            return True
            
        except WorksheetNotFound:
            # Sheet doesn't exist, will be created later
            logger.info("Master_Sheet doesn't exist yet, will be created on first run")
            return True
            
    except Exception as e:
        error_msg = f"Failed to fix Master_Sheet: {str(e)}"
        st.error(f"❌ {error_msg}")
        logger.error(error_msg)
        return False


# Functions to help display metrics/stats from the data
def get_relationship_stats(df):
    """Extract relationship stats from DataFrame for metrics display"""
    
    # Ensure columns exist, provide default empty Series if not
    contact_status_col = df["Contact Status"] if "Contact Status" in df.columns else pd.Series(dtype=str)
    connection_state_col = df["Connection State"] if "Connection State" in df.columns else pd.Series(dtype=str)
    unread_cnt_col = df["Unread Cnt"] if "Unread Cnt" in df.columns else pd.Series(0, index=df.index, dtype=int)

    # Count invites based on either Contact Status or Connection State
    invites_sent_count = len(df[
        (contact_status_col.str.contains("Invited", case=False, na=False)) | 
        (connection_state_col.isin(["INVITED", "PENDING"]))
    ])
    
    # Count comments based on Contact Status containing "Comment"
    comments_count = len(df[contact_status_col.str.contains("Comment", case=False, na=False)])
    
    stats = {
        "total": len(df),
        "sent": invites_sent_count,
        "connected": len(df[connection_state_col == "CONNECTED"]),
        "unread": unread_cnt_col.sum(),
        "comments": comments_count
    }
    return stats


def _determine_next_action(connection_state, contact_status, connection_actions, workflow_stages):
    """Determine the next recommended action based on current status"""
    # Handle NaN or empty values
    if pd.isna(connection_state) or connection_state == "":
        connection_state = "NOT_CONNECTED"
    
    if pd.isna(contact_status) or contact_status == "":
        contact_status = "Not contacted"
    
    # Normalize contact status for consistent matching
    contact_status_lower = contact_status.lower() if isinstance(contact_status, str) else ""
    
    # Map status for new leads to ensure they're not classified as "profile discovered"
    if connection_state == "NOT_CONNECTED" and (contact_status_lower == "not contacted" or "profile discovered" in contact_status_lower):
        # For new leads that have just been discovered but not contacted yet
        return "Generate messages"  # Default action for new leads
    
    # First check workflow stage (with more detailed matching)
    if contact_status in workflow_stages:
        return workflow_stages[contact_status][0]  # Return first recommended action
    else:
        # Try a case-insensitive partial match for more flexible status matching
        for stage, actions in workflow_stages.items():
            if stage.lower() in contact_status_lower:
                return actions[0]  # Return first recommended action
    
    # Then check connection state
    if connection_state in connection_actions:
        return connection_actions[connection_state][0]  # Return first recommended action
    
    # Default action if we can't determine
    return "Generate messages"


async def _process_all_icps_concurrently(queries_to_process, limit_per_icp, batcher_instance, st_progress_bar, st_progress_text):
    """
    Process multiple ICPs in parallel with support for multiple query variations per ICP.
    
    Args:
        queries_to_process: List of ICP dictionaries with description and negative_keywords
        limit_per_icp: Max number of profiles to fetch per ICP query variation
        batcher_instance: BatchRequestManager instance
        st_progress_bar: Streamlit progress bar
        st_progress_text: Streamlit text element for status updates
        
    Returns:
        List of dictionaries with original and optimized ICPs along with results
    """
    all_optimized_icps_data = [] # This will be returned
    optimization_step_weight = 0.1  # 10% for optimization
    processing_step_weight = 0.9    # 90% for fetching

    num_original_icps = len(queries_to_process)

    # Step 1: Optimize all original ICPs to get lists of query variations
    optimized_query_collections = []
    for i, icp_item in enumerate(queries_to_process):
        st_progress_text.text(f"Optimizing ICP #{i+1}/{num_original_icps}...")
        
        # Extract description and negative_keywords from the ICP item
        if isinstance(icp_item, str):
            # Legacy format (just a string)
            original_query_text = icp_item
            negative_keywords = ""
        elif isinstance(icp_item, dict):
            # New format (dictionary with description and negative_keywords)
            original_query_text = icp_item.get("description", "")
            negative_keywords = icp_item.get("negative_keywords", "")
        else:
            logger.warning(f"Unexpected ICP format: {type(icp_item)}. Skipping.")
            continue
        
        # optimize_query_with_gemini now accepts negative_keywords
        optimized_variations_list = optimize_query_with_gemini(original_query_text, negative_keywords)
        
        # Limit the number of variations to 3 to reduce API load
        if len(optimized_variations_list) > 3:
            logger.info(f"Limiting query variations from {len(optimized_variations_list)} to 3 to reduce API load")
            optimized_variations_list = optimized_variations_list[:3]
        
        optimized_query_collections.append({
            "original": original_query_text,
            "negative_keywords": negative_keywords,
            "optimized_variations": optimized_variations_list, # List of query strings
            "results": [],  # Placeholder for aggregated results for this original ICP
            "result_count": 0  # Placeholder
        })
        st_progress_bar.progress(((i + 1) / num_original_icps) * optimization_step_weight)
    
    # Step 2 & 3: Create and run fetch_profile tasks for EACH optimized query variation
    fetch_tasks = []
    # Keep track of which original ICP each task belongs to, and which variation it is
    task_to_icp_map = [] 

    for icp_idx, collection_detail in enumerate(optimized_query_collections):
        original_icp_text = collection_detail["original"]
        variations = collection_detail["optimized_variations"]
        for variation_idx, optimized_variation_text in enumerate(variations):
            st_progress_text.text(f"Preparing fetch for ICP #{icp_idx+1} (Var. {variation_idx+1}/{len(variations)}): {original_icp_text[:30]}...")
            fetch_tasks.append(
                process_query(optimized_variation_text, limit=limit_per_icp, batcher=batcher_instance)
            )
            task_to_icp_map.append({"icp_idx": icp_idx, "variation_text": optimized_variation_text})
            # Add a small delay between creating tasks to prevent overwhelming the API
            await asyncio.sleep(0.5)

    # Run all fetch_tasks concurrently
    all_variation_results_raw = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    # Step 4: Collate results back to their original ICPs
    # Initialize/reset results list for each original ICP in optimized_query_collections
    for collection_detail in optimized_query_collections:
        collection_detail["results"] = [] 

    current_progress = optimization_step_weight
    num_total_fetch_tasks = len(fetch_tasks)

    for task_idx, result_or_exception in enumerate(all_variation_results_raw):
        map_info = task_to_icp_map[task_idx]
        original_icp_idx = map_info["icp_idx"]
        variation_text = map_info["variation_text"] # For logging
        
        # Get the collection_detail for the original ICP this result belongs to
        collection_detail_for_this_result = optimized_query_collections[original_icp_idx]

        if isinstance(result_or_exception, Exception):
            st.error(f"Error processing variation '{variation_text[:50]}...' for ICP '{collection_detail_for_this_result['original'][:30]}...': {str(result_or_exception)}")
            logger.error(f"Exception during process_query for variation '{variation_text}' of ICP '{collection_detail_for_this_result['original']}': {result_or_exception}")
        else:
            # result_or_exception is a list of profile dicts from one call to process_query
            collection_detail_for_this_result["results"].extend(result_or_exception) # Accumulate results
        
        current_progress += (1 / num_total_fetch_tasks) * processing_step_weight if num_total_fetch_tasks > 0 else 0
        st_progress_bar.progress(min(current_progress, 1.0))
        st_progress_text.text(f"Processed fetch task {task_idx+1}/{num_total_fetch_tasks}. Aggregating results...")
        
    # After all variations are processed, update the final result_count for each original ICP
    # The results are already aggregated in collection_detail["results"]
    for i, final_detail in enumerate(optimized_query_collections):
        final_detail["result_count"] = len(final_detail["results"])
        all_optimized_icps_data.append(final_detail) # This is what the function returns
        st_progress_text.text(f"ICP #{i+1}/{num_original_icps} ('{final_detail['original'][:30]}...') aggregated {final_detail['result_count']} raw profiles from all its variations.")
        
    return all_optimized_icps_data


def main():
    # Initialize session state variables
    if 'page' not in st.session_state:
        st.session_state.page = 'setup'  # 'setup', 'processing', 'results'
    
    if 'queries' not in st.session_state:
        # Try to load existing queries from YAML
        try:
            loaded_queries = load_queries()
            st.session_state.queries = loaded_queries if loaded_queries else []
        except Exception:
            st.session_state.queries = []
    
    if 'optimized_icps' not in st.session_state:
        st.session_state.optimized_icps = []
    
    # Page title and header
    st.title("🔎 LinkedIn Lead Generator")
    
    # Get available sheets
    sheet_id = os.getenv("GOOGLE_SHEET_ID")
    try:
        spreadsheet = get_spreadsheet(sheet_id)
        sheet_titles = [ws.title for ws in spreadsheet.worksheets() if ws.title not in ("Master_Sheet", "Main Results")]
    except Exception as e:
        st.error(f"Error loading sheets: {str(e)}")
        sheet_titles = []
    
    # Create the main application tabs
    generate_tab, send_tab, sync_tab = st.tabs(["Generate ▶", "Send 🚀", "Sync & Monitor 🔄"])
    
    # GENERATE TAB - Find and generate LinkedIn leads
    with generate_tab:
        if st.session_state.page == 'setup':
            st.subheader("Find LinkedIn profiles matching your Ideal Customer Profiles")
            
            # Create tabs for setup and management
            api_keys_tab, icp_tab = st.tabs(["API Setup", "ICP Management"])
            
            with api_keys_tab:
                st.subheader("API Keys and Credentials")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Required Keys")
                    google_api_key = os.environ.get("GOOGLE_API_KEY", "")
                    cx_id = os.environ.get("CX_ID", "")
                    sheet_id = os.environ.get("GOOGLE_SHEET_ID", "")
                    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
                    unipile_api_key = os.environ.get("UNIPILE_API_KEY", "")
                    unipile_dsn = os.environ.get("UNIPILE_DSN", "")
                    unipile_account_id = os.environ.get("UNIPILE_ACCOUNT_ID", "")
                    
                    st.text_input("Google API Key", value=google_api_key, type="password", 
                                help="Required for Google Custom Search API")
                    st.text_input("Custom Search Engine ID (CX)", value=cx_id, type="password", 
                                help="ID of your Google Custom Search Engine")
                    st.text_input("Google Sheet ID", value=sheet_id, 
                                help="ID of the Google Sheet where results will be stored")
                    st.text_input("Gemini API Key", value=gemini_api_key, type="password", 
                                help="Required for ICP optimization (automatic)")
                
                with col2:
                    st.markdown("### Unipile API Keys")
                    st.text_input("Unipile API Key", value=unipile_api_key, type="password",
                                help="Required for LinkedIn interactions")
                    st.text_input("Unipile DSN", value=unipile_dsn, type="password",
                                help="Your Unipile instance DNS")
                    st.text_input("Unipile Account ID", value=unipile_account_id, type="password",
                                help="Your Unipile account identifier")
                
                st.markdown("""
                > ℹ️ These values are read from your `.env` file. 
                > Edit the `.env` file to change them permanently.
                """)
                
            with icp_tab:
                st.subheader("Ideal Customer Profile (ICP) Management")
                
                # ICP management UI
                left_col, right_col = st.columns([2, 1])
                
                with left_col:
                    # ICP Editor
                    st.markdown("### Add New ICP")
                    
                    # Input for new ICP
                    new_icp_key = "new_icp_input"  # Add a key for tracking this specific input
                    
                    # Initialize the "clear_input" flag if it doesn't exist
                    if "clear_input" not in st.session_state:
                        st.session_state.clear_input = False
                    
                    # Set initial value based on clear flag
                    initial_value = "" if st.session_state.clear_input else st.session_state.get(new_icp_key, "")
                    # Reset the clear flag
                    if st.session_state.clear_input:
                        st.session_state.clear_input = False
                    
                    new_icp = st.text_area(
                        "Enter Ideal Customer Profile criteria", 
                        value=initial_value,
                        height=100,
                        key=new_icp_key,
                        help="Describe your ideal customer profile in natural language. Example: 'Product managers at enterprise SaaS companies in Boston'"
                    )
                    
                    # Add field for negative keywords
                    new_icp_negative_keywords = st.text_input(
                        "Negative Keywords (optional, comma-separated)", 
                        key="new_icp_negative_keywords_input",
                        help="Terms to exclude from search results, e.g.: intern, assistant, junior, contract"
                    )
                    
                    # Button to add new ICP
                    col1, col2 = st.columns([1, 5])
                    with col1:
                        if st.button("Add ICP", key="add_icp_button", disabled=not new_icp.strip()):
                            if new_icp.strip():
                                # Add new ICP with negative keywords
                                new_icp_dict = {
                                    "description": new_icp.strip(),
                                    "negative_keywords": new_icp_negative_keywords.strip()
                                }
                                st.session_state.queries.append(new_icp_dict)
                                save_queries_to_yaml(st.session_state.queries)
                                
                                # Clear the input field for the next entry
                                st.session_state.clear_input = True
                                # Also clear the negative keywords input
                                st.session_state.new_icp_negative_keywords_input = ""
                                
                                st.success(f"✅ Added new ICP: {new_icp.strip()[:50]}{'...' if len(new_icp.strip()) > 50 else ''}")
                                # Rerun to refresh the UI
                                st.experimental_rerun()
                
                with right_col:
                    # Show existing ICPs with delete buttons
                    st.markdown("### Existing ICPs")
                    
                    if not st.session_state.queries:
                        st.info("No ICPs added yet. Add your first ICP on the left.")
                    else:
                        for i, query_item in enumerate(st.session_state.queries):
                            # Handle both string and dict formats
                            if isinstance(query_item, str):
                                description = query_item
                                negative_keywords = ""
                            else:
                                description = query_item.get("description", "")
                                negative_keywords = query_item.get("negative_keywords", "")
                            
                            with st.expander(f"ICP #{i+1}: {description[:50]}{'...' if len(description) > 50 else ''}", expanded=False):
                                # Display the full ICP description
                                st.text_area(f"ICP #{i+1} Description", value=description, height=100, key=f"icp_{i}_description")
                                
                                # Display and allow editing of negative keywords
                                edited_negative_keywords = st.text_input(
                                    f"Negative Keywords (comma-separated)", 
                                    value=negative_keywords,
                                    key=f"icp_{i}_negative_keywords"
                                )
                                
                                col1, col2 = st.columns([1, 1])
                                with col1:
                                    # Update button
                                    if st.button("Update", key=f"update_icp_{i}"):
                                        # Get the updated values
                                        updated_description = st.session_state[f"icp_{i}_description"]
                                        updated_negative_keywords = st.session_state[f"icp_{i}_negative_keywords"]
                                        
                                        # Update the ICP
                                        st.session_state.queries[i] = {
                                            "description": updated_description.strip(),
                                            "negative_keywords": updated_negative_keywords.strip()
                                        }
                                        save_queries_to_yaml(st.session_state.queries)
                                        st.success(f"✅ Updated ICP #{i+1}")
                                        # Rerun to refresh the UI
                                        st.experimental_rerun()
                                
                                with col2:
                                    # Delete button
                                    if st.button("Delete", key=f"delete_icp_{i}"):
                                        # Remove this ICP
                                        removed_icp = st.session_state.queries.pop(i)
                                        save_queries_to_yaml(st.session_state.queries)
                                        if isinstance(removed_icp, dict):
                                            st.success(f"✅ Removed ICP: {removed_icp.get('description', '')[:50]}{'...' if len(removed_icp.get('description', '')) > 50 else ''}")
                                        else:
                                            st.success(f"✅ Removed ICP: {removed_icp[:50]}{'...' if len(removed_icp) > 50 else ''}")
                                        # Rerun to refresh the UI
                                        st.experimental_rerun()
                    
                    # Run settings after the ICPs
                    st.markdown("---")
                    
                    # Run button - make this more prominent
                    st.markdown("### Run Settings")
                    
                    # Add cleanup toggle
                    fresh_run = st.toggle(
                        "Clean up previous runs", 
                        value=False,
                        help="Delete all previous sheets and clear Master_Sheet before running"
                    )
                    
                    # Add fix Master_Sheet toggle
                    fix_master = st.toggle(
                        "Fix Master_Sheet formatting",
                        value=True,
                        help="Ensure proper headers and remove blank rows in Master_Sheet"
                    )
                    
                    limit = st.slider(
                        "Max results per Google Search query variation (Google CSE limit: 100)", # Clarified text
                        min_value=10, 
                        max_value=200, # Allow slider up to 200, PageIterator will cap at 100 for each CSE call
                        value=100, 
                        step=10,
                        help="Google Custom Search fetches up to 100 results per individual query. Multiple AI-generated query variations are used to increase total profile discovery."
                    )
                    
                    if st.button("▶️ Run LinkedIn Scraper", type="primary", use_container_width=True):
                        if not st.session_state.queries:
                            st.error("❌ No ICPs to run")
                        else:
                            # Clean up if requested
                            if fresh_run:
                                st.info("Cleaning up previous runs...")
                                cleanup_success = asyncio.run(cleanup_spreadsheet(show_progress=True))
                                if not cleanup_success:
                                    st.warning("Cleanup had some issues but we'll continue with the run.")
                                    
                            # Fix Master_Sheet if requested
                            if fix_master:
                                st.info("Checking Master_Sheet formatting...")
                                fix_success = asyncio.run(fix_master_sheet_formatting())
                                if not fix_success:
                                    st.warning("Could not fix Master_Sheet formatting, but we'll continue with the run.")
                            
                            # Store the limit for use in processing page
                            st.session_state.run_limit = limit
                            # Switch to processing page
                            st.session_state.page = 'processing'
                            st.rerun()
                    
                    # Create a row with two buttons side by side
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🧹 Clean Up Spreadsheet", use_container_width=True):
                            st.info("Cleaning up spreadsheet...")
                            cleanup_success = asyncio.run(cleanup_spreadsheet(show_progress=True))
                            if cleanup_success:
                                st.success("✅ Spreadsheet cleanup completed successfully!")
                            else:
                                st.error("❌ Spreadsheet cleanup failed")
                    
                    with col2:
                        if st.button("🛠️ Fix Master_Sheet", use_container_width=True):
                            st.info("Fixing Master_Sheet formatting...")
                            fix_success = asyncio.run(fix_master_sheet_formatting())
                            if fix_success:
                                st.success("✅ Master_Sheet formatting fixed successfully!")
                            else:
                                st.error("❌ Failed to fix Master_Sheet formatting")
            
        # Handle the processing page in the Generate tab
        elif st.session_state.page == 'processing':
            st.subheader("Processing ICPs")
            progress_text = st.empty()
            progress_bar = st.progress(0.0)
            
            try:
                limit = st.session_state.get('run_limit', 100)
                queries = st.session_state.queries
                progress_text.text(f"Preparing to process {len(queries)} ICPs (max {limit} results each)...")
                
                batcher = Batcher(max_in_flight=5, delay=3)  # Create Batcher once
                
                # Run all async tasks together
                all_optimized_icps = asyncio.run(
                    _process_all_icps_concurrently(queries, limit, batcher, progress_bar, progress_text)
                )
                st.session_state.optimized_icps = all_optimized_icps
                
                progress_text.text("Saving all results to Google Sheets...")
                progress_bar.progress(0.95)  # Progress before sheet append
                
                if any(icp['result_count'] > 0 for icp in all_optimized_icps):
                    append_icp_results(all_optimized_icps)  # This is a sync function
                    progress_text.text("✅ Results saved successfully to Google Sheets!")
                else:
                    progress_text.text("✅ Processing complete. No new profiles found to save.")

                progress_bar.progress(1.0)
                st.session_state.page = 'results'
                st.rerun()
                
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
                logger.error(f"Overall processing error: {traceback.format_exc()}")
                st.button("Go back to setup", on_click=lambda: setattr(st.session_state, 'page', 'setup'))
                
        # Handle results page in the Generate tab
        elif st.session_state.page == 'results':
            st.subheader("Generation Results")
            
            # Combine all results for display
            all_profiles = []
            
            for icp_data in st.session_state.optimized_icps:
                if 'results' in icp_data:
                    normalized = normalize_results(icp_data['results'])
                    all_profiles.extend(normalized)
            
            # Convert to DataFrame
            if all_profiles:
                profiles_data = []
                for profile in all_profiles:
                    profiles_data.append({
                        "LinkedIn URL": profile.linkedin_url,
                        "Title": profile.title or "",
                        "Name": f"{profile.first_name or ''} {profile.last_name or ''}".strip(),
                        "Company": profile.company if hasattr(profile, "company") else "",
                        "Location": profile.location if hasattr(profile, "location") else "",
                        "Description": profile.description or ""
                    })
                
                results_df = pd.DataFrame(profiles_data)
            else:
                results_df = None
            
            if results_df is not None and not results_df.empty:
                total_profiles = len(results_df)
                st.markdown(f"### Found {total_profiles} matching profiles")
                
                # Add Google Sheet link
                sheet_id = os.environ.get("GOOGLE_SHEET_ID", "")
                if sheet_id:
                    sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit"
                    st.markdown(f"📊 [View all results in Google Sheets]({sheet_url})")
                    
                    # Show sheet-specific links for each ICP
                    st.markdown("### ICP-Specific Sheets")
                    st.markdown("Each ICP has its own dedicated sheet in the Google Spreadsheet:")
                    
                    for i, icp_data in enumerate(st.session_state.optimized_icps):
                        # Generate the sheet name using the same logic as in sheets.py
                        sheet_name = generate_sheet_name(icp_data["original"], i)
                        gid = i + 1  # Estimate the sheet GID (may not be perfect but often works)
                        sheet_specific_url = f"{sheet_url}#gid={gid}"
                        
                        st.markdown(f"- **{sheet_name}**: [{len(icp_data.get('results', []))} profiles]({sheet_specific_url})")
                
                # Show results data
                st.dataframe(results_df, use_container_width=True)
                
                # Download as CSV
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="linkedin_profiles.csv",
                    mime="text/csv",
                )
                
                # Add buttons to go to other tabs
                st.button("Continue to Send", on_click=lambda: st.session_state.update({"active_tab": "Send 🚀"}))
            else:
                st.warning("No profiles found matching your ICPs. Try refining your criteria.")
                st.button("Return to Setup", on_click=lambda: setattr(st.session_state, 'page', 'setup'))
    
    # SEND TAB - Send LinkedIn messages and follow-ups
    with send_tab:
        st.subheader("Send Messages and Connection Requests")
        
        if not sheet_titles:
            st.warning("No sheets available. Generate leads first in the Generate tab.")
        else:
            # Sheet selection
            selected_sheet = st.selectbox("Select sheet to use for sending", sheet_titles)
            
            if selected_sheet:
                try:
                    # Load profiles from selected sheet
                    ws = spreadsheet.worksheet(selected_sheet)
                    data = ws.get_all_records()
                    
                    # Create DataFrame
                    df = pd.DataFrame(data)
                    
                    if not df.empty:
                        # Get Master_Profiles sheet to check for Do Not Contact and current connection state
                        try:
                            master_profiles_sheet = create_or_get_master_profiles_sheet(spreadsheet)
                            master_profiles_data = master_profiles_sheet.get_all_records()
                            
                            # Create a lookup map using LinkedIn URL as key
                            master_profiles_map = {row.get('LinkedIn URL', ''): row for row in master_profiles_data if row.get('LinkedIn URL')}
                            
                            # Default values for profiles not in master sheet
                            default_master_profile = {
                                'Last System Action': 'Not contacted',
                                'Unipile Connection State': 'NOT_CONNECTED',
                                'Do Not Contact': 'FALSE'
                            }
                            
                            # Fetch Unipile data to get the most up-to-date connection states
                            with st.spinner("Fetching latest connection states from Unipile..."):
                                try:
                                    if "UNIPILE_API_KEY" in os.environ and "UNIPILE_DSN" in os.environ and "UNIPILE_ACCOUNT_ID" in os.environ:
                                        # Import the client and create it
                                        from src.unipile_client import UnipileClient
                                        client = UnipileClient()
                                        
                                        # Get async event loop
                                        try:
                                            loop = asyncio.get_event_loop()
                                        except RuntimeError:
                                            # If no event loop exists, create one
                                            loop = asyncio.new_event_loop()
                                            asyncio.set_event_loop(loop)
                                        
                                        # Fetch relations (connection states) from Unipile
                                        try:
                                            unipile_relations = loop.run_until_complete(client.list_relations())
                                            unipile_relations_map = {
                                                r.get("provider_id", ""): r.get("state", "NOT_CONNECTED") 
                                                for r in unipile_relations if r.get("provider_id")
                                            }
                                            
                                            # Fetch sent invitations
                                            unipile_invites = loop.run_until_complete(client.list_sent_invitations())
                                            unipile_invites_map = {
                                                i.get("provider_id", ""): True 
                                                for i in unipile_invites if i.get("provider_id")
                                            }
                                            
                                            # Close the client
                                            loop.run_until_complete(client.close())
                                            
                                            st.success(f"✅ Retrieved {len(unipile_relations)} connections and {len(unipile_invites)} sent invitations from Unipile.")
                                        except Exception as e:
                                            st.warning(f"❗ Could not fetch Unipile connection data. Using cached data from Master_Profiles sheet. Error: {str(e)}")
                                            unipile_relations_map = {}
                                            unipile_invites_map = {}
                                    else:
                                        st.warning("❗ Unipile credentials not found in environment variables. Using cached data from Master_Profiles sheet.")
                                        unipile_relations_map = {}
                                        unipile_invites_map = {}
                                except Exception as e:
                                    st.error(f"Error initializing Unipile client: {str(e)}")
                                    unipile_relations_map = {}
                                    unipile_invites_map = {}
                            
                            # Enrich the dataframe with master profiles data
                            enriched_rows = []
                            filtered_out_count = 0
                            do_not_contact_count = 0
                            already_connected_count = 0
                            pending_count = 0
                            
                            for _, row in df.iterrows():
                                linkedin_url = row.get('LinkedIn URL', '')
                                provider_id = row.get('Provider ID', '')
                                
                                if not linkedin_url:
                                    continue
                                
                                # Get master profile data if it exists
                                master_data = master_profiles_map.get(linkedin_url, default_master_profile)
                                
                                # Check if marked as Do Not Contact
                                if master_data.get('Do Not Contact', 'FALSE').upper() == 'TRUE':
                                    do_not_contact_count += 1
                                    filtered_out_count += 1
                                    continue
                                
                                # Check Unipile connection state (first from unipile_relations_map, then fall back to Master_Profiles)
                                # Extract profile identifier to try to match with Unipile provider_id
                                ident = linkedin_url.rstrip('/').split('/')[-1]
                                
                                # Look for this profile in unipile data
                                current_state = None
                                provider_id_found = False
                                
                                # Method 1: Direct provider_id match - most reliable 
                                if provider_id and provider_id in unipile_relations_map:
                                    current_state = unipile_relations_map.get(provider_id)
                                    provider_id_found = True
                                
                                # Method 2: Try to match by extracted identifier in provider_ids
                                if not current_state and ident:
                                    for pid, state in unipile_relations_map.items():
                                        if ident in pid:
                                            current_state = state
                                            # Since we found a match, update the row with the provider_id
                                            row['Provider ID'] = pid
                                            provider_id_found = True
                                            break
                                
                                # Check if this profile has a pending invitation
                                has_pending_invite = False
                                
                                # Method 1: Direct provider_id match for invites
                                if provider_id and provider_id in unipile_invites_map:
                                    has_pending_invite = True
                                    provider_id_found = True
                                    
                                # Method 2: Try to match invitation by identifier
                                if not has_pending_invite and ident:
                                    for pid in unipile_invites_map:
                                        if ident in pid:
                                            has_pending_invite = True
                                            # Update provider_id
                                            row['Provider ID'] = pid
                                            provider_id_found = True
                                            break
                                
                                # Method 3: Check Master_Profiles for provider_id if not already found
                                if not provider_id_found and 'Provider ID' in master_data and master_data['Provider ID']:
                                    master_provider_id = master_data['Provider ID']
                                    row['Provider ID'] = master_provider_id
                                    
                                    # Check if this provider_id is in our relations or invites maps
                                    if master_provider_id in unipile_relations_map:
                                        current_state = unipile_relations_map[master_provider_id]
                                        provider_id_found = True
                                    elif master_provider_id in unipile_invites_map:
                                        has_pending_invite = True
                                        provider_id_found = True
                                
                                # If still no current state from Unipile, use Master_Profiles
                                if not current_state:
                                    current_state = master_data.get('Unipile Connection State', 'NOT_CONNECTED')
                                
                                # Update row with most current data
                                row['Connection State'] = current_state
                                
                                # If has pending invite but not yet reflected in connection state
                                if has_pending_invite and current_state == 'NOT_CONNECTED':
                                    row['Connection State'] = 'PENDING'
                                
                                # If already connected or pending, increment counters
                                if current_state == 'CONNECTED':
                                    already_connected_count += 1
                                elif current_state == 'PENDING' or has_pending_invite:
                                    pending_count += 1
                                
                                # Update the Contact Status from master if available
                                if 'Last System Action' in master_data:
                                    action = master_data.get('Last System Action')
                                    # Only override if Last System Action has been set to something other than 'Not contacted'
                                    if action and action != 'Not contacted':
                                        row['Contact Status'] = action
                                
                                # Add the row to our enriched list
                                enriched_rows.append(row)
                            
                            # Create a new DataFrame with our enriched data
                            df = pd.DataFrame(enriched_rows)
                            
                            # Display filtering stats
                            if filtered_out_count > 0:
                                st.info(f"ℹ️ {filtered_out_count} profiles were filtered out: {do_not_contact_count} marked 'Do Not Contact', {already_connected_count} already connected, {pending_count} with pending invitations.")
                            
                        except Exception as e:
                            st.warning(f"Could not access Master_Profiles sheet: {str(e)}. Using sheet data only.")
                        
                        # Set default values for missing status fields
                        if "Contact Status" not in df.columns or df["Contact Status"].isna().all():
                            df["Contact Status"] = "Not contacted"
                        else:
                            # Fill NA values with "Not contacted"
                            df["Contact Status"] = df["Contact Status"].fillna("Not contacted")
                            
                        if "Connection State" not in df.columns or df["Connection State"].isna().all():
                            df["Connection State"] = "NOT_CONNECTED"
                        else:
                            # Fill NA values with "NOT_CONNECTED"
                            df["Connection State"] = df["Connection State"].fillna("NOT_CONNECTED")
                        
                        # Define workflow stages and valid transitions
                        workflow_stages = {
                            "Not contacted": ["Generate messages", "Invite", "Mark contacted"],
                            "Messages generated": ["Invite", "Edit messages", "Mark contacted"],
                            "Invited": ["Follow up", "Mark connected"],
                            "Invited + Commented": ["Follow up", "Mark connected"],
                            "Message sent": ["Follow up", "Mark responded"],
                            "Follow-up sent": ["Follow up again", "Mark responded"],
                            "Follow-up scheduled": ["Check status", "Mark responded"],
                        }
                        
                        # Map Connection State to recommended actions
                        connection_actions = {
                            "NOT_CONNECTED": ["Invite", "Generate messages"],
                            "INVITED": ["Wait for acceptance", "Follow up"],
                            "PENDING": ["Wait for acceptance", "Follow up"],
                            "CONNECTED": ["Send message", "Follow up"]
                        }
                        
                        # Filter for profiles to message
                        st.subheader("Filter Profiles")
                        
                        # Add filters
                        filter_col1, filter_col2, filter_col3 = st.columns(3)
                        
                        with filter_col1:
                            connection_filter = st.multiselect(
                                "Connection State",
                                options=["", "NOT_CONNECTED", "INVITED", "PENDING", "CONNECTED"],
                                default=["NOT_CONNECTED", "INVITED", "PENDING"],
                                help="Filter by connection status"
                            )
                        
                        with filter_col2:
                            contact_status_filter = st.multiselect(
                                "Contact Status",
                                options=["", "Not contacted", "Messages generated", "Invited", "Invited + Commented", "Message sent", "Follow-up sent", "Follow-up scheduled"],
                                default=["Not contacted"],
                                help="Filter by previous contact status"
                            )
                            
                        with filter_col3:
                            recommended_action = st.selectbox(
                                "Recommended Action",
                                options=["All actions", "Need to generate messages", "Need to invite", "Need to follow up", "Wait for acceptance", "Connected profiles", "Mark responded"],
                                index=0,
                                help="Filter by recommended next action"
                            )
                        
                        # Apply filters
                        filtered_df = df
                        
                        # Apply Connection State filter
                        if connection_filter:
                            filtered_df = filtered_df[filtered_df["Connection State"].isin(connection_filter) | 
                                                    (filtered_df["Connection State"].isnull() & ("" in connection_filter))]
                        
                        # Apply Contact Status filter
                        if contact_status_filter:
                            filtered_df = filtered_df[filtered_df["Contact Status"].isin(contact_status_filter) | 
                                                    (filtered_df["Contact Status"].isnull() & ("" in contact_status_filter))]
                        
                        # Apply Recommended Action filter
                        if recommended_action != "All actions":
                            if recommended_action == "Need to generate messages":
                                filtered_df = filtered_df[filtered_df["Contact Status"] == "Not contacted"]
                            elif recommended_action == "Need to invite":
                                filtered_df = filtered_df[(filtered_df["Connection State"] == "NOT_CONNECTED") & 
                                                         (filtered_df["Contact Status"].isin(["Not contacted", "Messages generated"]))]
                            elif recommended_action == "Need to follow up":
                                filtered_df = filtered_df[(filtered_df["Connection State"].isin(["INVITED", "PENDING", "CONNECTED"])) & 
                                                         (filtered_df["Contact Status"].isin(["Invited", "Message sent", "Follow-up sent"]))]
                            elif recommended_action == "Wait for acceptance":
                                filtered_df = filtered_df[filtered_df["Connection State"] == "PENDING"]
                            elif recommended_action == "Connected profiles":
                                filtered_df = filtered_df[filtered_df["Connection State"] == "CONNECTED"]
                            elif recommended_action == "Mark responded":
                                filtered_df = filtered_df[(filtered_df["Connection State"] == "CONNECTED") &
                                                        (filtered_df["Contact Status"].isin(["Message sent", "Follow-up sent", "Follow-up scheduled"]))]
                                
                        # Add recommended next action based on current statuses
                        filtered_df["Recommended Action"] = filtered_df.apply(
                            lambda row: _determine_next_action(row["Connection State"], row["Contact Status"], connection_actions, workflow_stages),
                            axis=1
                        )
                        
                        # Display filtered profiles with editable fields
                        st.subheader(f"Profiles to Process ({len(filtered_df)} matching)")
                        
                        # Ensure all expected columns are present before showing in editor
                        expected_columns = [
                            "LinkedIn URL", "First Name", "Last Name", "Title", "Company", "Location", 
                            "Connection Msg", "Comment Msg", "FU-1", "FU-2", "FU-3",
                            "Provider ID", "Connection State", "Contact Status", "Recommended Action"
                        ]
                        
                        for col in expected_columns:
                            if col not in filtered_df.columns:
                                filtered_df[col] = ""  # Add missing columns with empty strings
                        
                        # Debug information (show columns that will be displayed)
                        logger.debug(f"DataFrame columns for data_editor: {filtered_df.columns.tolist()}")
                        
                        # Show editable data grid
                        edited_df = st.data_editor(
                            filtered_df,
                            column_config={
                                "LinkedIn URL": st.column_config.LinkColumn("LinkedIn URL", disabled=True, display_text="Open Profile"),
                                "First Name": st.column_config.TextColumn("First Name", disabled=True),
                                "Last Name": st.column_config.TextColumn("Last Name", disabled=True),
                                "Title": st.column_config.TextColumn("Title", disabled=True),
                                "Company": st.column_config.TextColumn("Company", disabled=True),
                                "Location": st.column_config.TextColumn("Location", disabled=True),
                                "Connection Msg": st.column_config.TextColumn("Connection Msg", width="large"),
                                "Comment Msg": st.column_config.TextColumn("Comment Msg", width="large"),
                                "FU-1": st.column_config.TextColumn("Follow-up 1", width="large"),
                                "FU-2": st.column_config.TextColumn("Follow-up 2", width="large"),
                                "FU-3": st.column_config.TextColumn("Follow-up 3", width="large"),
                                "Provider ID": st.column_config.TextColumn("Provider ID", disabled=True),
                                "Connection State": st.column_config.TextColumn("Connection State", disabled=True),
                                "Contact Status": st.column_config.TextColumn("Contact Status", disabled=True),
                                "Recommended Action": st.column_config.TextColumn("Recommended Action", disabled=True),
                            },
                            hide_index=True,
                            use_container_width=True,
                            num_rows="dynamic"
                        )
                        
                        # Select which rows to process
                        selected_indices = st.multiselect(
                            "Select profiles to process", 
                            options=list(range(len(edited_df))),
                            format_func=lambda i: f"{edited_df.iloc[i]['First Name']} {edited_df.iloc[i]['Last Name']} - {edited_df.iloc[i]['Title']} ({edited_df.iloc[i]['Recommended Action']})"
                        )
                        
                        # Message generation options
                        st.subheader("Message Options")
                        
                        reuse_messages = st.checkbox("Re-use existing messages if present", value=True,
                                                    help="If checked, will use existing messages in the sheet. If unchecked, will re-generate all messages.")
                        
                        # Determine available campaign modes based on selected profiles
                        available_modes = ["Generate only"]
                        
                        if len(selected_indices) > 0:
                            selected_profiles = edited_df.iloc[selected_indices]
                            
                            # Check if any profiles need invites
                            if any(selected_profiles["Connection State"] == "NOT_CONNECTED"):
                                available_modes.append("Invite only")
                                available_modes.append("Invite + Comment")
                            
                            # Check if any profiles are already connected or pending
                            if any(selected_profiles["Connection State"].isin(["CONNECTED", "PENDING", "INVITED"])):
                                available_modes.append("Message only")
                                
                            # Always add the full option
                            available_modes.append("Full (invite, comment, follow-ups)")
                        else:
                            # If no profiles selected, show all options
                            available_modes = ["Generate only", "Invite only", "Invite + Comment", "Message only", "Full (invite, comment, follow-ups)"]
                        
                        # Campaign mode selection
                        mode = st.radio(
                            "Campaign Mode", 
                            available_modes,
                            help="Generate: just create messages, Invite: send connection requests, Full: all actions"
                        )
                        
                        # Follow-up configuration
                        with st.expander("Follow-up Timing", expanded=False):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                follow1 = st.number_input("Days until follow-up 1", 1, 30, 3)
                            with col2:
                                follow2 = st.number_input("Days until follow-up 2", 1, 60, 7)
                            with col3:
                                follow3 = st.number_input("Days until follow-up 3", 1, 90, 14)
                        
                        # Schedule option - only show if we're not just generating messages
                        if mode != "Generate only":
                            send_later = st.checkbox("Schedule for later", value=False, 
                                                    help="Schedule messages to be sent at a specific time")
                            
                            if send_later:
                                send_date = st.date_input("Send date")
                                send_time = st.time_input("Send time")
                        else:
                            send_later = False
                        
                        # Launch button
                        max_sends = len(selected_indices) if selected_indices else len(edited_df)
                        # Ensure max_sends is at least 1 to avoid the min_value error
                        max_sends = max(1, max_sends)
                        # Calculate a safe initial value (min of 10 and max_sends)
                        initial_value = min(10, max_sends)
                        limit = st.number_input("Maximum profiles to process in this run", 1, max_sends, initial_value)
                        
                        # Dynamic button text based on mode
                        button_text = "Generate Messages" if mode == "Generate only" else "🚀 Send Messages"
                        
                        # When the launch button is clicked, generate personalized messages and run campaign
                        if st.button(f"Launch {button_text} Campaign", use_container_width=True):
                            # Define the mapping from sheet columns to Profile model attributes
                            sheet_to_profile_map = {
                                "LinkedIn URL": "linkedin_url",
                                "Title": "title",
                                "First Name": "first_name",
                                "Last Name": "last_name",
                                "Company": "company",
                                "Location": "location",
                                "Description": "description",
                                "Profile Image URL": "profile_image_url",
                                "Connection Msg": "connection_msg",
                                "Comment Msg": "comment_msg",
                                "FU-1": "followup1",
                                "FU-2": "followup2",
                                "FU-3": "followup3",
                                "InMail": "inmail",
                                "Contact Status": "contact_status",
                                "Connection State": "connection_state",
                                "Follower Cnt": "followers_count",
                                # Add other fields if they exist in Profile model and sheet
                            }

                            targets = []
                            processed_rows_df = edited_df.iloc[selected_indices] if selected_indices else edited_df
                            
                            for _, row_series in processed_rows_df.iterrows():
                                row = row_series.to_dict() # Convert Series to Dict
                                profile_dict_for_model = {}
                                
                                for sheet_col, model_attr in sheet_to_profile_map.items():
                                    if sheet_col in row:
                                        profile_dict_for_model[model_attr] = row[sheet_col]
                                
                                # Ensure required fields like linkedin_url are present and not empty
                                if not profile_dict_for_model.get("linkedin_url"):
                                    st.warning(f"Skipping row, missing or empty LinkedIn URL: {row.get('First Name', '')} {row.get('Last Name', '')}")
                                    continue

                                # Handle potential type issues, e.g., followers_count should be int
                                if "followers_count" in profile_dict_for_model:
                                    fc_val = profile_dict_for_model["followers_count"]
                                    if fc_val == "" or fc_val is None:
                                        profile_dict_for_model["followers_count"] = 0
                                    else:
                                        try:
                                            profile_dict_for_model["followers_count"] = int(fc_val)
                                        except ValueError:
                                            logger.warning(f"Could not convert followers_count '{fc_val}' to int for {profile_dict_for_model.get('linkedin_url')}. Defaulting to 0.")
                                            profile_dict_for_model["followers_count"] = 0
                                
                                try:
                                    targets.append(Profile(**profile_dict_for_model))
                                except Exception as e:
                                    st.error(f"Error creating Profile object from row {profile_dict_for_model.get('linkedin_url', 'N/A')}: {str(e)}.")
                                    logger.error(f"Data causing Profile creation error: {profile_dict_for_model}")
                            
                            targets = targets[:limit] # Apply the limit

                            # Don't regenerate messages if "Generate only" mode is not selected and messages exist
                            should_generate_messages = True
                            if mode != "Generate only":
                                # Check if most targets already have messages
                                messages_exist = all(
                                    [p.connection_msg and len(p.connection_msg.strip()) > 0 for p in targets[:min(3, len(targets))]]
                                )
                                if messages_exist:
                                    should_generate_messages = False
                                    st.info("Using existing messages - skipping regeneration")
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Follow-up configuration
                            follow_days = [follow1]
                            if follow2 > 0:
                                follow_days.append(follow2)
                            if follow3 > 0:
                                follow_days.append(follow3)
                            
                            # Schedule settings
                            schedule_time = None
                            if send_later:
                                schedule_dt = datetime.combine(send_date, send_time)
                                schedule_time = schedule_dt.isoformat()
                                
                            stats = asyncio.run(run_campaign(
                                profiles=targets,
                                followup_days=follow_days,
                                mode=mode,
                                spreadsheet_id=sheet_id,
                                sheet_name=selected_sheet,
                                schedule_time=schedule_time,
                                should_generate_messages=should_generate_messages
                            ))
                            
                            # Show results
                            st.success(f"Campaign completed! Generated: {stats.generated}, Sent: {stats.sent}, Errors: {stats.errors}, Skipped: {stats.skipped}")
                            
                            # Set a flag to indicate that data potentially changed (for Sync tab)
                            if mode != "Generate only" and (stats.sent > 0 or stats.generated > 0):
                                st.session_state.data_potentially_changed = True
                                st.session_state.last_synced_sheet = selected_sheet

                            # If there were errors, show a detailed error section
                            if stats.errors > 0:
                                with st.expander(f"⚠️ {stats.errors} Error(s) Occurred", expanded=True):
                                    st.warning("Check the Error Msg column in the sheet for details on each profile")
                                    st.markdown("""
                                    **Common campaign errors:**
                                    - '0' means the profile couldn't be processed (often due to API limitations)
                                    - Missing permissions or rate limits from LinkedIn
                                    - Network connectivity issues
                                    
                                    Try again with fewer profiles or try later when API limits reset.
                                    """)
                            
                            # Update status of processed profiles
                            status_updates = {}
                            
                            if mode == "Generate only":
                                status_updates["Contact Status"] = "Messages generated"
                            elif mode == "Invite only":
                                status_updates["Contact Status"] = "Invited"
                            elif mode == "Invite + Comment":
                                status_updates["Contact Status"] = "Invited + Commented"
                            elif mode == "Message only":
                                status_updates["Contact Status"] = "Message sent"
                            elif mode == "Full (invite, comment, follow-ups)":
                                status_updates["Contact Status"] = "Follow-up scheduled"
                            
                            # Update status in worksheet if we processed any profiles
                            if status_updates and (stats.generated > 0 or stats.sent > 0):
                                try:
                                    # Get all records to find the row numbers
                                    all_records = ws.get_all_records()
                                    
                                    # Update rows for each target
                                    for profile in targets:
                                        # Find matching row
                                        for i, record in enumerate(all_records):
                                            if record.get("LinkedIn URL") == profile.linkedin_url:
                                                # Row is 1-indexed and we need to skip header row
                                                row_num = i + 2
                                                
                                                # Update status fields
                                                for field, value in status_updates.items():
                                                    col_idx = list(all_records[0].keys()).index(field) + 1
                                                    ws.update_cell(row_num, col_idx, value)
                                        
                                    st.info(f"Updated status for {len(targets)} profiles")
                                except Exception as e:
                                    st.warning(f"Could not update status in worksheet: {str(e)}")
                            
                            # Add sync button
                            if st.button("Sync Status Data"):
                                with st.spinner("Syncing status data..."):
                                    sync_results = asyncio.run(sync_status(sheet_id, selected_sheet))
                                    st.success(f"Sync completed! Updated {sync_results['updated']} records")
                    else:
                        st.warning("No profiles found in the selected sheet.")
                except Exception as e:
                    st.error(f"Error loading sheet: {str(e)}")
    
    # SYNC & MONITOR TAB - Track relationship status and conversation data
    with sync_tab:
        st.subheader("Sync & Monitor Relationships")
        
        if not sheet_titles:
            st.warning("No sheets available. Generate leads first in the Generate tab.")
        else:
            # Add tabs in the Sync section for different functions
            monitor_tab, master_profiles_tab = st.tabs(["Monitor Relationships", "Master Profiles"])
            
            with monitor_tab:
                # Initialize session state variables for sync tab if they don't exist
                if 'data_potentially_changed' not in st.session_state:
                    st.session_state.data_potentially_changed = False
                if 'last_synced_sheet' not in st.session_state:
                    st.session_state.last_synced_sheet = None
                    
                # Sheet selection
                selected_sheet = st.selectbox("Select sheet to monitor", sheet_titles)
                
                if selected_sheet:
                    try:
                        # Auto-sync if data changed for the current sheet
                        if st.session_state.data_potentially_changed and st.session_state.last_synced_sheet == selected_sheet:
                            with st.spinner("Automatically syncing after recent campaign..."):
                                logger.info(f"Auto-syncing sheet '{selected_sheet}' due to recent campaign activity.")
                                sync_results = asyncio.run(sync_status(sheet_id, selected_sheet))
                                st.success(f"Auto-sync completed! Updated {sync_results['updated']} records")
                                # Reset the flag
                                st.session_state.data_potentially_changed = False
                                st.session_state.last_synced_sheet = None
                        
                        # Load profiles from selected sheet
                        ws = spreadsheet.worksheet(selected_sheet)
                        data = ws.get_all_records()
                        
                        # Create DataFrame
                        df = pd.DataFrame(data)
                        
                        if not df.empty:
                            # Show metrics at the top
                            stats = get_relationship_stats(df)
                            
                            # Create metrics row
                            metric_cols = st.columns(5)
                            with metric_cols[0]:
                                st.metric("Total Profiles", stats["total"])
                            with metric_cols[1]:
                                st.metric("Invites Sent", stats["sent"])
                            with metric_cols[2]:
                                st.metric("Connected", stats["connected"])
                            with metric_cols[3]:
                                st.metric("Comments", stats["comments"])
                            with metric_cols[4]:
                                st.metric("Unread Messages", stats["unread"])
                            
                            # Sync button
                            if st.button("🔄 Sync Now", type="primary"):
                                with st.spinner("Syncing status data from Unipile..."):
                                    sync_results = asyncio.run(sync_status(sheet_id, selected_sheet))
                                    st.success(f"Sync completed! Processed: {sync_results['processed']}, Updated: {sync_results['updated']}, Errors: {sync_results['errors']}")
                                    
                                    # Reload data
                                    data = ws.get_all_records()
                                    df = pd.DataFrame(data)
                                    
                                    # Update metrics
                                    stats = get_relationship_stats(df)
                                    st.rerun()
                        
                            # Add filters
                            st.subheader("Filter Profiles")
                            filter_cols = st.columns(3)
                            
                            with filter_cols[0]:
                                connection_filter = st.multiselect(
                                    "Connection State",
                                    options=["", "NOT_CONNECTED", "INVITED", "PENDING", "CONNECTED"],
                                    default=[],
                                    help="Filter by connection status"
                                )
                            
                            with filter_cols[1]:
                                unread_filter = st.checkbox("Show only unread messages", value=False,
                                                        help="Show only profiles with unread messages")
                                
                            with filter_cols[2]:
                                sort_by = st.selectbox(
                                    "Sort by",
                                    options=["Last Action UTC", "Connection State", "First Name"],
                                    index=0,
                                    help="Sort profiles by this field"
                                )
                            
                            # Apply filters
                            filtered_df = df.copy()
                            
                            if connection_filter:
                                filtered_df = filtered_df[filtered_df["Connection State"].isin(connection_filter)]
                                
                            if unread_filter:
                                try:
                                    filtered_df = filtered_df[filtered_df["Unread Cnt"] > 0]
                                except (KeyError, TypeError):
                                    st.warning("Unread count column not found or invalid. Sync first to get this data.")
                            
                            # Apply sorting
                            if sort_by in filtered_df.columns:
                                try:
                                    # For dates, handle potential formatting issues
                                    if sort_by == "Last Action UTC":
                                        filtered_df = filtered_df.sort_values(by=sort_by, ascending=False, na_position='last')
                                    else:
                                        filtered_df = filtered_df.sort_values(by=sort_by)
                                except Exception as e:
                                    st.warning(f"Could not sort by {sort_by}: {str(e)}")
                            
                            # Display filtered data
                            st.subheader(f"Profile Status ({len(filtered_df)} profiles)")
                            
                            # Select columns to show
                            display_cols = ["First Name", "Last Name", "Title", "Connection State", "Contact Status"]
                            # Add additional columns if they exist
                            if "Last Action UTC" in filtered_df.columns:
                                display_cols.append("Last Action UTC")
                            if "Unread Cnt" in filtered_df.columns:
                                display_cols.append("Unread Cnt")
                            if "Last Msg UTC" in filtered_df.columns:
                                display_cols.append("Last Msg UTC")
                            
                            # Ensure all selected columns exist (and don't remove Connection State or Contact Status - always add them if missing)
                            for col in ["Connection State", "Contact Status"]:
                                if col not in filtered_df.columns:
                                    filtered_df[col] = ""
                            
                            # Ensure Contact Status is populated with 'Not contacted' where empty
                            if "Contact Status" in filtered_df.columns:
                                filtered_df["Contact Status"] = filtered_df["Contact Status"].fillna("Not contacted")
                                # Replace empty strings with 'Not contacted'
                                filtered_df.loc[filtered_df["Contact Status"] == "", "Contact Status"] = "Not contacted"
                            
                            # Filter display columns to those that exist in the dataframe
                            display_cols = [col for col in display_cols if col in filtered_df.columns]
                            
                            # Display the data
                            st.dataframe(filtered_df[display_cols], use_container_width=True)
                            
                            # Export option
                            if st.button("Export to CSV"):
                                csv = filtered_df.to_csv(index=False)
                                st.download_button(
                                    label="Download CSV",
                                    data=csv,
                                    file_name=f"linkedin_profiles_{selected_sheet}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                )
                        else:
                            st.warning("No profiles found in the selected sheet.")
                    except Exception as e:
                        st.error(f"Error loading sheet: {str(e)}")
            
            with master_profiles_tab:
                st.subheader("Master Profiles Management")
                
                try:
                    # Get the Master_Profiles sheet
                    master_sheet = create_or_get_master_profiles_sheet(spreadsheet)
                    master_data = master_sheet.get_all_records()
                    
                    if not master_data:
                        st.warning("No profiles found in Master_Profiles sheet.")
                    else:
                        # Create a DataFrame
                        master_df = pd.DataFrame(master_data)
                        
                        # Show some metrics
                        stat_cols = st.columns(4)
                        with stat_cols[0]:
                            st.metric("Total Unique Profiles", len(master_df))
                        with stat_cols[1]:
                            do_not_contact_count = sum(1 for dnc in master_df.get("Do Not Contact", []) if str(dnc).upper() == "TRUE")
                            st.metric("Do Not Contact", do_not_contact_count)
                        with stat_cols[2]:
                            connected_count = sum(1 for state in master_df.get("Unipile Connection State", []) if state == "CONNECTED")
                            st.metric("Connected", connected_count)
                        with stat_cols[3]:
                            pending_count = sum(1 for state in master_df.get("Unipile Connection State", []) if state in ["PENDING", "INVITED"])
                            st.metric("Pending", pending_count)
                        
                        # Add filters for the Master Profiles view
                        st.subheader("Filter Master Profiles")
                        filter_cols = st.columns(3)
                        
                        with filter_cols[0]:
                            dnc_filter = st.radio(
                                "Do Not Contact",
                                options=["All", "Only Do Not Contact", "Exclude Do Not Contact"],
                                index=0,
                                help="Filter by Do Not Contact status"
                            )
                        
                        with filter_cols[1]:
                            connection_state_filter = st.multiselect(
                                "Connection State",
                                options=["", "NOT_CONNECTED", "INVITED", "PENDING", "CONNECTED"],
                                default=[],
                                help="Filter by connection status"
                            )
                            
                        with filter_cols[2]:
                            source_icp_filter = st.text_input(
                                "Source ICP (contains)",
                                value="",
                                help="Filter by ICP name (partial match)"
                            )
                        
                        # Apply filters
                        filtered_master_df = master_df.copy()
                        
                        # Apply Do Not Contact filter
                        if dnc_filter == "Only Do Not Contact":
                            filtered_master_df = filtered_master_df[filtered_master_df["Do Not Contact"].str.upper() == "TRUE"]
                        elif dnc_filter == "Exclude Do Not Contact":
                            filtered_master_df = filtered_master_df[filtered_master_df["Do Not Contact"].str.upper() != "TRUE"]
                            
                        # Apply Connection State filter
                        if connection_state_filter:
                            filtered_master_df = filtered_master_df[filtered_master_df["Unipile Connection State"].isin(connection_state_filter)]
                            
                        # Apply Source ICP filter
                        if source_icp_filter:
                            filtered_master_df = filtered_master_df[filtered_master_df["Source ICPs"].str.contains(source_icp_filter, case=False, na=False)]
                        
                        # Display the filtered data
                        st.subheader(f"Master Profiles ({len(filtered_master_df)} profiles)")
                        
                        # Define display columns
                        display_cols = [
                            "LinkedIn URL", "First Name", "Last Name", "Title", "Company", 
                            "Last System Action", "Unipile Connection State", "Source ICPs", "Do Not Contact"
                        ]
                        
                        # Ensure all display columns exist
                        display_cols = [col for col in display_cols if col in filtered_master_df.columns]
                        
                        # Display the data with the ability to edit Do Not Contact column
                        edited_master_df = st.data_editor(
                            filtered_master_df[display_cols],
                            column_config={
                                "LinkedIn URL": st.column_config.TextColumn("LinkedIn URL", disabled=True),
                                "First Name": st.column_config.TextColumn("First Name", disabled=True),
                                "Last Name": st.column_config.TextColumn("Last Name", disabled=True),
                                "Title": st.column_config.TextColumn("Title", disabled=True),
                                "Company": st.column_config.TextColumn("Company", disabled=True),
                                "Last System Action": st.column_config.TextColumn("Last System Action", disabled=True),
                                "Unipile Connection State": st.column_config.TextColumn("Unipile Connection State", disabled=True),
                                "Source ICPs": st.column_config.TextColumn("Source ICPs", disabled=True),
                                "Do Not Contact": st.column_config.CheckboxColumn("Do Not Contact", help="Mark profiles to never contact")
                            },
                            hide_index=True,
                            use_container_width=True,
                            num_rows="dynamic"
                        )
                        
                        # Save changes button
                        if st.button("Save Changes to Do Not Contact List"):
                            try:
                                # Get the original full master data to compare
                                changes_made = False
                                
                                # Find rows that have been changed
                                for i, edited_row in edited_master_df.iterrows():
                                    linkedin_url = edited_row["LinkedIn URL"]
                                    new_dnc_value = "TRUE" if edited_row["Do Not Contact"] else "FALSE"
                                    
                                    # Find this URL in the original data
                                    original_rows = master_df[master_df["LinkedIn URL"] == linkedin_url]
                                    
                                    if not original_rows.empty:
                                        original_idx = original_rows.index[0]
                                        original_dnc = master_df.loc[original_idx, "Do Not Contact"]
                                        
                                        # Check if value has changed
                                        original_dnc_bool = original_dnc.upper() == "TRUE" if isinstance(original_dnc, str) else bool(original_dnc)
                                        new_dnc_bool = new_dnc_value.upper() == "TRUE"
                                        
                                        if original_dnc_bool != new_dnc_bool:
                                            # Find the row in the master sheet (2-based index for sheets)
                                            cell = master_sheet.find(linkedin_url)
                                            if cell:
                                                row_idx = cell.row
                                                # Update the Do Not Contact column (column O in Master_Profiles)
                                                master_sheet.update_cell(row_idx, 15, new_dnc_value)
                                                changes_made = True
                                
                                if changes_made:
                                    st.success("✅ Do Not Contact list updated successfully!")
                                else:
                                    st.info("No changes detected to save.")
                                    
                            except Exception as e:
                                st.error(f"Error saving changes: {str(e)}")
                                
                        # Bulk add to Do Not Contact
                        with st.expander("Bulk Add to Do Not Contact"):
                            st.info("Paste a list of LinkedIn URLs to mark as Do Not Contact")
                            bulk_dnc_urls = st.text_area(
                                "LinkedIn URLs (one per line)",
                                height=150,
                                help="Paste LinkedIn profile URLs, one per line"
                            )
                            
                            if st.button("Add All to Do Not Contact"):
                                if not bulk_dnc_urls.strip():
                                    st.warning("No URLs provided.")
                                else:
                                    urls = [url.strip() for url in bulk_dnc_urls.split("\n") if url.strip()]
                                    
                                    updated_count = 0
                                    not_found_urls = []
                                    
                                    for url in urls:
                                        try:
                                            # Find the URL in the master sheet
                                            cell = master_sheet.find(url)
                                            if cell:
                                                row_idx = cell.row
                                                # Update the Do Not Contact column (column O)
                                                master_sheet.update_cell(row_idx, 15, "TRUE")
                                                updated_count += 1
                                            else:
                                                not_found_urls.append(url)
                                        except Exception as e:
                                            st.error(f"Error updating {url}: {str(e)}")
                                            
                                    if updated_count > 0:
                                        st.success(f"✅ Added {updated_count} URLs to Do Not Contact list")
                                    
                                    if not_found_urls:
                                        st.warning(f"❗ {len(not_found_urls)} URLs were not found in the Master_Profiles sheet")
                                        with st.expander(f"View {len(not_found_urls)} not found URLs"):
                                            for url in not_found_urls:
                                                st.text(url)
                                                
                except Exception as e:
                    st.error(f"Error accessing Master_Profiles sheet: {str(e)}")
                    st.info("If this is your first time using the tool, run a campaign in the Generate tab to create the Master_Profiles sheet.")


if __name__ == "__main__":
    main() 