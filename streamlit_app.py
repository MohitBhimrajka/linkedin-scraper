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


def optimize_query_with_gemini(query_text):
    """Use Gemini API to optimize a search query."""
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    
    if not gemini_api_key:
        st.error("❌ GEMINI_API_KEY not found in environment variables")
        return query_text
    
    client = genai.Client(api_key=gemini_api_key)
    model = "gemini-2.5-flash-preview-04-17"
    
    prompt = f"""
    You are a world-class expert in LinkedIn search optimization and Google dorking techniques, specializing in crafting highly precise search queries to find the exact LinkedIn profiles matching specific Ideal Customer Profile (ICP) criteria.

    Your task is to transform the given "Original ICP criteria" into a HIGHLY TARGETED, OPTIMIZED Google search string that will produce extremely accurate results when used with a Google Custom Search Engine (CSE) configured to search within LinkedIn.

    **Advanced Optimization Guidelines:**

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

    **Output Requirements:**
    * Return ONLY the optimized search query as a single, continuous string.
    * Do NOT include explanations, labels, or any text before/after the search string.
    * Do NOT include site:linkedin.com/in/ as the CSE is already configured to search only within LinkedIn.

    **Examples of Excellent Optimized Queries:**
    
    ORIGINAL: "Product managers at enterprise SaaS companies in Boston"
    OPTIMIZED: ("product manager" OR "product management" OR "product owner" OR "senior product manager") ("enterprise software" OR "B2B SaaS" OR SaaS OR "software as a service") (Boston OR "greater boston area" OR massachusetts OR MA) -intern -junior -associate
    
    ORIGINAL: "Finance executives at Series B startups in Europe"
    OPTIMIZED: ("CFO" OR "Chief Financial Officer" OR "VP Finance" OR "Head of Finance" OR "Finance Director") ("Series B" OR "Series C" OR "growth stage" OR "venture backed" OR "venture funded") ("Europe" OR "European Union" OR EU OR London OR Paris OR Berlin OR Amsterdam OR Madrid OR "Nordic" OR DACH)

    Original ICP criteria:
    {query_text}

    Optimized Google Search String for LinkedIn:
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
    
    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        return response.text.strip()
    except Exception as e:
        st.error(f"❌ Error optimizing ICP criteria with Gemini: {str(e)}")
        return query_text


def save_queries_to_yaml(queries):
    """Save the list of ICP queries to the YAML file."""
    query_data = {"queries": [{"query": q} for q in queries]}
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
    
    # Extract optimized queries
    queries = [icp["optimized"] for icp in optimized_icps]
    logger.info(f"Starting LinkedIn profile collection with {len(queries)} ICPs (limit: {limit})")
    
    # Create Batcher for rate limiting and parallelism
    batcher = Batcher(max_in_flight=10, delay=2)
    
    # Process each ICP criteria
    all_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, query in enumerate(queries):
        original_icp = optimized_icps[i]["original"]
        optimized_icp = query
        
        status_text.text(f"Processing ICP {i+1}/{len(queries)}: {original_icp[:50]}...")
        query_results = await process_query(query, limit, batcher)
        
        # Save results by ICP
        result_count = len(query_results)
        optimized_icps[i]["result_count"] = result_count
        optimized_icps[i]["results"] = query_results
        
        all_results.extend(query_results)
        logger.info(f"Completed ICP search with {result_count} results")
        progress_bar.progress((i + 1) / len(queries))
    
    # Normalize and deduplicate results
    unique_profiles = normalize_results(all_results)
    
    # Append to Google Sheets - both to main sheet and individual ICP sheets
    if unique_profiles:
        logger.info(f"Appending {len(unique_profiles)} profiles to Google Sheets")
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
        
        # Get all worksheets
        worksheets = spreadsheet.worksheets()
        
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
            
            if sheet_title == "Master_Sheet":
                # Clear all data in Master_Sheet except headers
                try:
                    # Get all values to determine the size
                    all_values = worksheet.get_all_values()
                    if len(all_values) > 1:  # If there are rows beyond the header
                        # Clear everything after row 1
                        worksheet.batch_clear([f"A2:Z{len(all_values)}"])
                        if show_progress:
                            progress_text.text(f"Cleared all runs from Master_Sheet")
                except Exception as e:
                    if show_progress:
                        st.warning(f"Could not clear Master_Sheet: {str(e)}")
                    logger.warning(f"Failed to clear Master_Sheet: {str(e)}")
            elif sheet_title != "Sheet1":  # Skip the default Sheet1
                # Delete all other sheets
                try:
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
    if pd.isna(connection_state) or connection_state == "":
        connection_state = "NOT_CONNECTED"
    
    if pd.isna(contact_status) or contact_status == "":
        contact_status = "Not contacted"
    
    # First check workflow stage
    if contact_status in workflow_stages:
        return workflow_stages[contact_status][0]  # Return first recommended action
    
    # Then check connection state
    if connection_state in connection_actions:
        return connection_actions[connection_state][0]  # Return first recommended action
    
    # Default action if we can't determine
    return "Generate messages"


# Create an async function to process all ICPs concurrently
async def _process_all_icps_concurrently(queries_to_process, limit_per_icp, batcher_instance, st_progress_bar, st_progress_text):
    all_optimized_icps_data = []
    optimization_step_weight = 0.2  # 20% of progress for optimization
    processing_step_weight = 0.8    # 80% of progress for fetching

    num_queries = len(queries_to_process)

    # Step 1: Optimize all queries (synchronous, but can be batched if Gemini API were async)
    optimized_query_details = []
    for i, original_query_text in enumerate(queries_to_process):
        st_progress_text.text(f"Optimizing ICP #{i+1}/{num_queries}...")
        optimized_text = optimize_query_with_gemini(original_query_text)  # This is a sync call
        optimized_query_details.append({
            "original": original_query_text,
            "optimized": optimized_text,
            "results": [],  # Placeholder
            "result_count": 0  # Placeholder
        })
        st_progress_bar.progress(((i + 1) / num_queries) * optimization_step_weight)
    
    # Step 2: Create coroutines for fetching profiles for each optimized query
    fetch_tasks = []
    for i, detail in enumerate(optimized_query_details):
        st_progress_text.text(f"Preparing to fetch for ICP #{i+1}/{num_queries}: {detail['original'][:50]}...")
        # process_query is an async function
        fetch_tasks.append(process_query(detail['optimized'], limit=limit_per_icp, batcher=batcher_instance))

    # Step 3: Run all fetch_profile tasks concurrently
    all_query_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    # Step 4: Collate results
    current_progress = optimization_step_weight
    for i, result_or_exception in enumerate(all_query_results):
        detail = optimized_query_details[i]
        if isinstance(result_or_exception, Exception):
            st.error(f"Error processing optimized ICP '{detail['optimized'][:50]}...': {str(result_or_exception)}")
            logger.error(f"Exception during process_query for '{detail['optimized']}': {result_or_exception}")
            detail["results"] = []
            detail["result_count"] = 0
        else:
            detail["results"] = result_or_exception
            detail["result_count"] = len(result_or_exception)
        
        all_optimized_icps_data.append(detail)
        current_progress += (1/num_queries) * processing_step_weight
        st_progress_bar.progress(min(current_progress, 1.0))  # Cap progress at 1.0
        st_progress_text.text(f"Fetched data for ICP #{i+1}/{num_queries}. Found {detail['result_count']} profiles.")
        
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
                        help="Describe the professionals you want to find. The AI will optimize your query.",
                        placeholder="""Example:
                        Product executives at SaaS companies in growth stage (Series B or C) located in San Francisco, New York, or Austin
                        
                        OR
                        
                        Operations leaders (COO, VP Ops) at e-commerce or retail tech companies in Europe (London, Berlin, Amsterdam)"""
                    )
                    
                    # Buttons for adding and editing
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if st.button("Add ICP", use_container_width=True):
                            if new_icp.strip():
                                # Add the ICP to the list
                                st.session_state.queries.append(new_icp.strip())
                                save_queries_to_yaml(st.session_state.queries)
                                
                                # Set the clear input flag instead of directly modifying the widget value
                                st.session_state.clear_input = True
                                
                                # Set a flag to show a notification
                                st.session_state.show_success = True
                                st.session_state.success_message = f"ICP #{len(st.session_state.queries)} added successfully!"
                                
                                # Rerun to refresh the UI
                                st.rerun()
                            else:
                                st.error("❌ ICP criteria cannot be empty")
                    
                    # Show success message if flag is set
                    if st.session_state.get('show_success', False):
                        # Create a custom success notification
                        st.markdown(
                            f"""
                            <div style="
                                padding: 10px; 
                                border-radius: 5px; 
                                margin-bottom: 15px;
                                background-color: #d1e7dd;
                                border-left: 5px solid #0a3622;
                                animation: fadeIn 0.5s;
                            ">
                                <div style="display: flex; align-items: center;">
                                    <div style="font-size: 20px; margin-right: 10px;">✅</div>
                                    <div style="font-weight: 500; color: #0a3622;">
                                        {st.session_state.success_message}
                                    </div>
                                </div>
                            </div>
                            <style>
                                @keyframes fadeIn {{
                                    0% {{ opacity: 0; transform: translateY(-20px); }}
                                    100% {{ opacity: 1; transform: translateY(0); }}
                                }}
                            </style>
                            """, 
                            unsafe_allow_html=True
                        )
                        
                        # Clear the success flag after 3 seconds (on next rerun)
                        # This is useful for subsequent adds
                        if 'success_time' not in st.session_state:
                            st.session_state.success_time = time.time()
                        elif time.time() - st.session_state.success_time > 3:
                            st.session_state.show_success = False
                            st.session_state.pop('success_time', None)
                    
                    # Information about ICPs
                    st.markdown("""
                    ### Tips for effective ICPs
                    - Describe your target professionals in natural language
                    - Include job titles, industries, company types, and locations
                    - Be specific about seniority levels when relevant
                    - Mention company stages or sizes if important
                    - Don't worry about search syntax - the AI will optimize your criteria
                    """)
                
                with right_col:
                    st.markdown("### Your ICPs")
                    
                    # Show existing ICPs with edit/delete options
                    if not st.session_state.queries:
                        st.info("No ICPs added yet. Add your first ICP on the left.")
                    else:
                        for i, query in enumerate(st.session_state.queries):
                            with st.expander(f"ICP #{i+1}", expanded=False):
                                st.code(query)
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button("Delete", key=f"delete_{i}"):
                                        st.session_state.queries.pop(i)
                                        save_queries_to_yaml(st.session_state.queries)
                                        st.rerun()
                                with col2:
                                    if st.button("Edit", key=f"edit_{i}"):
                                        # Store the ICP to edit and its index
                                        st.session_state.editing_icp = query
                                        st.session_state.editing_index = i
                                        # Show the edit dialog
                                        st.session_state.show_edit_dialog = True
                                        st.rerun()
                    
                    # Edit dialog (if needed)
                    if st.session_state.get('show_edit_dialog', False):
                        with st.container():
                            st.markdown("### Edit ICP")
                            
                            edited_icp = st.text_area(
                                "Edit ICP criteria",
                                value=st.session_state.editing_icp,
                                height=100
                            )
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("Save Changes"):
                                    # Update the ICP
                                    if edited_icp.strip():
                                        st.session_state.queries[st.session_state.editing_index] = edited_icp
                                        save_queries_to_yaml(st.session_state.queries)
                                        st.success(f"✅ Updated ICP #{st.session_state.editing_index + 1}")
                                        # Close the dialog
                                        st.session_state.show_edit_dialog = False
                                        st.rerun()
                                    else:
                                        st.error("❌ ICP criteria cannot be empty")
                            
                            with col2:
                                if st.button("Cancel"):
                                    # Close the dialog without saving
                                    st.session_state.show_edit_dialog = False
                                    st.rerun()
                    
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
                        "Maximum results per ICP", 
                        min_value=10, 
                        max_value=1000, 
                        value=100, 
                        step=10,
                        help="Each ICP search will fetch up to this many results"
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
                
                batcher = Batcher(max_in_flight=5, delay=2)  # Create Batcher once
                
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
                                if provider_id:
                                    current_state = unipile_relations_map.get(provider_id)
                                
                                if not current_state:
                                    # Try to match by extracted identifier
                                    for pid, state in unipile_relations_map.items():
                                        if ident in pid:
                                            current_state = state
                                            # Since we found a match, update the row with the provider_id
                                            row['Provider ID'] = pid
                                            break
                                
                                # Check if this profile has a pending invitation
                                has_pending_invite = False
                                if provider_id:
                                    has_pending_invite = provider_id in unipile_invites_map
                                    
                                if not has_pending_invite and not current_state:
                                    # Try to match in invites by identifier
                                    for pid in unipile_invites_map:
                                        if ident in pid:
                                            has_pending_invite = True
                                            # Update provider_id
                                            row['Provider ID'] = pid
                                            break
                                
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
                                    row['Contact Status'] = master_data.get('Last System Action')
                                
                                # Add the row to our filtered list
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
                            "Message sent": ["Follow up", "Mark responded"],
                            "Follow-up sent": ["Follow up again", "Mark responded"],
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
                                options=["", "Not contacted", "Messages generated", "Invited", "Message sent", "Follow-up sent"],
                                default=["Not contacted"],
                                help="Filter by previous contact status"
                            )
                            
                        with filter_col3:
                            recommended_action = st.selectbox(
                                "Recommended Action",
                                options=["All actions", "Need to invite", "Need to generate messages", "Need to follow up", "Connected profiles"],
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
                            if recommended_action == "Need to invite":
                                filtered_df = filtered_df[(filtered_df["Connection State"] == "NOT_CONNECTED") & 
                                                         (filtered_df["Contact Status"].isin(["Not contacted", "Messages generated"]))]
                            elif recommended_action == "Need to generate messages":
                                filtered_df = filtered_df[filtered_df["Contact Status"] == "Not contacted"]
                            elif recommended_action == "Need to follow up":
                                filtered_df = filtered_df[(filtered_df["Connection State"].isin(["INVITED", "PENDING", "CONNECTED"])) & 
                                                         (filtered_df["Contact Status"].isin(["Invited", "Message sent", "Follow-up sent"]))]
                            elif recommended_action == "Connected profiles":
                                filtered_df = filtered_df[filtered_df["Connection State"] == "CONNECTED"]
                                
                        # Add recommended next action based on current statuses
                        filtered_df["Recommended Action"] = filtered_df.apply(
                            lambda row: _determine_next_action(row["Connection State"], row["Contact Status"], connection_actions, workflow_stages),
                            axis=1
                        )
                        
                        # Display filtered profiles with editable fields
                        st.subheader(f"Profiles to Process ({len(filtered_df)} matching)")
                        
                        # Show editable data grid
                        edited_df = st.data_editor(
                            filtered_df,
                            column_config={
                                "LinkedIn URL": st.column_config.TextColumn("LinkedIn URL", disabled=True),
                                "First Name": st.column_config.TextColumn("First Name", disabled=True),
                                "Last Name": st.column_config.TextColumn("Last Name", disabled=True),
                                "Title": st.column_config.TextColumn("Title", disabled=True),
                                "Connection Msg": st.column_config.TextColumn("Connection Msg", width="large"),
                                "Comment Msg": st.column_config.TextColumn("Comment Msg", width="large"),
                                "FU-1": st.column_config.TextColumn("Follow-up 1", width="large"),
                                "FU-2": st.column_config.TextColumn("Follow-up 2", width="large"),
                                "FU-3": st.column_config.TextColumn("Follow-up 3", width="large"),
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
                                status_updates["Contact Status"] = "Invited"
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
                            
                            # Ensure all selected columns exist
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