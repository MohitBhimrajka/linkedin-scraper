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

# Import from our scraper
from src.runner import process_query, load_queries
from src.throttle import Batcher
from src.transform import normalize_results, Profile
from src.sheets import append_rows, append_icp_results, get_spreadsheet, get_google_sheets_client, generate_sheet_name
from src.logging_conf import logger
from src.campaign import run_campaign


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
    You are an expert in crafting highly effective Google search queries, specifically for finding LinkedIn profiles via a Google Custom Search Engine (CSE) that is already configured to search *only* within LinkedIn.

    Your task is to transform the given "Original ICP criteria" into an OPTIMIZED, SINGLE-LINE Google search string.

    **Key Guidelines for Optimization:**

    1.  **Preserve Core Intent:** The optimized query must accurately reflect ALL aspects of the original criteria. Do not add new concepts or omit existing ones.
    2.  **Leverage Google Search Operators:**
        *   Use `AND` (often implicit, but can be explicit for clarity), `OR` (for alternatives), `NOT` (or `-` prefix) for exclusions.
        *   Use `"quotation marks"` for exact phrases (e.g., "Software Engineer", "Product Management").
        *   Use `(parentheses)` for grouping complex conditions and controlling the order of operations.
        *   Strategically use `site:linkedin.com/in/` at the beginning of the query. While the CSE is configured for LinkedIn, explicitly adding this can sometimes refine results further to personal profiles and is a good practice for general Google dorking.
    3.  **Focus on LinkedIn Profile Keywords:** Think about terms most likely to appear directly in LinkedIn profiles:
        *   Job titles (e.g., "Chief Technology Officer", "VP of Sales")
        *   Skills (e.g., "Python", "Strategic Planning", "SaaS")
        *   Industry terms (e.g., "Fintech", "Healthcare IT")
        *   Company names (if relevant, use `("Company A" OR "Company B")`)
        *   Location (e.g., "San Francisco Bay Area", "London")
        *   Seniority indicators (e.g., "Director", "Manager", "Lead", "Senior")
    4.  **Conciseness and Effectiveness:** Aim for a query that is as concise as possible while maximizing relevance and minimizing noise.
    5.  **Structure:** Translate the *meaning* of the original criteria into a functional search string. Do not try to replicate the visual structure (like bullet points) of the input if it's not optimal for a search query.

    **Output Requirements:**
    *   You MUST return ONLY the optimized search criteria text as a single, continuous string.
    *   Do NOT include any explanations, labels (like "Optimized Query:"), or any other text before or after the search string.
    *   Do not include the site linkedin.com/in/ in the search string as I am using a CSE that is already configured to search *only* within LinkedIn.

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
    
    # SETUP PAGE - Setup ICPs and settings
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
                
                st.text_input("Google API Key", value=google_api_key, type="password", 
                            help="Required for Google Custom Search API")
                st.text_input("Custom Search Engine ID (CX)", value=cx_id, type="password", 
                            help="ID of your Google Custom Search Engine")
                st.text_input("Google Sheet ID", value=sheet_id, 
                            help="ID of the Google Sheet where results will be stored")
                st.text_input("Gemini API Key", value=gemini_api_key, type="password", 
                            help="Required for ICP optimization (automatic)")
                
                st.markdown("""
                > ℹ️ These values are read from your `.env` file. 
                > Edit the `.env` file to change them permanently.
                """)
            
            with col2:
                st.markdown("### Service Account")
                service_account_exists = os.path.exists("service-account.json")
                
                if service_account_exists:
                    st.success("✅ Service account file found (service-account.json)")
                else:
                    st.error("❌ Service account file not found")
                    st.markdown("""
                    1. Create a service account in Google Cloud Console
                    2. Download the JSON key file
                    3. Rename it to `service-account.json`
                    4. Place it in the `linkedin-scraper` directory
                    """)
                
                st.markdown("### Google Sheet Setup")
                if sheet_id:
                    sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit"
                    st.markdown(f"📊 [Open your Google Sheet]({sheet_url})")
                    st.markdown("""
                    > Make sure you've shared your Google Sheet with the service account email
                    > with Editor permissions.
                    """)
                else:
                    st.warning("⚠️ No Google Sheet ID configured")
        
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
                
                # Buttons
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
                
                # Run settings
                st.markdown("---")
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
    
    # PROCESSING PAGE - Show optimization and search progress
    elif st.session_state.page == 'processing':
        st.subheader("Processing LinkedIn Profiles")
        
        # Progress indicators
        status_container = st.container()
        
        with status_container:
            # Phase 1: Optimize ICPs
            st.markdown("### Phase 1: Optimizing ICPs with AI")
            optimization_progress = st.progress(0)
            optimization_status = st.empty()
            
            # Optimize all ICPs
            if not st.session_state.get('optimization_complete', False):
                optimization_status.text("Optimizing your ICPs with Gemini AI...")
                
                # Show original ICPs
                st.markdown("#### Your Original ICPs:")
                for i, icp in enumerate(st.session_state.queries):
                    st.code(icp, language=None)
                
                # Process ICPs
                async def optimize_and_store():
                    optimized = await optimize_all_icps(st.session_state.queries)
                    st.session_state.optimized_icps = optimized
                    st.session_state.optimization_complete = True
                
                # Simulate progress for better UX
                placeholder_progress = st.empty()
                for i in range(len(st.session_state.queries)):
                    optimization_progress.progress((i + 0.5) / len(st.session_state.queries))
                    placeholder_progress.text(f"Optimizing ICP #{i+1}...")
                    time.sleep(1)  # Simulate processing time
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(optimize_and_store())
                
                # Complete progress
                optimization_progress.progress(1.0)
                optimization_status.text("✅ ICP optimization complete!")
                placeholder_progress.empty()
                
                # Display optimized ICPs
                st.markdown("#### Optimized ICPs:")
                for i, icp_data in enumerate(st.session_state.optimized_icps):
                    with st.expander(f"ICP #{i+1}", expanded=True):
                        st.markdown("**Original:**")
                        st.code(icp_data["original"], language=None)
                        st.markdown("**Optimized:**")
                        st.code(icp_data["optimized"], language=None)
                
                st.rerun()  # Refresh to show Phase 2
            
            # If optimization is complete, show Phase 2
            if st.session_state.get('optimization_complete', False):
                # Phase 2: Search for profiles
                st.markdown("### Phase 2: Searching for LinkedIn Profiles")
                search_progress = st.progress(0)
                search_status = st.empty()
                
                if not st.session_state.get('search_complete', False):
                    search_status.text("Searching for matching LinkedIn profiles...")
                    
                    # Run the search
                    async def search_and_store():
                        results_df, updated_icps = await run_scraper(
                            st.session_state.optimized_icps,
                            st.session_state.run_limit
                        )
                        st.session_state.search_results = results_df
                        st.session_state.optimized_icps = updated_icps
                        st.session_state.search_complete = True
                    
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(search_and_store())
                    
                    # Complete progress
                    search_progress.progress(1.0)
                    search_status.text("✅ LinkedIn profile search complete!")
                    
                    # Move to results page
                    st.session_state.page = 'results'
                    st.rerun()
    
    # RESULTS PAGE - Show search results
    elif st.session_state.page == 'results':
        st.subheader("Found LinkedIn Profiles")
        
        # Show navigation buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("◀️ Back to ICPs", use_container_width=True):
                # Reset states
                st.session_state.page = 'setup'
                st.session_state.optimization_complete = False
                st.session_state.search_complete = False
                del st.session_state.optimized_icps
                del st.session_state.search_results
                st.rerun()
        
        with col2:
            if st.button("📤 Go to Campaign", use_container_width=True):
                st.session_state.page = 'campaign'; st.rerun()
        
        # Results summary
        results_df = st.session_state.search_results
        
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
            
            # Show tabs for results by ICP and all results
            icp_results_tab, all_results_tab = st.tabs(["Results by ICP", "All Results"])
            
            with icp_results_tab:
                # Show results by ICP
                for i, icp_data in enumerate(st.session_state.optimized_icps):
                    with st.expander(f"ICP #{i+1} - {len(icp_data.get('results', [])) if 'results' in icp_data else 0} profiles", expanded=i==0):
                        st.markdown("**Original ICP:**")
                        st.code(icp_data["original"])
                        
                        st.markdown("**Optimized ICP:**")
                        st.code(icp_data["optimized"])
                        
                        st.markdown(f"**Found {icp_data.get('result_count', 0)} profiles**")
                        
                        # If we have individual results for this ICP
                        if 'results' in icp_data and icp_data['results']:
                            # Convert to profile objects and then DataFrame for display
                            icp_profiles = normalize_results(icp_data['results'])
                            if icp_profiles:
                                profiles_data = []
                                for profile in icp_profiles:
                                    profiles_data.append({
                                        "LinkedIn URL": profile.linkedin_url,
                                        "Title": profile.title or "",
                                        "Name": f"{profile.first_name or ''} {profile.last_name or ''}".strip(),
                                        "Description": profile.description or ""
                                    })
                                
                                icp_df = pd.DataFrame(profiles_data)
                                st.dataframe(icp_df, use_container_width=True)
                            else:
                                st.info("No unique profiles found for this ICP.")
                        else:
                            st.info("No results data available for this ICP.")
            
            with all_results_tab:
                # Show all results combined
                st.dataframe(results_df, use_container_width=True)
                
                # Download as CSV
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="linkedin_profiles.csv",
                    mime="text/csv",
                )
        else:
            st.warning("No profiles found matching your ICPs. Try refining your criteria.")

    # CAMPAIGN PAGE - Run outreach campaigns
    elif st.session_state.page == 'campaign':
        st.subheader("Run Campaign")

        # Fetch worksheets list
        sheet_id = os.getenv("GOOGLE_SHEET_ID")
        spreadsheet = get_spreadsheet(sheet_id)
        sheet_titles = [ws.title for ws in spreadsheet.worksheets() if ws.title not in ("Master_Sheet","Main Results")]

        # Sheet selection
        selected_sheet = st.selectbox("Select ICP sheet to target", sheet_titles)
        
        if selected_sheet:
            # Campaign mode selection
            mode = st.radio(
                "What should happen?", 
                ["Generate only", "Invite only", "Invite + Comment", "Full (invite, comment, follow-ups, InMail)"],
                help="Generate: just create messages, Invite: send connection requests, Full: all actions"
            )
            
            # Follow-up configuration
            st.subheader("Follow-up Timing")
            col1, col2, col3 = st.columns(3)
            with col1:
                follow1 = st.number_input("Days until follow‑up 1", 1, 30, 3)
            with col2:
                follow2 = st.number_input("Days until follow‑up 2", 1, 60, 7)
            with col3:
                follow3 = st.number_input("Days until follow‑up 3", 1, 90, 14)
            
            # Load profiles from selected sheet
            ws = spreadsheet.worksheet(selected_sheet)
            try:
                data = ws.get_all_records()
            except Exception as e:
                st.error(f"Error loading sheet data: {str(e)}")
                st.info("Trying to load with explicit headers...")
                # Define the expected headers
                expected_headers = [
                    "LinkedIn URL", "Title", "First Name", "Last Name", "Description", 
                    "Profile Image URL", "Connection Msg", "Comment Msg", "F/U‑1", 
                    "F/U‑2", "F/U‑3", "InMail", "Contact Status", "Last Action UTC", "Error Msg"
                ]
                try:
                    data = ws.get_all_records(expected_headers=expected_headers)
                except Exception as e2:
                    st.error(f"Failed to load sheet data: {str(e2)}")
                    data = []
            
                                        # Create DataFrame for display and selection
            df = pd.DataFrame(data)
            
            if not df.empty:
                # Preview profiles
                st.subheader("Profile Preview")
                st.dataframe(df[["LinkedIn URL", "Title", "First Name", "Last Name"]], use_container_width=True)
                
                # Profile selection
                selected_indices = st.multiselect(
                    "Preview & choose specific profiles",
                    options=list(range(len(df))),
                    format_func=lambda i: f"{df.iloc[i]['First Name']} {df.iloc[i]['Last Name']} - {df.iloc[i]['Title']}"
                )
                
                # Limit setting
                max_sends = len(selected_indices) if selected_indices else len(df)
                limit = st.number_input("Maximum profiles to process in this run", 1, max_sends, min(10, max_sends))
                
                # Launch button
                if st.button("▶️ Launch Campaign", type="primary"):
                    
                    # Create targets list from selected rows or all rows
                    targets = []
                    
                    if selected_indices:
                        # Use only selected rows
                        for i in selected_indices[:limit]:
                            row = df.iloc[i]
                            targets.append(Profile(
                                linkedin_url=row["LinkedIn URL"],
                                title=row["Title"],
                                first_name=row["First Name"],
                                last_name=row["Last Name"],
                                description=row["Description"],
                                profile_image_url=row["Profile Image URL"]
                            ))
                    else:
                        # Use all rows up to limit
                        for i, row in df.head(limit).iterrows():
                            targets.append(Profile(
                                linkedin_url=row["LinkedIn URL"],
                                title=row["Title"],
                                first_name=row["First Name"],
                                last_name=row["Last Name"],
                                description=row["Description"],
                                profile_image_url=row["Profile Image URL"]
                            ))
                    
                    if targets:
                        
                        # Show progress
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        status_text.text("Starting campaign...")
                        
                        # Run campaign with selected mode
                        stats = asyncio.run(run_campaign(
                            profiles=targets, 
                            followup_days=(follow1, follow2, follow3),
                            mode=mode,  # Pass the full mode string instead of just the first word
                            spreadsheet_id=sheet_id,
                            sheet_name=selected_sheet
                        ))
                        
                        progress_bar.progress(1.0)
                        
                        # Show results based on mode
                        if mode.startswith("Generate"):
                            st.success(f"✅ Generated messages for {stats.generated} profiles. Check the sheet for results.")
                        else:
                            st.success(f"✅ Campaign completed! Generated: {stats.generated}, Sent: {stats.sent}, Errors: {stats.errors}")
                            
                        # Add a link to the sheet
                        sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit#gid={ws.id}"
                        st.markdown(f"[View updated profiles in Google Sheets]({sheet_url})")
                    else:
                        st.warning("No profiles selected to process!")
            else:
                st.warning(f"The selected sheet '{selected_sheet}' has no data.")


if __name__ == "__main__":
    main() 