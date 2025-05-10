# LinkedIn Scraper

A Python application that uses Google Custom Search to find LinkedIn profiles matching specific search criteria and stores the results in Google Sheets.

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Set up Google API credentials:
   - Create a Google Custom Search Engine (CSE) and get your CX_ID
   - Generate a Google API Key with access to Custom Search API
   - Create a Google Service Account for Sheets API access and download the JSON key
   - Share your target Google Sheet with the service account email
   - (Optional) Get a Gemini API key for query optimization

### Unipile credentials
1. Sign up for Unipile and create a LinkedIn connector  
2. Copy **API Key**, **DSN**, and **LinkedIn Account ID**  
3. Add them to `.env` (`UNIPILE_API_KEY`, `UNIPILE_DSN`, `UNIPILE_ACCOUNT_ID`)

3. Create a `.env` file with your credentials (see `.env.example`)

## Environment Variables

- `GOOGLE_API_KEY`: Your Google API key with Custom Search API enabled
- `CX_ID`: Your Custom Search Engine ID
- `GOOGLE_SHEET_ID`: The ID of the Google Sheet where results will be stored
- `GEMINI_API_KEY`: (Optional) Your Gemini API key for query optimization
- `UNIPILE_API_KEY`: Your Unipile API key for LinkedIn automation
- `UNIPILE_DSN`: Your Unipile data source name (e.g., api1.unipile.com:13111)
- `UNIPILE_ACCOUNT_ID`: Your LinkedIn account ID from Unipile

## Usage

### Command Line Interface

Run the script with optional limit parameter (default 100):
```bash
python main.py --limit 200
```

The script will:
1. Load search queries from `config/queries.yaml`
2. Execute Google Custom Searches for each query (paginated)
3. Process and normalize the results
4. Append unique profiles to Google Sheets

### Streamlit Web Interface

For a user-friendly interface, run the Streamlit app:
```bash
# Option 1: Use the provided script
./run_app.sh

# Option 2: Run manually
streamlit run streamlit_app.py
```

The Streamlit app provides:
- A visual interface to manage queries
- Query optimization using Gemini AI
- Real-time progress updates
- Preview of results before they're added to Google Sheets
- CSV export option

## Output Columns

The results are stored in Google Sheets with the following columns:
1. LinkedIn URL
2. Title
3. First Name
4. Last Name 
5. Description
6. Profile Image URL 
7. Connection Msg
8. Comment Msg
9. F/U-1
10. F/U-2
11. F/U-3
12. InMail
13. Contact Status
14. Last Action UTC

## Running a Campaign

Once you've collected LinkedIn profiles, you can run automated outreach campaigns:

1. Navigate to the Campaign tab in the Streamlit app
2. Select which ICP sheets (target audiences) to include
3. Configure follow-up timing (1st, 2nd, and 3rd follow-up messages)
4. Click "Launch Campaign" to begin outreach

The campaign will:
- Enrich each LinkedIn profile with additional data
- Generate personalized connection messages using Gemini AI
- Send connection requests with custom messages
- Comment on recent posts (if available)
- Schedule follow-up messages to be sent automatically after connections are accepted
- Track progress in the Google Sheet