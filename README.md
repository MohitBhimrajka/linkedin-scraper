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

3. Create a `.env` file with your credentials (see `.env.example`)

## Environment Variables

- `GOOGLE_API_KEY`: Your Google API key with Custom Search API enabled
- `CX_ID`: Your Custom Search Engine ID
- `GOOGLE_SHEET_ID`: The ID of the Google Sheet where results will be stored
- `GEMINI_API_KEY`: (Optional) Your Gemini API key for query optimization

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