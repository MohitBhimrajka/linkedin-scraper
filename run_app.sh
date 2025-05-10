#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
  source venv/bin/activate
fi

# Install requirements if needed
if [ "$1" == "--install" ]; then
  echo "Installing requirements..."
  pip install -r requirements.txt
fi

# Change to the linkedin-scraper directory
cd "$(dirname "$0")"

# Run the Streamlit app
echo "Starting LinkedIn Lead Generator app..."
streamlit run streamlit_app.py 