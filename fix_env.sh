#!/bin/bash

# Script to fix the .env file by removing inline comments
# Created to fix the issue with UNIPILE_DSN

# Backup the original file
cp .env .env.backup

# Process the file and fix the UNIPILE_DSN line
cat .env | sed 's/UNIPILE_DSN=\(.*\)#.*/UNIPILE_DSN=\1/' > .env.fixed

# Check if the processing completed successfully
if [ $? -eq 0 ]; then
    # Replace the original file with the fixed one
    mv .env.fixed .env
    chmod 600 .env  # Set proper permissions
    echo "✅ .env file has been fixed successfully."
    echo "Original file was backed up as .env.backup"
    echo "UNIPILE_DSN now contains only the value with no comments."
else
    echo "❌ Error occurred while trying to fix the .env file."
    echo "Please check the file and modify it manually."
fi

# Verify the fix
echo ""
echo "UNIPILE_DSN value is now:"
grep UNIPILE_DSN .env

echo ""
echo "You can test the connection with:"
echo "curl -s -H \"X-API-KEY:\$UNIPILE_API_KEY\" \"https://\$UNIPILE_DSN/api/v1/status\"" 