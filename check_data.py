import asyncio
from src.unipile_client import UnipileClient
from src.logging_conf import logger

async def check_api_data():
    """Check one API call and print the result structure"""
    client = UnipileClient()
    try:
        # Test invitations
        print("Testing list_sent_invitations...")
        invitations = await client.list_sent_invitations()
        print(f"Retrieved data type: {type(invitations)}")
        print(f"Data structure: {invitations}")
        
        # Test relations
        print("\nTesting list_relations...")
        relations = await client.list_relations()
        print(f"Retrieved data type: {type(relations)}")
        print(f"Data structure: {relations}")
        
        # Test conversations
        print("\nTesting list_conversations...")
        conversations = await client.list_conversations()
        print(f"Retrieved data type: {type(conversations)}")
        print(f"Data structure: {conversations}")
        
        return True
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(check_api_data()) 