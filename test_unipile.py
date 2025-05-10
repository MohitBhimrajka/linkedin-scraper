import asyncio
from src.unipile_client import UnipileClient
from src.logging_conf import logger

async def test_api_endpoints():
    """Test the various Unipile API endpoints"""
    client = UnipileClient()
    all_passed = True
    try:
        # Test invitations
        print("1. Testing list_sent_invitations...")
        try:
            invitations = await client.list_sent_invitations()
            print(f"✅ SUCCESS: Retrieved {len(invitations)} invitations")
            if invitations and len(invitations) > 0:
                print(f"   Sample data keys: {list(invitations[0].keys())[:5]}...")
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            all_passed = False
        
        # Test relations
        print("\n2. Testing list_relations...")
        try:
            relations = await client.list_relations()
            print(f"✅ SUCCESS: Retrieved {len(relations)} relations")
            if relations and len(relations) > 0:
                print(f"   Sample data keys: {list(relations[0].keys())[:5]}...")
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            all_passed = False
        
        # Test conversations
        print("\n3. Testing list_conversations...")
        try:
            conversations = await client.list_conversations()
            print(f"✅ SUCCESS: Retrieved {len(conversations)} conversations")
            if conversations and len(conversations) > 0:
                print(f"   Sample data keys: {list(conversations[0].keys())[:5]}...")
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            all_passed = False
        
        return all_passed
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False
    finally:
        await client.close()

if __name__ == "__main__":
    success = asyncio.run(test_api_endpoints())
    print(f"\nOverall test result: {'✅ PASSED' if success else '❌ FAILED'}")
