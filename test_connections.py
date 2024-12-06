import os
from dotenv import load_dotenv
from supabase.client import create_client
from datetime import datetime

def test_supabase_connection():
    try:
        # Load environment variables
        load_dotenv()
        
        # Print URLs and keys (optional, for debugging)
        print("URL:", os.getenv("SUPABASE_URL"))
        print("Key exists:", bool(os.getenv("SUPABASE_SERVICE_KEY")))
        
        # Create client
        supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_SERVICE_KEY")
        )
        
        # Test connection by trying to fetch from nods_page
        response = supabase.table("nods_page").select("*").limit(1).execute()
        print("Connection successful!")
        print(f"Response: {response}")
        
        # Test if tables exist and count rows
        tables = ['nods_page', 'nods_page_section', 'conversations', 'conversation_messages']
        print("\nTables check:")
        for table in tables:
            try:
                result = supabase.table(table).select("count", count='exact').execute()
                print(f"- {table}: ✓ ({result.count} rows)")
            except Exception as e:
                print(f"- {table}: ✗ (Error: {str(e)})")

        # Test insert operation with error catching
        print("\nTesting insert operation:")
        try:
            test_data = {
                "started_at": datetime.now().isoformat(),
                "status": "test"
            }
            print(f"Attempting to insert test data: {test_data}")
            result = supabase.table("conversations").insert(test_data).execute()
            print(f"Insert successful! Result: {result}")
            
            # Clean up test data
            if result.data and len(result.data) > 0:
                test_id = result.data[0]['id']
                supabase.table("conversations").delete().eq('id', test_id).execute()
                print("Test data cleaned up successfully")
                
        except Exception as e:
            print(f"Insert operation failed: {str(e)}")
            if hasattr(e, 'response'):
                print(f"Response: {e.response}")
        
        return True
        
    except Exception as e:
        print(f"Connection failed: {str(e)}")
        if hasattr(e, '__dict__'):
            print(f"Error details: {e.__dict__}")
        return False

if __name__ == "__main__":
    print("\n=== Testing Supabase Connection and Operations ===")
    test_supabase_connection()