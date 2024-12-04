import os
from dotenv import load_dotenv
from supabase.client import create_client

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
        
        # Test if tables exist
        page_sections = supabase.table("nods_page_section").select("*").limit(1).execute()
        print("\nTables check:")
        print("- nods_page: ✓")
        print("- nods_page_section: ✓")
        
        return True
        
    except Exception as e:
        print(f"Connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_supabase_connection()