import os
import getpass
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client
from langchain_community.memory import ConversationBufferMemory
from langchain_community.chains import ConversationalRetrievalChain
from langchain_core.documents import Document
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any, Optional
import hashlib
from pathlib import Path
import asyncio
from langchain_core.retrievers import BaseRetriever
from langchain_community.document_loaders import(
    DirectoryLoader,
    PyPDFLoader, 
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    JSONLoader)

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["SUPABASE_URL"] = os.getenv("SUPABASE_URL")
os.environ["SUPABASE_SERVICE_KEY"] = os.getenv("SUPABASE_SERVICE_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Set up chat model
chat_model = ChatOpenAI(temperature = 0.5) #set at 0.5 

# Initialize Supabase client
supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"]
)

# Initialize chat model
model = ChatOpenAI(model="gpt-4-turbo")

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Create vector store using Supabase 
vector_store = SupabaseVectorStore(
    client = supabase,
    embedding = embeddings,
    table_name = "nods_page_section", 
    query_name = "match_page_sections", 
    
)
class ConversationManager:
    """Centralized manager for conversation history"""
    def __init__(self, retriever, supabase_client, chat_model: Optional[ChatOpenAI] = None):
        self.retriever = retriever
        self.supabase = supabase_client
        self.chat_model = chat_model or ChatOpenAI(temperature=0.5)
        self.memory = ConversationBufferMemory(
            memory_key = "chat_history",
            return_messages = True
        )
        self.rag_chain = self._initialize_rag_chain()
    
    def _initialize_memory(self) -> ConversationBufferMemory:
        """Initialize the conversation memory"""
        return ConversationalRetrievalChain.from_llm(
            llm=self.chat_model,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt_template}
        )
    
    def _initialize_rag_chain(self) -> ConversationalRetrievalChain:
        """Initialize the RAG chain for conversational retrieval"""
        prompt_template = """
        You are a professor at a university helping students understand course materials. 
        Using the following context, guide but do not explicitly answer the student's question.
        Context:{context}

        Question:{question}

        Provide a detailed, educational response that will help with student's learning.
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        return ConversationalRetrievalChain.from_llm(
            llm=self.chat_model,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={'prompt': prompt}
        )
    async def create_conversation(self):
        """Create a new conversation in Supabase"""
        try:
            conversation_data ={
                "started_at": datetime.now().isoformat(),
                "status": "active"
            }

            result = self.supabase.table("conversations").insert(conversation_data).execute()
            print(f"Create conversation result: {result}")
    
            if not result.data:
                raise Exception("No data returned from Supabase")
            
            self.conversation_id = result.data[0]['id']
            print(f"Created conversation with ID: {self.conversation_id}")
            return self.conversation_id
        except Exception as e:
            print(f"Error creating conversation: {str(e)}")
            print(f"Full error details: {e.__dict__}")
            raise
    
    async def store_message(self, conversation_id: int, content: str, role: str = "user") -> Dict:
        """Store a message in the conversation_messages table"""
        try:
            current_time = datetime.now().isoformat()
            message_data = {
                "conversation_id": conversation_id,
                "content": content,
                "role": role,
                "created_at": current_time,
                "updated_at": current_time,
                "timestamp": current_time,
                "metadata": {}
            }

            print(f"Storing message with data: {message_data}") #debug print

            result = self.supabase.table("conversation_messages").insert(message_data).execute()
        
            if not result.data:
                raise Exception("No data returned from message storage")
            
            return result.data[0]
    
        except Exception as e:
            print(f"Error storing message: {str(e)}")
            raise
            
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query and store it in conversation history"""
       try:
            response = self.rag_chain.invoke({"question": query})
            return response
        except Exception as e:
            print(f"Error processing query: {e}")
            return {"error": str(e)}
    
    def get_conversation_history(self) -> list:
        """Get the conversation history from the memory"""
        return self.memory.chat_memory.messages
    
    async def get_conversation_history_from_db(self) -> List[Dict]:
        """Get the conversation history from the database"""
        try:
            result = self.supabase.table("conversation_messages")\
                .select("*")\
                .eq("conversation_id", self.conversation_id)\
                .order("created_at")\
                .execute()
        
            return result.data
        except Exception as e:
            print(f"Error retrieving conversation history: {str(e)}")
            return []


# Function to process the documents 
def process_document(file_path):
    """
    Process a document file into chunks.
    
    Args:
        file_path (str): Path to the document file.
    
    Returns:
        List[Document]: List of document chunks.
    """
    try:
        # Get appropriate loader
        loader = DirectoryLoader(Path(file_path).parent, glob=Path(file_path).name)
        
        # Load the document
        documents = loader.load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )
        chunks = text_splitter.split_documents(documents)
        return chunks
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        raise


#This class stores it into the Supabase table
class DocumentProcessor:
    """
    Class for processing and storing documents in Supabase.
    """
    def __init__(self, supabase_client, embeddings_model):
        self.supabase = supabase_client
        self.embeddings_model = embeddings_model
        self.conversation_manager = None

    def initialize_conversation(self):
        """Initialize or reset the conversation manager"""
        retriever = self._setup_retriever()
        self.conversation_manager = ConversationManager(retriever, self.supabase)
        return self.conversation_manager
    
    def _setup_retriever(self):
        """Set up the retriever for the document processor"""
        return SupabaseVectorStore(
            client=self.supabase,
            embedding=self.embeddings_model,
            table_name="nods_page_section",
            query_name="match_page_sections"
        ).as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
    
    async def process_and_store_document(self, 
                                       file_path: str,
                                       parent_page_id: int = None,
                                       document_type: str = None) -> Dict[str, Any]:
        """
        Main entry point for document processing and storage
        """
        try:
            # Check if document already exists in Supabase
            result =  await self.supabase.table("nods_page_section").select("*").eq("path", file_path).execute()
            if result.data:
                print(f"Document already exists in Supabase: {file_path}")
                return{
                    "status": "skipped",
                    "message": f"Document already exists in Supabase"
                }
            #process the document into chunks using process_chunks method
            chunks = self.process_document(file_path)
            
            # Create page entry
            page_data = {
                "parent_page_id": parent_page_id,
                "path": file_path,
                "checksum": self.calculate_checksum(str(chunks)),
                "meta": {
                    "processed_at": datetime.now().isoformat(),
                    "chunk_count": len(chunks)
                },
                "type": document_type or file_path.split('.')[-1],
                "source": "file_upload"
            }
            
            # Insert page and get ID
            page_result = await self.supabase.table("nods_page").insert(page_data).execute()
            page_id = page_result.data[0]['id']
            
            # Process and store sections
            sections = []
            for i, chunk in enumerate(chunks):
                try:
                # Generate embedding
                    embedding = self.embeddings_model.embed_query(chunk.page_content)
                
                    section_data = {
                    "page_id": page_id,
                    "content": chunk.page_content,
                    "token_count": len(chunk.page_content.split()),  # Simple token count
                    "embedding": embedding,
                    "slug": f"section-{i}",
                    "heading": chunk.metadata.get('heading', f"Section {i}")
                    }
                    sections.append(section_data)
                
                # Store in batches of 100
                    if len(sections) >= 100:
                        self.supabase.table("nods_page_section").insert(sections).execute()
                        sections = []
                except Exception as e:
                    print(f"Error processing chunck {i}: {str(e)}")
                    continue

            # Store any remaining sections
            if sections:
                await self.supabase.table("nods_page_section").insert(sections).execute()
                
            return {
                "status": "success",
                "page_id": page_id,
                "sections_count": len(chunks)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }

class BatchDocumentProcessor:
    """
    Class for processing a batch of documents.
    """
    def __init__(self, doc_processor: DocumentProcessor):
        self.doc_processor = doc_processor

    async def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Process all supported documents in a directory"""
        directory = Path(directory_path)
        results = []
        
        # Get all files in directory
        supported_extensions = {
            '.pdf', '.docx', '.doc', '.txt', '.csv', 
            '.xlsx', '.xls', '.html', '.md', '.json',
            '.pptx', '.ppt'
        }
        
        files = [
            f for f in directory.glob('**/*') 
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]
        
        # Process each file
        for file_path in files:
            try:
                result = await self.doc_processor.process_and_store_document(
                    str(file_path),
                    document_type=file_path.suffix[1:]  # Remove the dot from extension
                )
                results.append({
                    "file": str(file_path),
                    "result": result
                })
                print(f"Processed {file_path}")
            except Exception as e:
                results.append({
                    "file": str(file_path),
                    "error": str(e)
                })
                print(f"Error processing {file_path}: {e}")
        
        return results

async def process_and_query_documents(docs_directory: str, query: str): 
    try:
        # Initialize document processor and conversation manager
        doc_processor = DocumentProcessor(supabase, embeddings)
        conversation_manager = doc_processor.initialize_conversation()
        
        #create new conversation
        await conversation_manager.create_conversation()
        
        #process all documents
        batch_processor = BatchDocumentProcessor(doc_processor)
        processing_results = await batch_processor.process_directory(docs_directory)

        print("\n=== Document Processing Results ===")
        print("Documents processed:", len(processing_results))
        print("Successful:", len([r for r in processing_results if r['result'].get('status') == 'success']))
        print("Skipped:", len([r for r in processing_results if r['result'].get('status') == 'skipped']))
        print("Errors:", len([r for r in processing_results if r.get('error')]))

    
        #process query using conversation manager
        print("\n=== Query Processing ===")
        print("Query:", query)

        response = await conversation_manager.process_query(query)

        # Get relevant sections
        if response["status"] == "success":
            print(response["answer"])
            print(f"\nConversation ID: {response["conversation_id"]}")
            print(f"Source Documents: {len(response["source_documents"])}")
            
            # Retrieve conversation history
            history = await conversation_manager.get_conversation_history_from_db()
            print("\nStored Messages:", len(history))
        else:
            print("\nError processing query:", response["error"])
        
        return response

    except Exception as e:
        print(f"\nError in process_and_query_documents: {str(e)}")
        return None, None
    

if __name__ == "__main__":
    # Directory containing your documents
    docs_directory = "/Users/miki/desktop/langchain_rag/data/"
    query = "What are the main topics across these documents?"
    
    # Run the async function
    response = asyncio.run(process_and_query_documents(docs_directory, query))
