import os
import getpass
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any
import hashlib
from pathlib import Path
import asyncio
from langchain_community.document_loaders import(
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
chat_model = ChatOpenAI(temperature = 0.7)

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
    query_name = "match_page_sections" 
)

# Function to get loader based on file extension
def get_loader(file_path):
    """Return appropriate loader based on file extension"""
    extension = file_path.lower().split('.')[-1]
    
    loaders = {
        # Documents
        'pdf': PyPDFLoader,
      
        # Data files
        'txt': TextLoader,
        'csv': CSVLoader,
        'xls': UnstructuredExcelLoader,
        'xlsx': UnstructuredExcelLoader,
        
        # Web content
        'html': UnstructuredHTMLLoader,
        'htm': UnstructuredHTMLLoader,
        'md': UnstructuredMarkdownLoader,
        'json': lambda fp: JSONLoader(fp, jq_schema='.[]')  # Adjust jq_schema as needed
    }
    loader_class = loaders.get(extension)
    if loader_class is None:
        raise ValueError(f"Unsupported file type: .{extension}")
    
    return loader_class(file_path)

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
        loader = get_loader(file_path)
        
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

# Function to store documents into vector store
def store_documents_in_supabase(
    chunks: List[Document], 
    batch_size: int = 100,
    additional_metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Store document chunks in Supabase with batching and metadata.
    
    Args:
        chunks (List[Document]): List of Document objects to store.
        batch_size (int): Number of chunks to process in each batch.
        additional_metadata (Dict[str, Any]): Optional additional metadata to add to all chunks.
    
    Returns:
        Dict[str, Any]: Status and processing information.
    """
    try:
        total_chunks = len(chunks)
        successful_batches = 0
        failed_batches = []
        
        # Add base metadata to all chunks
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "timestamp": datetime.now().isoformat(),
                "chunk_index": i,
                "total_chunks": total_chunks,
                "batch_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            })
            
            # Add any additional metadata if provided
            if additional_metadata:
                chunk.metadata.update(additional_metadata)
        
        # Process in batches
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            try:
                vector_store.add_documents(batch)
                successful_batches += 1
                print(f"Processed batch {successful_batches}: chunks {i} to {i + len(batch)}")
            except Exception as batch_error:
                failed_batches.append({
                    "batch_number": successful_batches + 1,
                    "start_index": i,
                    "error": str(batch_error)
                })
                print(f"Error in batch {successful_batches + 1}: {str(batch_error)}")
        
        # Prepare result report
        result = {
            "status": "success" if not failed_batches else "partial_success",
            "total_chunks": total_chunks,
            "successful_batches": successful_batches,
            "chunks_stored": successful_batches * batch_size,
            "failed_batches": failed_batches,
            "metadata_sample": chunks[0].metadata if chunks else None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log warning if any batches failed
        if failed_batches:
            print(f"Warning: {len(failed_batches)} batches failed to process")
            
        return result
        
    except Exception as e:
        error_result = {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "total_chunks": len(chunks),
            "chunks_processed": successful_batches * batch_size if 'successful_batches' in locals() else 0,
            "timestamp": datetime.now().isoformat()
        }
        print(f"Error storing documents: {str(e)}")
        return error_result

# Set up retrieval chain
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Number of relevant chunks to retrieve
)

# Set up memory for conversation history
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Creating a Chat prompt template 
prompt_template = """
You are a professor at a university helping students understand course materials. 
Using the following context, guide but not explicitly answer the student's question.
Context:{context}

Question:{question}

Provide a detailed, educational response that will help with student's learning.
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# Function to log queries and responses
def log_query(question: str, response: Dict[str, Any]):
    """
    Log queries and responses for analysis.
    
    Args:
        question (str): The user's question.
        response (Dict[str, Any]): The system's response.
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "response": response["answer"] if response["status"] == "success" else None,
        "status": response["status"],
        "error": response.get("error")
    }
    # Store in Supabase (implementation not shown)

class DocumentProcessor:
    """
    Class for processing and storing documents in Supabase.
    """
    def __init__(self, supabase_client, embeddings_model):
        self.supabase = supabase_client
        self.embeddings_model = embeddings_model
        
    def calculate_checksum(self, content: str) -> str:
        """Calculate SHA-256 checksum of content"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def process_and_store_document(self, 
                                       file_path: str,
                                       parent_page_id: int = None,
                                       document_type: str = None) -> Dict[str, Any]:
        """
        Process and store a document in Supabase.
        
        Args:
            file_path (str): Path to the document file.
            parent_page_id (int, optional): ID of the parent page.
            document_type (str, optional): Type of the document.
        
        Returns:
            Dict[str, Any]: Status and processing information.
        """
        try:
            # Process document into chunks
            chunks = process_document(file_path)
            
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
            page_result = self.supabase.table("nods_page").insert(page_data).execute()
            page_id = page_result.data[0]['id']
            
            # Process and store sections
            sections = []
            for i, chunk in enumerate(chunks):
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
            
            # Store any remaining sections
            if sections:
                self.supabase.table("nods_page_section").insert(sections).execute()
                
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

class EnhancedRetriever:
    """Class for retrieving relevant sections from Supabase."""
    def __init__(self, supabase_client, embeddings_model):
        self.supabase = supabase_client
        self.embeddings_model = embeddings_model
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Required method for Langchain retriever interface"""
        # Convert the async method to sync for Langchain compatibility
        sections = self.supabase.rpc(
            'match_page_sections',
            {
                'embedding': self.embeddings_model.embed_query(query),
                'match_threshold': 0.5,
                'match_count': 5,
                'min_content_length': 50
            }
        ).execute()
        
        # Convert sections to Langchain Documents
        documents = []
        for section in sections.data:
            # Get parent pages for context
            parents = self.supabase.rpc(
                'get_page_parents',
                {'page_id': section['page_id']}
            ).execute()
            
            # Create metadata including parents
            metadata = {
                'page_id': section['page_id'],
                'parents': parents.data
            }
            
            # Create Document object
            doc = Document(
                page_content=section['content'],
                metadata=metadata
            )
            documents.append(doc)
        
        return documents
# Function to set up the RAG chain
def setup_rag_chain(document_processor: DocumentProcessor, 
                   retriever: EnhancedRetriever,
                   chat_model: ChatOpenAI):
    """
    Set up the Retrieval-Augmented Generation (RAG) chain.
    
    Args:
        document_processor (DocumentProcessor): Instance of DocumentProcessor.
        retriever (EnhancedRetriever): Instance of EnhancedRetriever.
        chat_model (ChatOpenAI): Instance of ChatOpenAI.
    
    Returns:
        ConversationalRetrievalChain: The configured RAG chain.
    """
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={'prompt': prompt}
    )
    
    return rag_chain

class BatchDocumentProcessor:
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

async def process_and_query_documents(directory_path: str, query: str):
    try:
        # Initialize batch processor
        doc_processor = DocumentProcessor(supabase, embeddings)
        batch_processor = BatchDocumentProcessor(doc_processor)
        
        # Process all documents
        processing_results = await batch_processor.process_directory(directory_path)
        print("Documents processing results:", processing_results)

        # Query across all processed documents
        #From:
        sections = await retriever.get_relevant_documents(
            query,
            match_threshold=0.5,
            match_count=5
        )
        #To 
        sections = retriever.get_relevant_documents(query)
        print("Retrieved sections:", sections)

        return processing_results, sections

    except Exception as e:
        print(f"Error: {str(e)}")
        return None, None

if __name__ == "__main__":
    # Directory containing your documents
    docs_directory = "/Users/miki/desktop/langchain_rag/data/"
    query = "What are the main topics across these documents?"
    
    # Run the async function
    results, sections = asyncio.run(process_and_query_documents(docs_directory, query))
