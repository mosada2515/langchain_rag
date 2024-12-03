import os
import getpass
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client
from langchain.memory import RetrievalQA #basic memory
from langchain.memory import SimpleMemory
from langchain.memory import ConversationBufferMemory ##these tow for advance RAG with memory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from langchain_community.document_loaders import(
    PyPDFLoader, 
    PowerPointLoader,
    WordDocumentLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    JSONLoader)


#environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["SUPABASE_URL"] = os.getenv("SUPABASE_URL")
os.environ["SUPABASE_SERVICE_KEY"] = os.getenv("SUPABASE_SERVICE_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

#Initialize Supabase client
supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"]
)
#initialize chat model
model = ChatOpenAI(model="gpt-4-turbo")
#initialize embeddings
embeddings = OpenAIEmbeddings()

#create vector store using Supabase 
vector_store = SupabaseVectorStore(
    client = supabase,
    embedding = embeddings,
    table_name = "documents", ###remeber to create this table in supabase
    query_name = "match_documents" ###remeber to create this query in supabase
)

#function to get loader based on file extension
def get_loader(file_path):
    """Return appropriate loader based on file extension"""
    extension = file_path.lower().split('.')[-1]
    
    loaders = {
        # Documents
        'pdf': PyPDFLoader,
        'ppt': PowerPointLoader,
        'pptx': PowerPointLoader,
        'doc': WordDocumentLoader,
        'docx': WordDocumentLoader,
        
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

#function to process the documents
def process_document(file_path):
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


#Creating a Chat prompt template 
prompt_template = """
You are a professor at a university helping students understand course materials. 
Using the following context, guide but not explicitly answer the student's question.
Context:{context}

Question:{question}

Provide a detailed, educational response that will help with student's learning.
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

#set up chat model
chat_model = ChatOpenAI(temperature = 0.7)
