import os
import json
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pypdf import PdfReader
from pdf2image import convert_from_path
import asyncio
from datetime import datetime
import requests
from pydantic import BaseModel

# Vespa imports
from vespa.package import (
    ApplicationPackage, Field, Schema, Document, HNSW,
    RankProfile, Function, FieldSet, SecondPhaseRanking,
    Summary, DocumentSummary
)
from vespa.deployment import VespaCloud
from vespa.application import Vespa
from vespa.io import VespaResponse

# Supabase
from supabase.client import create_client

# ColPali imports
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device
from vidore_benchmark.utils.image_utils import scale_image, get_base64_image

# Google Gemini
import google.generativeai as genai

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


class EducationalVisualRAG:
    def __init__(self):
        # Initialize environment variables
        self.VESPA_TENANT_NAME = "hakness"
        self.VESPA_APP_NAME = "harknessapp"
        self.VESPA_SCHEMA_NAME = "course_page"
        
        # Read key from file instead of env
        try:
            with open('harkness.harknessapp.pem', 'r') as f:
                vespa_token = f.read().strip()
            print("Token loaded from file:", bool(vespa_token))
            print("\nFull token content:")
            print(vespa_token)
            print("\nToken length:", len(vespa_token))
            self.VESPA_TOKEN = vespa_token
        except Exception as e:
            print(f"Error reading key file: {e}")
            raise
        
        # Initialize Supabase
        self.supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_SERVICE_KEY")
        )
        
        # Initialize Gemini
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.gemini = genai.GenerativeModel("gemini-1.5-pro")
        
        # Initialize ColPali
        self.device = get_torch_device("auto")
        self.model = ColPali.from_pretrained(
            "vidore/colpali-v1.2",
            torch_dtype=torch.float32,
            device_map=self.device
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")
        
        # Initialize Vespa
        self._setup_vespa()

    def _add_rank_profiles(self):
        """Add rank profiles to the schema"""
        # Define similarity functions used in all rank profiles
        mapfunctions = [
            Function(
                name="similarities",
                expression="""
                    sum(
                        query(qt) * unpack_bits(attribute(embedding)), v
                    )
                """,
            ),
            Function(
                name="normalized",
                expression="""
                    (similarities - reduce(similarities, min)) / (reduce((similarities - reduce(similarities, min)), max)) * 2 - 1
                """,
            ),
            Function(
                name="quantized",
                expression="""
                    cell_cast(normalized * 127.999, int8)
                """,
            ),
        ]

        # Define the basic BM25 rank profile
        bm25 = RankProfile(
            name="bm25",
            inputs=[("query(qt)", "tensor<float>(querytoken{}, v[128])")],
            first_phase="bm25(text_content)",
            functions=mapfunctions,
        )
        
        # Add the rank profiles to the schema
        self.schema.add_rank_profile(bm25)
        
        # Add hybrid ranking profile
        hybrid = RankProfile(
            name="hybrid",
            first_phase="bm25(text_content) + nearestNeighbor(embedding)",
            inherits="bm25",
        )
        self.schema.add_rank_profile(hybrid)

    def _setup_vespa(self):
        """Setup Vespa schema and application"""
        self.schema = Schema(
            name=self.VESPA_SCHEMA_NAME,
            document=Document(
                fields=[
                    Field(name="id", type="string", indexing=["summary", "index"]),
                    Field(name="course_code", type="string", indexing=["summary", "index"]),
                    Field(name="professor", type="string", indexing=["summary", "index"]),
                    Field(name="term", type="string", indexing=["summary", "index"]),
                    Field(
                        name="title",
                        type="string",
                        indexing=["summary", "index"],
                        match=["text"],
                        index="enable-bm25",
                    ),
                    Field(name="page_number", type="int", indexing=["summary", "attribute"]),
                    Field(name="thumbnail", type="raw", indexing=["summary"]),
                    Field(name="full_image", type="raw", indexing=["summary"]),
                    Field(
                        name="text_content",
                        type="string",
                        indexing=["summary", "index"],
                        match=["text"],
                        index="enable-bm25",
                    ),
                    Field(
                        name="embedding",
                        type="tensor<int8>(patch{}, v[16])",
                        indexing=["attribute", "index"],
                        ann=HNSW(
                            distance_metric="hamming",
                            max_links_per_node=32,
                            neighbors_to_explore_at_insert=400,
                        ),
                    ),
                ]
            )
        )
        
        # Add rank profiles
        self._add_rank_profiles()
        
        # Create and deploy Vespa application
        self.app_package = ApplicationPackage(
            name=self.VESPA_APP_NAME,
            schema=[self.schema]
        )

        # Initialize Vespa cloud connection
        self.vespa_cloud = VespaCloud(
            tenant=self.VESPA_TENANT_NAME,
            application=self.VESPA_APP_NAME,
            key_content=self.VESPA_TOKEN, 
            application_package=self.app_package
        )
        
        # Deploy and connect
        self.vespa_cloud.deploy()
        self.vespa_app = Vespa(
            url=self.vespa_cloud.get_token_endpoint(),
            vespa_cloud_secret_token=self.VESPA_TOKEN
        )

    async def process_course_materials(self, 
                                     directory_path: str, 
                                     course_metadata: Dict[str, str]) -> Dict[str, Any]:
        """Process all course materials in a directory"""
        try:
            # Find all PDF files
            pdf_files = list(Path(directory_path).glob("**/*.pdf"))
            print(f"Found {len(pdf_files)} PDF files to process")
            
            # Store course metadata in Supabase
            course_data = {
                "course_code": course_metadata["course_code"],
                "professor": course_metadata["professor"],
                "term": course_metadata["term"],
                "title": course_metadata["title"],
                "created_at": datetime.now().isoformat()
            }
            course_result = await self.supabase.table("courses").insert(course_data).execute()
            course_id = course_result.data[0]['id']
            
            # Process files in batches
            results = []
            batch_size = 3  # Process 3 files concurrently
            for i in range(0, len(pdf_files), batch_size):
                batch = pdf_files[i:i + batch_size]
                batch_results = await asyncio.gather(
                    *[self._process_single_file(pdf, course_id, course_metadata) 
                      for pdf in batch]
                )
                results.extend(batch_results)
            
            return {
                "status": "completed",
                "course_id": course_id,
                "total_files": len(pdf_files),
                "processed": len(results)
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _process_single_file(self, 
                                 file_path: Path, 
                                 course_id: str,
                                 course_metadata: Dict[str, str]) -> Dict[str, Any]:
        """Process a single PDF file"""
        try:
            # Convert PDF to images and extract text
            images = convert_from_path(str(file_path))
            reader = PdfReader(str(file_path))
            
            # Store file metadata in Supabase
            file_data = {
                "course_id": course_id,
                "filename": file_path.name,
                "path": str(file_path),
                "page_count": len(images),
                "uploaded_at": datetime.now().isoformat()
            }
            file_result = await self.supabase.table("course_materials").insert(file_data).execute()
            
            # Process each page
            vespa_docs = []
            for page_no, (image, page) in enumerate(zip(images, reader.pages)):
                # Generate visual embeddings
                embedding = await self._generate_embedding(image)
                
                # Extract text
                text_content = page.extract_text()
                
                # Create Vespa document
                doc_id = hashlib.md5(f"{file_path}_{page_no}".encode()).hexdigest()
                doc = {
                    "id": doc_id,
                    "fields": {
                        "course_code": course_metadata["course_code"],
                        "professor": course_metadata["professor"],
                        "term": course_metadata["term"],
                        "title": str(file_path.stem),
                        "page_number": page_no,
                        "text_content": text_content,
                        "embedding": self._float_to_binary_embedding(embedding),
                        "thumbnail": self._create_thumbnail(image),
                        "full_image": self._process_image_for_vespa(image)
                    }
                }
                vespa_docs.append(doc)
            
            # Feed documents to Vespa
            await self._feed_vespa_batch(vespa_docs)
            
            return {
                "status": "success",
                "file": str(file_path),
                "pages_processed": len(vespa_docs)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "file": str(file_path),
                "error": str(e)
            }

    async def query(self, 
                   student_query: str,
                   course_code: Optional[str] = None) -> Dict[str, Any]:
        """Process student query with visual and conversational capabilities"""
        try:
            # Generate query embedding
            query_embedding = await self._generate_query_embedding(student_query)
            
            # Search Vespa
            response = self.vespa_app.query(
                body={
                    "yql": """
                    select * from course_page where {
                        rank: nearestNeighbor(embedding, query_embedding) or
                        match(text_content contains query_text)
                    }
                    """,
                    "ranking": "hybrid",
                    "input.query_embedding": query_embedding,
                    "input.query_text": student_query,
                    "hits": 3
                }
            )
            
            # Generate educational response using Gemini
            context = self._extract_context(response.hits)
            educational_response = await self._generate_educational_response(
                student_query, context
            )
            
            return {
                "answer": educational_response,
                "relevant_pages": [
                    {
                        "title": hit.fields.get("title"),
                        "page_number": hit.fields.get("page_number"),
                        "thumbnail": hit.fields.get("thumbnail")
                    }
                    for hit in response.hits
                ]
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _generate_educational_response(self, query: str, context: str) -> str:
        """Generate educational response using Gemini"""
        prompt = f"""You are a helpful teaching assistant. Using the following course material context, 
        guide the student's learning without directly giving the answer.
        
        Context: {context}
        Student Question: {query}
        
        Provide a response that:
        1. Helps the student understand the key concepts
        2. Guides them toward finding the answer through critical thinking
        3. Suggests relevant topics to review
        4. Asks thought-provoking follow-up questions
        
        Response should facilitate learning, not just provide the answer."""
        
        response = self.gemini.generate_content(prompt)
        return response.text

    # Utility methods (embedding generation, image processing, etc.)
    def _generate_embedding(self, image) -> np.ndarray:
        inputs = self.processor.process_images([image]).to(self.device)
        with torch.no_grad():
            return self.model(**inputs)[0].cpu().numpy()

    @staticmethod
    def _float_to_binary_embedding(embedding: np.ndarray) -> Dict[str, List[int]]:
        binary_vector = np.packbits(np.where(embedding > 0, 1, 0)).astype(np.int8).tolist()
        return {str(i): v for i, v in enumerate(binary_vector)}

    @staticmethod
    def _create_thumbnail(image, max_size=(200, 200)):
        thumb = image.copy()
        thumb.thumbnail(max_size)
        return EducationalVisualRAG._process_image_for_vespa(thumb)

    @staticmethod
    def _process_image_for_vespa(image) -> str:
        import base64
        from io import BytesIO
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85, optimize=True)
        return base64.b64encode(buffered.getvalue()).decode()

# Usage Example
if __name__ == "__main__":
    # Initialize the system
    rag = EducationalVisualRAG()
    
    # Course metadata
    course_metadata = {
        "course_code": "CS101",
        "professor": "Dr. Smith",
        "term": "Spring 2024",
        "title": "Introduction to Computer Science"
    }
    
    # Process course materials
    asyncio.run(rag.process_course_materials(
        directory_path="/Users/miki/Desktop/langchain_rag/data",
        course_metadata=course_metadata
    ))
    
    # Query example
    result = asyncio.run(rag.query(
        "Can you explain the diagram showing how binary search works?",
        course_code="CS101"
    ))

    


