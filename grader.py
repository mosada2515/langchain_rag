import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json
import logging
from queue import Queue
from datetime import datetime
from openai import OpenAI
from google.cloud import documentai_v1 as documentai
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    document_id: str
    success: bool
    needs_human_review: bool
    extracted_text: Optional[Dict] = None
    error_message: Optional[str] = None
    confidence_scores: Optional[Dict] = None

@dataclass
class GradingResult:
    document_id: str
    score: float
    feedback: str
    grader_type: str  # 'ai' or 'human'
    confidence: float
    timestamp: str

class BatchDocumentProcessor:
    def __init__(
        self,
        project_id: str,
        location: str,
        processor_id: str,
        confidence_threshold: float = 0.8,
        batch_size: int = 10,
        max_workers: int = 4
    ):
        load_dotenv()
        self.project_id = project_id
        self.location = location
        self.processor_id = processor_id
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.human_review_queue = Queue()
        
        # Initialize Document AI client
        self.doc_client = documentai.DocumentProcessorServiceClient()
        self.processor_name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Load rubric and sample grades
        self.rubric = self._load_rubric()
        self.sample_grades = self._load_sample_grades()

    def _load_rubric(self) -> Dict:
        """Load grading rubric from file"""
        try:
            with open("grading_rubric.json", "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading rubric: {e}")
            return {}

    def _load_sample_grades(self) -> List[Dict]:
        """Load sample grades from directory"""
        samples_dir = Path("sample_grades")
        samples = []
        try:
            for file in samples_dir.glob("*.json"):
                with open(file, "r") as f:
                    samples.append(json.load(f))
            return samples
        except Exception as e:
            logger.error(f"Error loading sample grades: {e}")
            return []

    def process_single_document(self, file_path: str) -> ProcessingResult:
        """Process a single document and determine if it needs human review"""
        try:
            with open(file_path, "rb") as image:
                image_content = image.read()

            document = {"content": image_content, "mime_type": "image/jpeg"}
            request = {"name": self.processor_name, "raw_document": document}
            response = self.doc_client.process_document(request=request)

            # Track confidence scores
            confidence_scores = {
                'word_level': [],
                'paragraph_level': []
            }

            extracted_data = {'pages': []}
            needs_human_review = False

            for page in response.document.pages:
                page_data = self._process_page(page, confidence_scores)
                extracted_data['pages'].append(page_data)

            # Determine if human review is needed
            avg_word_confidence = sum(confidence_scores['word_level']) / len(confidence_scores['word_level']) if confidence_scores['word_level'] else 0
            needs_human_review = avg_word_confidence < self.confidence_threshold

            return ProcessingResult(
                document_id=Path(file_path).stem,
                success=True,
                needs_human_review=needs_human_review,
                extracted_text=extracted_data,
                confidence_scores=confidence_scores
            )

        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return ProcessingResult(
                document_id=Path(file_path).stem,
                success=False,
                needs_human_review=True,
                error_message=str(e)
            )

    def _process_page(self, page, confidence_scores: Dict) -> Dict:
        """Process a single page and track confidence scores"""
        page_data = {'blocks': []}
        
        for block in page.blocks:
            block_data = {'paragraphs': []}
            
            for paragraph in block.paragraphs:
                para_data = self._process_paragraph(paragraph, confidence_scores)
                block_data['paragraphs'].append(para_data)
                
            page_data['blocks'].append(block_data)
            
        return page_data

    def _process_paragraph(self, paragraph, confidence_scores: Dict) -> Dict:
        """Process a single paragraph and track confidence scores"""
        para_data = {
            'words': [],
            'text': '',
            'confidence': 0
        }
        
        word_texts = []
        para_confidence = 0
        
        for word in paragraph.words:
            word_data = self._process_word(word)
            para_data['words'].append(word_data)
            word_texts.append(word_data['text'])
            confidence_scores['word_level'].append(word_data['confidence'])
            para_confidence += word_data['confidence']
        
        para_data['text'] = ' '.join(word_texts)
        para_data['confidence'] = para_confidence / len(paragraph.words) if paragraph.words else 0
        confidence_scores['paragraph_level'].append(para_data['confidence'])
        
        return para_data

    def _process_word(self, word) -> Dict:
        """Process a single word and return its data"""
        return {
            'text': word.text,
            'confidence': word.layout.confidence if word.layout.confidence else 0,
            'symbols': [symbol.text for symbol in word.symbols]
        }

    def grade_document(self, processed_result: ProcessingResult) -> GradingResult:
        """Grade a processed document using OpenAI"""
        if processed_result.needs_human_review:
            self.human_review_queue.put(processed_result)
            return GradingResult(
                document_id=processed_result.document_id,
                score=0.0,
                feedback="Queued for human review due to low confidence scores",
                grader_type="pending_human",
                confidence=0.0,
                timestamp=datetime.now().isoformat()
            )

        try:
            prompt = self._create_grading_prompt(processed_result.extracted_text)
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert grader. Grade the following work according to the rubric and samples provided."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            # Parse the grading response
            grading_response = json.loads(response.choices[0].message.content)
            
            return GradingResult(
                document_id=processed_result.document_id,
                score=grading_response['score'],
                feedback=grading_response['feedback'],
                grader_type='ai',
                confidence=grading_response.get('confidence', 0.9),
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(f"Error grading document {processed_result.document_id}: {e}")
            self.human_review_queue.put(processed_result)
            return GradingResult(
                document_id=processed_result.document_id,
                score=0.0,
                feedback=f"Error during AI grading: {str(e)}. Queued for human review.",
                grader_type="pending_human",
                confidence=0.0,
                timestamp=datetime.now().isoformat()
            )

    def _create_grading_prompt(self, extracted_text: Dict) -> str:
        """Create a detailed prompt for the AI grader"""
        return f"""
        Grade the following work according to this rubric:
        {json.dumps(self.rubric, indent=2)}

        Here are some sample grades for reference:
        {json.dumps(self.sample_grades, indent=2)}

        Work to grade:
        {json.dumps(extracted_text, indent=2)}

        Provide a JSON response with the following structure:
        {{
            "score": float,
            "feedback": string,
            "confidence": float,
            "reasoning": string
        }}
        """

    def process_batch(self, input_directory: str) -> Tuple[List[GradingResult], List[ProcessingResult]]:
        """Process a batch of documents in parallel"""
        input_files = list(Path(input_directory).glob("*.jpg"))
        processed_results = []
        grading_results = []
        
        # Process documents in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(self.process_single_document, str(file)): file 
                            for file in input_files[:self.batch_size]}
            
            for future in future_to_file:
                try:
                    result = future.result()
                    processed_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")

        # Grade processed documents
        for proc_result in processed_results:
            if proc_result.success:
                grade_result = self.grade_document(proc_result)
                grading_results.append(grade_result)

        return grading_results, processed_results

def main():
    # Initialize processor
    processor = BatchDocumentProcessor(
        project_id="your-project-id",
        location="us",
        processor_id="your-processor-id",
        confidence_threshold=0.8,
        batch_size=100,
        max_workers=4
    )

    # Process batch
    grading_results, processing_results = processor.process_batch("input_documents")

    # Save results
    with open("grading_results.json", "w") as f:
        json.dump([vars(result) for result in grading_results], f, indent=2)

    # Print summary
    print(f"Processed {len(processing_results)} documents")
    print(f"Documents needing human review: {processor.human_review_queue.qsize()}")

if __name__ == "__main__":
    main()




