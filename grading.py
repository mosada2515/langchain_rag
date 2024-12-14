from google.cloud import documentai_v1 as documentai 
from google.cloud import storage 
from openai import OpenAI
from typing import List, Dict, Optional, Any 
from dataclasses import dataclass
from pathlib import Path
import asyncio
import json 
import logging
from datetime import datetime 
from concurrent.futures import ThreadPoolExecutor
import os 
from dotenv import load_dotenv
from queue import Queue
from tqdm import tqdm
from validation import validate_primary_grade_response, validate_review_grade_response, ValidationError

#configure logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

@dataclass 
class ProcessedDocument: 
    """Represents a document processed by DocumentAI"""
    document_id: str
    extracted_text: str
    confidence_score: float
    needs_review: bool
    page_count: int
    error: Optional[str] = None

@dataclass
class GradingResult: 
    """Represents the final grading result for a document"""
    document_id: str
    score: float
    feedback: str
    grader_type: str #ai or human 
    confidence: float
    needs_review: bool 
    error: Optional[str] = None 


class GradingTool: 
    """Base class for grading tools"""
    def __init__(self, name: str):
        self.name = name

    async def execute(self, input_data: Dict) -> Dict:
        """Execute the grading tool"""
        raise NotImplementedError
        
class RubricMatcher(GradingTool):
    """Tool for matching answers against rubric criteria"""
    def __init__(self):
        super().__init__("rubric_matcher")

    async def execute(self, input_data: Dict) -> Dict:
        """Excetute the rubric matching logic"""
        answer = input_data.get("answer", "")
        criteria = input_data.get("criteria", {})

        results = {
            'matches': {},
            'total_points': 0,
            'feedback': []
        }

        #Make this better later
        for criterion, points in criteria.items():
            #Simple keyword matching
            if any(key.lower() in answer.lower() for key in criterion.split()):
                results['matches'][criterion] = points
                results['total_points'] += points
                

        return results

class QualityAnalyzer(GradingTool):
    """Tool for analyzing the answer quality"""

    def __init__(self):
        super().__init__("quality_analyzer")

    async def execute(self, input_data: Dict) -> Dict:
        text = input_data.get("text", "")

        metrics = {
            'word_count' : len(text.split()),
            'sentence_count': text.count('.') + text.count('!') + text.count('?'),
            'completeness': 1.0 if len(text) > 100 else len(text) / 100
        }

        return metrics
    
class GradingAgent:
    """AI agent for grading documents"""
    def __init__(
            self, 
            role: str, 
            openai_client: OpenAI, 
            model: str = "gpt-4", 
            tools: Optional[List[GradingTool]] = None, 
    ): 
        self.role = role
        self.openai = openai_client
        self.model = model
        self.tools = tools or []
        self.grading_history = []

    async def grade_document(
            self,
            text: str,
            rubric: Dict,
            context: Dict
    ) -> Dict: 
        """Grade a document using AI and tools"""
        try: 
            #Initial grading prompt
            grading_prompt = self._create_grading_prompt(text, rubric, context)
            initial_grade = await self._get_completion(grading_prompt)

            #check for tools needed (here we need to find the actual API call)
            tool_results = {}
            needed_tools = initial_grade.get('needed_tools', [])
            if needed_tools:
                for tool in self.tools:
                    if tool.name in needed_tools:
                        result = await tool.execute({
                            'text': text,
                            'criteria': rubric,
                            'context': context
                        })
                        tool_results[tool.name] = result

            # If tool results exist, refine grading
            if tool_results:
                tool_review_prompt = self._create_tool_review_prompt(initial_grade, tool_results)
                final_grade = await self._get_completion(tool_review_prompt)
            else:
                final_grade = initial_grade

            # Update history
            self.grading_history.append({
                'timestamp': datetime.now().isoformat(),
                'context': context,
                'grade': final_grade
            })
            
            return final_grade
            
        except Exception as e:
            logger.error(f"Error in grading: {str(e)}")
            return {'error': str(e)}
    
    async def _get_completion(self, prompt: str) -> Dict: 
        """Get completion from OpenAI and validate JSON reponse."""

        try: 
            response = await self.openai.chat.completions.create(
                model=self.model, 
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ], 
                temperature=0.1
            )
            #Parse the JSON
            raw_text = response.choices[0].message.content
            #Attempt to load JSON
            try: 
                data = json.loads(raw_text)
            except json.JSONDecodeError:
                logger.error("AI did no return valid JSON")
                return{} 
            
            #Validatie based on role
            if self.role == "primary_grader": 
                validate_primary_grade_response(data)
            elif self.role == "review_grader": 
                validate_review_grade_response(data)


            return data 
        except ValidationError as ve:
            logger.error(f"Validation error in {self.role} response: {ve}")
            return {}
        except Exception as e: 
            logger.error(f"Error getting completion: {e}")
            return {}
        
    def _get_system_prompt(self) -> str:
        """Get role-specific system prompt"""
        if self.role == "primary_grader":
            return """You are an expert grader focused on: 
-Accurate evaluation against rubrics
-Detailed, constructive feedback
-consistent grading standards
-Identifying key concepts and misconceptions
Your ouput should be a JSON object with fields:
{
    "score": float,
    "feedback": str,
    "key_concepts": List[str],
    "needed_tools":[str(optional)]
}
"""
        elif self.role == "review_grader":
            return """You are a reviewing grader focused on:
- Verifying initial grades
- Identifying potential issues
- Suggesting adjustments
- Maintaining consistency
Your output should be JSON with fields for:
{
  "verified_score": float,
  "adjustments": str,
  "confidence": float,
  "needs_human_review": bool
}"""

        return ""
    def _create_grading_prompt(self, text: str, rubric: Dict, context: Dict) -> str: 
        return f"""Grade this response using the provided rubric:
Text:
{text}

Rubric: 
{json.dumps(rubric, indent=2)}

Context: 
{json.dumps(context, indent=2)}

Provide a detailed evaluation including: 
1. Numerical score based on the available points
2. Specific feedback
3. Key concepts identified
4. Tools need for additional analysis(if any)"""
    
    def _create_tool_review_prompt(self, initial_grade: Dict, tool_results: Dict) -> str:
        return f"""Review and finalize this grade considering the following tool results: 

Initial Grade:
{json.dumps(initial_grade, indent=2)}

Tool Results:
{json.dumps(tool_results, indent=2)}

Incorporate tool insights and provide a final grade.json
"""
class BatchGradingSystem: 
    """System for batch processing and grading documents"""
    def __init__(
            self,
            project_id: str,
            processor_id: str, 
            location: str = "us", 
            confidence_threshold: float = 0.8, 
            batch_size: int = 20,
            max_workers: int =4
            
    ):
        load_dotenv()

        #Initialize document AI
        self.doc_client = documentai.DocumentProcessorServiceClient()
        self.processor_name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"

        #initialize OpenAi
        self.openai = OpenAI()

        #Initialize grading agents 
        self.primary_grader = GradingAgent(
            "primary_grader",
            self.openai,
            tools=[RubricMatcher()]
    
        )

        self.review_grader = GradingAgent(
            "review_grader",
            self.openai,
            tools=[QualityAnalyzer()]
        )

        #configuration 
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.max_workers = max_workers

        self.review_queue = Queue()
        self.results ={
            'completed': [],
            'needs_review': [],
            'errors': []

        }
        
    async def process_document(self, file_path: str) -> ProcessedDocument: 
        """Process a single document with DocumentAI"""
        try: 
            #read document
            with open(file_path, 'rb') as image: 
                image_content = image.read()

            #process with DocumentAI
            request = documentai.ProcessRequest(
                name=self.processor_name,
                raw_document=documentai.RawDocument(
                    content=image_content,
                    mime_type="application/pdf" #TODO: make this dynamic and be able to store it in supabase
                )
            )
            result = self.doc_client.process_document(request=request)
            document = result.document



            # Calculate confidence
            block_confidences = []
            for page in document.pages:
                for block in page.blocks:
                    if block.layout.confidence is not None:
                        block_confidences.append(block.layout.confidence)
            
            confidence = sum(block_confidences) / len(block_confidences) if block_confidences else 0.0

            return ProcessedDocument(
                document_id=Path(file_path).stem,
                extracted_text=document.text,
                confidence_score=confidence,
                needs_review=confidence < self.confidence_threshold,
                page_count=len(document.pages)
            )

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return ProcessedDocument(
                document_id=Path(file_path).stem,
                extracted_text="",
                confidence_score=0.0,
                needs_review=True,
                page_count=0,
                error=str(e)
            )

         








       

            


    
