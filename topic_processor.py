import os
from typing import List, Dict, Optional
from dataclasses import dataclass
from openai import OpenAI
import asyncio
import json
import PyPDF2
from dotenv import load_dotenv
from logging import getLogger, INFO, StreamHandler
from pathlib import Path

# Set up logging
logger = getLogger(__name__)
logger.setLevel(INFO)
logger.addHandler(StreamHandler())

load_dotenv()

@dataclass
class Question:
    text: str
    question_type: str  # homework, quiz, test
    topic: str
    is_problem: bool = True 
    subtopic: Optional[str] = None
    difficulty: Optional[float] = None
    points: Optional[float] = None
    assessment_name: str = ""

@dataclass
class Assessment:
    name: str
    type: str  # homework, quiz, test
    questions: List[Question]
    total_points: float = 0.0
    due_date: Optional[str] = None

class AcademicQuestionProcessor:
    def __init__(self, topics: List[str], model: str = "gpt-3.5-turbo"):
        """Initialize with OpenAI and course topics"""
        self.topics = topics
        self.model = model
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.openai = OpenAI(api_key=api_key)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file"""
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""

    async def extract_questions(self, text: str) -> List[Question]:
        """Extract questions from document text"""
        prompt = f"""Given this document text, first identify only the actual problems/questions that require answers or solutions.
Ignore:
- General instructions
- Review notes
- Section headers
- Other non-question text

For each identified question:
1. Extract the complete question text
2. Determine if it's a real problem requiring solution (vs rhetorical/discussion question)
3. Identify type (homework/quiz/test)
4. Identify subtopic (if applicable)
5. Make sure you get the subquestion as well(Questions 1.a, 1.b, etc)
6. Match to most relevant topic from: {self.topics}
7. Estimate difficulty (0-1 scale)
8. Extract points if specified

ONLY return questions that are actual problems requiring solutions.

Return as JSON array:
{{
    "questions": [
        {{
            "question": "complete question text",
            "is_problem": true,
            "type": "homework/quiz/test", 
            "topic": "matching topic",
            "points": 5,
            "difficulty": 0.7
        }}
    ]
}}

Document text:
{text}"""

        try:
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing academic questions and identifying their core learning objectives. You excel at matching questions to their most relevant course topics based on the fundamental concepts being tested."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000,
                response_format={ "type": "json_object" }
            )
            
            response_content = response.choices[0].message.content
            logger.info(f"Raw OpenAI response: {response_content}")  # Debug logging
            
            try:
                result = json.loads(response_content)
                # Ensure we're working with a list of questions
                questions_list = result.get("questions", []) if isinstance(result, dict) else result
                
                return [
                    Question(
                        text=q.get("question", ""),
                        question_type=q.get("type", "homework"),
                        topic=q.get("topic", self.topics[0]),  # Default to first topic if none specified
                        points=float(q.get("points", 0)) if q.get("points") is not None else None,
                        difficulty=float(q.get("difficulty", 0.5)) if q.get("difficulty") is not None else None,
                        assessment_name=q.get("assessment_name", "")
                    )
                    for q in questions_list
                    if q.get("question") and q.get("is_problem", False)  # Only include actual problems with text
                ]
                
            except json.JSONDecodeError as je:
                logger.error(f"JSON parsing error: {je}")
                logger.error(f"Failed to parse response: {response_content}")
                return []
                
        except Exception as e:
            logger.error(f"Error extracting questions: {str(e)}")
            logger.error(f"Full error details: ", exc_info=True)
            return []
    async def categorize_question(self, question: str) -> Dict:
        """Categorize a single question by topic"""
        prompt = f"""Given this question and list of course topics, determine:
1. The most relevant topic
2. Estimated difficulty (0-1)
3. Question type (conceptual, computational, analytical)

Topics: {self.topics}

Question: {question}"""

        try:
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at categorizing academic questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error categorizing question: {str(e)}")
            return {}

    def save_assessment(self, assessment: Assessment, output_dir: Path):
        """Save processed assessment to JSON"""
        output_file = output_dir / f"{assessment.name}_{assessment.type}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                "name": assessment.name,
                "type": assessment.type,
                "total_points": assessment.total_points,
                "due_date": assessment.due_date,
                "questions": [
                    {
                        "text": q.text,
                        "topic": q.topic,
                        "difficulty": q.difficulty,
                        "points": q.points
                    }
                    for q in assessment.questions
                ]
            }, f, indent=2)

async def get_topics_from_json(syllabus_file: Path) -> List[str]:
    """Read topics from the previously generated JSON file"""
    json_file = syllabus_file.with_suffix('.topics.json')
    if not json_file.exists():
        logger.error(f"Topics file not found: {json_file}")
        return []
        
    with open(json_file) as f:
        data = json.load(f)
        return [topic["name"] for topic in data]

async def main():
    # First get topics from existing JSON
    syllabus_dir = Path("data_syllabus")
    syllabus_files = list(syllabus_dir.glob("*.pdf"))
    
    if not syllabus_files:
        logger.error("No syllabus files found")
        return
        
    print("\nAvailable syllabus files:")
    for i, file_path in enumerate(syllabus_files, 1):
        print(f"{i}. {file_path.name}")
        
    while True:
        try:
            choice = int(input("\nEnter the number of the syllabus file to process (or 0 to exit): "))
            if choice == 0:
                return
            if 1 <= choice <= len(syllabus_files):
                break
            print("Invalid choice. Please select a valid number.")
        except ValueError:
            print("Please enter a valid number.")
    
    # User selects syllabus    
    selected_syllabus = syllabus_files[choice - 1]
    
    # Get topics from corresponding JSON
    topics = await get_topics_from_json(selected_syllabus)
    
    if not topics:
        logger.error("No topics found in JSON file") 
        return
    
    print("\nExtracted topics:")
    for i, topic in enumerate(topics, 1):
        print(f"{i}. {topic}")
    
    # Initialize question processor with topics
    processor = AcademicQuestionProcessor(topics)
    
    # Process graded materials
    graded_dir = Path("data_graded")
    if not graded_dir.exists():
        logger.error("data_graded directory not found")
        return
        
    files = list(graded_dir.glob("*.txt")) + list(graded_dir.glob("*.pdf"))
    
    if not files:
        logger.error("No files found in data_graded directory")
        return
        
    print("\nAvailable files to process:")
    for i, file_path in enumerate(files, 1):
        print(f"{i}. {file_path.name}")
        
    while True:
        try:
            choice = int(input("\nEnter the number of the file to process (or 0 to exit): "))
            if choice == 0:
                return
            if 1 <= choice <= len(files):
                break
            print("Invalid choice. Please select a valid number.")
        except ValueError:
            print("Please enter a valid number.")
    
    selected_file = files[choice - 1]
    logger.info(f"Processing {selected_file.name}...")
    
    # Extract text based on file type
    text = ""
    if selected_file.suffix == ".pdf":
        text = processor.extract_text_from_pdf(str(selected_file))
    else:
        with open(selected_file) as f:
            text = f.read()
    
    # Process the questions
    questions = await processor.extract_questions(text)
    
    if questions:
        assessment = Assessment(
            name=selected_file.stem,
            type="graded",  # You might want to detect this from filename or content
            questions=questions,
            total_points=sum(q.points or 0 for q in questions)
        )
        
        # Create output directory
        output_dir = Path("processed_graded")
        output_dir.mkdir(exist_ok=True)
        
        processor.save_assessment(assessment, output_dir)
        logger.info(f"Saved processed assessment to {output_dir}")
        
        # Print categorized questions
        print("\nCategorized Questions:")
        for i, q in enumerate(questions, 1):
            print(f"\n{i}. Topic: {q.topic}")
            print(f"   Difficulty: {q.difficulty}")
            print(f"   Points: {q.points}")
            print(f"   Question: {q.text[:100]}...")  # Show first 100 chars
if __name__ == "__main__":
    asyncio.run(main())


