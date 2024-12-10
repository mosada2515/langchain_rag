import os
from typing import List, Set, Dict, Optional
from openai import OpenAI
from dataclasses import dataclass
import asyncio
import json
from dotenv import load_dotenv
from logging import getLogger, INFO, StreamHandler
from PyPDF2 import PdfReader
from pathlib import Path

# Set up logging
logger = getLogger(__name__)
logger.setLevel(INFO)
logger.addHandler(StreamHandler())


"""Store this all in supabase"""

load_dotenv()

@dataclass
class Topic:
    name: str
    

class SyllabusProcessor:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """Initialize with OpenAI"""
        self.model = model
            
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.openai = OpenAI(api_key=api_key)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""

    async def process_syllabus(self, text: str) -> List[Topic]:
        """Process syllabus text through OpenAI"""
        if not text:
            return []
        
        try:
            prompt = f"""List the course topics in order that they appear in the syllabus. 
Return ONLY a JSON array of strings, for example: ["Topic 1", "Topic 2", "Topic 3"]

Syllabus text:
{text}"""

            response = self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at identifying topics from syllabi. Return only a JSON array of topic strings."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            result_text = response.choices[0].message.content.strip()
            # Remove any markdown formatting that might be present
            result_text = result_text.replace('```json', '').replace('```', '').strip()
            logger.info(f"OpenAI response: {result_text}")
            
            try:
                topics_list = json.loads(result_text)
                if not isinstance(topics_list, list):
                    logger.error(f"Expected list but got {type(topics_list)}")
                    return []
                
                topics = [Topic(name=topic) for topic in topics_list if isinstance(topic, str)]
                logger.info(f"Successfully parsed {len(topics)} topics")
                return topics
                
            except json.JSONDecodeError as je:
                logger.error(f"JSON parsing error: {je}")
                logger.error(f"Failed to parse: {result_text}")
                return []
                
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            return []

    

async def main():
    """Main function with improved error handling and user interaction"""
    try:
        processor = SyllabusProcessor()
        
        data_folder = Path(__file__).parent / "data_syllabus"
        if not data_folder.exists():
            logger.info("Creating data_syllabus folder...")
            data_folder.mkdir(exist_ok=True)
            logger.info("Please place PDF files in the data_syllabus folder and run again.")
            return

        pdf_files = list(data_folder.glob("*.pdf"))
        if not pdf_files:
            logger.info("No PDF files found in data_syllabus folder.")
            return

        print("\nAvailable PDF files:")
        for i, file_path in enumerate(pdf_files, 1):
            print(f"{i}. {file_path.name}")

        while True:
            try:
                choice = int(input("\nEnter the number of the file to process (or 0 to exit): "))
                if choice == 0:
                    return
                if 1 <= choice <= len(pdf_files):
                    break
                print("Invalid choice. Please select a valid number.")
            except ValueError:
                print("Please enter a valid number.")

        selected_file = pdf_files[choice - 1]
        logger.info(f"Processing {selected_file.name}...")

        # Extract text
        text = processor.extract_text_from_pdf(str(selected_file))
        if not text:
            logger.error("Could not extract text from PDF.")
            return
        logger.info(f"Extracted text length: {len(text)}")
        logger.info(f"First 500 characters: {text[:500]}")

        # Process syllabus
        topics = await processor.process_syllabus(text)

        if topics:
            print(f"\nFound {len(topics)} verified topics:")
            for i, topic in enumerate(topics, 1):
                print(f"{i}. {topic.name}")
            
            # Offer to save results
            save = input("\nWould you like to save these results? (y/n): ").lower()
            if save == 'y':
                output_file = selected_file.with_suffix('.topics.json')
                with open(output_file, 'w') as f:
                    json.dump([{"name": t.name} for t in topics], f, indent=2)
                print(f"Results saved to {output_file}")
        else:
            logger.info("No verified topics found in the document.")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
if __name__ == "__main__":
    asyncio.run(main())