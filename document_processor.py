import PyPDF2
import os
from typing import List, Tuple

class DocumentProcessor:
    def __init__(self):
        self.supported_formats = ['.pdf']
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract metadata
                metadata = {
                    'pages': len(pdf_reader.pages),
                    'file_name': os.path.basename(pdf_path)
                }
                
                # Extract text from each page
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n[Page {page_num + 1}]\n{page_text}"
                
                return text, metadata
                
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return "", {}
    
    def process_papers_folder(self, folder_path: str) -> List[Tuple[str, str, dict]]:
        """Process all PDFs in a folder"""
        papers = []
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")
            return papers
        
        for filename in os.listdir(folder_path):
            if filename.endswith('.pdf'):
                filepath = os.path.join(folder_path, filename)
                print(f"Processing: {filename}")
                
                text, metadata = self.extract_text_from_pdf(filepath)
                
                if text:
                    papers.append((filename, text, metadata))
                    print(f"✅ Extracted {len(text)} characters")
                else:
                    print(f"❌ Failed to extract text from {filename}")
        
        return papers
