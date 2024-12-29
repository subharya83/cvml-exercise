import os
import hashlib
from bs4 import BeautifulSoup
import PyPDF2

def compute_hash(content):
    """Compute SHA-256 hash of the given content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def extract_text_from_html(file_path):
    """Extract plain text from an HTML file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            return soup.get_text(separator=' ')
    except Exception as e:
        print(f"Error extracting HTML: {file_path} - {e}")
        return ""

def extract_text_from_pdf(file_path):
    """Extract plain text from a PDF file."""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
    except Exception as e:
        print(f"Error extracting PDF: {file_path} - {e}")
        return ""

def extract_text(file_path):
    """Extract plain text from a file based on its extension."""
    _, ext = os.path.splitext(file_path.lower())
    if ext in ['.txt']:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext in ['.html', '.htm']:
        return extract_text_from_html(file_path)
    elif ext in ['.pdf']:
        return extract_text_from_pdf(file_path)
    else:
        print(f"Unsupported file type: {file_path}")
        return ""

def find_and_remove_duplicates(directory):
    """Find and remove duplicate files based on their content."""
    hashes = {}
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            text_content = extract_text(file_path)
            if not text_content:
                continue
            file_hash = compute_hash(text_content)
            if file_hash in hashes:
                print(f"Duplicate found: {file_path} (original: {hashes[file_hash]})")
                os.remove(file_path)
            else:
                hashes[file_hash] = file_path

if __name__ == "__main__":
    directory = "/path/to/your/directory"
    find_and_remove_duplicates(directory)
