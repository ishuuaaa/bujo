
import pypdf

def extract_text_from_pdf(filepath):
    reader = pypdf.PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text
