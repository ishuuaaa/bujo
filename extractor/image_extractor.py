
import pytesseract
from PIL import Image

def extract_text_from_image(filepath):
    img = Image.open(filepath)
    return pytesseract.image_to_string(img)
