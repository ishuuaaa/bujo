
import os
import json
import pickle
from extractor.pdf_extractor import extract_text_from_pdf
from extractor.image_extractor import extract_text_from_image
from extractor.text_extractor import extract_text_from_txt

model = pickle.load(open("classifier/model.pkl", "rb"))

PDF = [".pdf"]
IMG = [".jpg", ".png", ".jpeg"]
TXT = [".txt"]

OUTPUT_DIR = "organized_notes/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

metadata = {}

def classify_text(text):
    return model.predict([text])[0]

def organize_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()

    if ext in PDF:
        text = extract_text_from_pdf(filepath)
    elif ext in IMG:
        text = extract_text_from_image(filepath)
    elif ext in TXT:
        text = extract_text_from_txt(filepath)
    else:
        print(f"Unsupported file: {filepath}")
        return

    topic = classify_text(text)
    topic_folder = os.path.join(OUTPUT_DIR, topic)
    os.makedirs(topic_folder, exist_ok=True)

    dest_path = os.path.join(topic_folder, os.path.basename(filepath))
    os.rename(filepath, dest_path)

    metadata[os.path.basename(filepath)] = {
        "topic": topic,
        "path": dest_path
    }

    print(f"✔ Organized → {filepath} → {topic}")

def main():
    folder = input("Enter folder to organize: ")

    files = os.listdir(folder)
    for f in files:
        organize_file(os.path.join(folder, f))

    with open("organized_notes/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("🎉 Notes organized successfully!")

if __name__ == "__main__":
    main()
