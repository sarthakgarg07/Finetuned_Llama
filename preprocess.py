import re
import json

def clean_text(text):
    """Remove special characters and extra spaces."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def load_dataset(file_path):
    """Load and clean dataset for fine-tuning."""
    with open(file_path, "r") as f:
        data = json.load(f)
    
    cleaned_data = [{"prompt": clean_text(d["prompt"]), "response": clean_text(d["response"])} for d in data]
    return cleaned_data

if __name__ == "__main__":
    dataset = load_dataset("training_data.json")
    print(f"Loaded {len(dataset)} cleaned training samples.")
