import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# Configuration Variables
MODEL_NAME = "Llama-2-7b-chat-hf"  # The base model to fine-tune
TRAINING_DATA_FILE = "training_data.json"  # Path to the training dataset
OUTPUT_DIR = "./fine_tuned_model"  # Output directory for the fine-tuned model
LOGGING_DIR = "./logs"  # Directory to store logs
BATCH_SIZE = 2  # Batch size during training
EPOCHS = 3  # Number of training epochs
MAX_LENGTH = 512  # Max token length for padding/truncation
SAVE_STEPS = 1000  # Number of steps between saving model checkpoints
SAVE_TOTAL_LIMIT = 2  # Total number of saved checkpoints to keep

# Load Base Model and Tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
)

# Define LoRA Configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Adapt these depending on your model's layers
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM
)

# Apply LoRA Adapters to the model
print("Applying LoRA adapters...")
model = get_peft_model(model, lora_config)

# Function to load dataset from JSON file
def load_dataset(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return [{"prompt": d["prompt"], "response": d["response"]} for d in data]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

# Function to tokenize the dataset
def tokenize_data(examples):
    inputs = tokenizer(examples["prompt"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    targets = tokenizer(examples["response"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    return {"input_ids": inputs["input_ids"], "labels": targets["input_ids"]}

# Load and preprocess dataset
print(f"Loading dataset from {TRAINING_DATA_FILE}...")
dataset = load_dataset(TRAINING_DATA_FILE)
if not dataset:
    raise ValueError("The dataset is empty or failed to load.")

# Convert list to HuggingFace Dataset and tokenize
dataset = Dataset.from_list(dataset)
print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_data, batched=True, remove_columns=["prompt", "response"])

# Training Configuration
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    logging_dir=LOGGING_DIR,
    logging_steps=10,
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    evaluation_strategy="steps",  # Add evaluation strategy if required
    eval_steps=500,  # Evaluate every 500 steps (optional)
    load_best_model_at_end=True,  # Option to load the best model after training
    fp16=True,  # Enable mixed precision training
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Train the model
print("Starting model training...")
trainer.train()

# Save the fine-tuned model and tokenizer
print(f"Saving the fine-tuned model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Fine-tuned model with LoRA and dataset saved successfully!")
