import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import torch

MODEL_NAME = "/Llama-2-7b-chat-hf"

# Load Base Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
)

# Define LoRA Configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM
)

# Apply LoRA Adapters
model = get_peft_model(model, lora_config)

# Load and Prepare Training Data
def load_dataset(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return [{"prompt": d["prompt"], "response": d["response"]} for d in data]

def tokenize_data(examples):
    inputs = tokenizer(examples["prompt"], padding="max_length", truncation=True, max_length=512)
    targets = tokenizer(examples["response"], padding="max_length", truncation=True, max_length=512)
    return {"input_ids": inputs["input_ids"], "labels": targets["input_ids"]}

# Load dataset
dataset = load_dataset("training_data.json")
dataset = Dataset.from_list(dataset)
tokenized_dataset = dataset.map(tokenize_data, batched=True)

# Training Configuration
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=1000,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train and Save Model
trainer.train()
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print(" Fine-tuned model with LoRA and dataset saved successfully!")
