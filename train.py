from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
import json

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # Replace with your model

def load_training_data(file_path):
    """Load training dataset for fine-tuning."""
    with open(file_path, "r") as f:
        data = json.load(f)
    
    return [{"input_text": d["prompt"], "output_text": d["response"]} for d in data]

def tokenize_data(data, tokenizer):
    """Tokenize dataset for training."""
    inputs = tokenizer([d["input_text"] for d in data], padding=True, truncation=True, return_tensors="pt")
    labels = tokenizer([d["output_text"] for d in data], padding=True, truncation=True, return_tensors="pt").input_ids
    return {"input_ids": inputs["input_ids"], "labels": labels}

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Load and tokenize dataset
    training_data = load_training_data("training_data.json")
    tokenized_data = tokenize_data(training_data, tokenizer)

    # Training setup
    training_args = TrainingArguments(
        output_dir="./model_output",
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
        train_dataset=tokenized_data,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
