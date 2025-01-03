import os
import torch
from datasets import load_dataset
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_prepare_data(data_path, format="csv"):
    """Load and prepare the dataset."""
    if format == "csv":
        dataset = load_dataset("csv", data_files=data_path)
    else:
        dataset = load_dataset("json", data_files=data_path)
    
    return dataset

def format_instruction(row):
    """Format the instruction and response into a single string."""
    return f"### Instruction:\n{row['question']}\n\n### Response:\n{row['answer']}\n\n"

def preprocess_function(examples, tokenizer):
    """Tokenize and format the examples."""
    formatted_texts = [format_instruction({"question": q, "answer": a}) 
                      for q, a in zip(examples["question"], examples["answer"])]
    
    return tokenizer(
        formatted_texts,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )

def main(args):
    # Initialize tokenizer and model
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
    model = LlamaForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load and preprocess the dataset
    dataset = load_and_prepare_data(args.data_path, args.format)
    
    # Split dataset
    dataset = dataset["train"].train_test_split(test_size=0.1)
    
    # Tokenize datasets
    tokenized_train = dataset["train"].map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    tokenized_val = dataset["test"].map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset["test"].column_names
    )

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        learning_rate=args.learning_rate,
        fp16=True,
        gradient_accumulation_steps=4,
        logging_steps=100,
        report_to="tensorboard"
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model(os.path.join(args.output_dir, "final_model"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--format", type=str, default="csv", choices=["csv", "json"])
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    
    args = parser.parse_args()
    main(args)