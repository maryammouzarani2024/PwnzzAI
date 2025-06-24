import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Sample customer data
customer_data = [
    "Customer John Doe made a purchase of $50.",
    "Customer Jane Smith asked for product recommendations.",
    "Customer Alice Brown returned a defective product.",
    "Customer Bob White inquired about the status of their order.",
    "Customer Maria Green subscribed to our monthly newsletter.",
    "Customer Steve Black complained about delayed shipping.",
    "Customer Lisa Blue requested a refund for a damaged item.",
    "Customer Richard Gray bought a new laptop and asked for technical support."
]

# Convert the sample data into a Hugging Face dataset
data_dict = {"customer_info": customer_data}
dataset = Dataset.from_dict(data_dict)

# Load GPT-2 pre-trained model and tokenizer
model_name = "gpt2"  # You can also use smaller models like "distilgpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set pad_token to eos_token (because GPT-2 does not have a padding token by default)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["customer_info"], padding="max_length", truncation=True, max_length=512)

# Apply the tokenization to the whole dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="./gpt2_finetuned",        # Directory to save the model
    evaluation_strategy="no",          # Evaluation strategy during training
    learning_rate=2e-5,                   # Learning rate for fine-tuning
    per_device_train_batch_size=2,        # Batch size per device (CPU in your case)
    per_device_eval_batch_size=2,         # Evaluation batch size
    num_train_epochs=3,                   # Number of epochs to train
    weight_decay=0.01,                    # Weight decay to avoid overfitting
    save_strategy="epoch",                # Save the model after each epoch
    no_cuda=True                           # Force training on CPU
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                          # The model to train
    args=training_args,                   # Training arguments
    train_dataset=tokenized_datasets,     # Training dataset
)

# Start fine-tuning
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./gpt2_finetuned")
tokenizer.save_pretrained("./gpt2_finetuned")

print("Fine-tuning complete! Model saved to './gpt2_finetuned'")

# Test routine: Generate text based on a prompt

def test_model(prompt):
    # Encode the input prompt to tokens
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate text from the model
    with torch.no_grad():
        output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
    
    # Decode the generated tokens back to text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print(f"Prompt: {prompt}")
    print(f"Generated text: {generated_text}")

# Test the model with a sample prompt
test_prompt = "Customer Sarah White is asking for help with"
test_model(test_prompt)

