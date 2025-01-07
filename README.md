# Custom-Llama-3.2-1B-Language-Model-Integration-with-LangChain-for-Interactive-Q-A


# 🌟 Interactive Q&A System with Llama 3.2-1B and Hugging Face Integration 🌟

Welcome to the **Interactive Question-Answering System** project! This repository demonstrates how to build an advanced Q&A system powered by **Meta's Llama 3.2-1B model** using **Hugging Face Transformers**, **LangChain**, and **custom fine-tuning**. 🚀

---

## 📖 Overview

This project showcases:
- The **Llama 3.2-1B** model integration for efficient and interactive Q&A systems.
- Step-by-step **model download**, **setup**, and **interaction**.
- Fine-tuning with **custom datasets** to adapt the model for domain-specific tasks.
- Usage of **LangChain** to create pipelines for enhanced functionality.
- Saving/loading models from Google Drive for seamless collaboration.

---

## ✨ Features

- **Interactive Console-Based Q&A**: Users can ask questions and receive real-time answers.
- **Customizable Fine-Tuning**: Train the model on your dataset for domain-specific applications.
- **LangChain Support**: Build pipelines for better query management and workflows.
- **Efficient Storage**: Save models locally or in Google Drive for easy reuse.

---

## 🚀 Quickstart Guide

### 1️⃣ Prerequisites

Install the required libraries:
```bash
pip install llama-stack transformers langchain datasets accelerate
2️⃣ Download the Llama Model
List available models:

bash
Copy code
!llama model list
Download Llama 3.2-1B:

bash
Copy code
!llama model download --source meta --model-id Llama3.2-1B
🛠️ Model Configuration
Save the config.json for optimal performance:

python
Copy code
import json

config_data = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 2048,
    "num_hidden_layers": 16,
    "num_attention_heads": 32,
    "max_position_embeddings": 131072,
    "vocab_size": 128256,
    "torch_dtype": "bfloat16"
}

with open("/root/.llama/checkpoints/Llama3.2-1B/config.json", "w") as f:
    json.dump(config_data, f, indent=4)

print("Configuration saved!")
💬 Interactive Q&A System
Run this script to start the interactive Q&A session:

python
Copy code
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_path = "/root/.llama/checkpoints/Llama3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Set pad_token if not defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Interactive loop
print("🤖 Welcome to the Interactive Q&A System! Type 'exit' to quit.")
while True:
    question = input("You: ")
    if question.lower() == "exit":
        print("👋 Goodbye!")
        break
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=50)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"💡 Model: {answer}")
🎯 Fine-Tuning the Model
Customize the model with your dataset:

1️⃣ Define Your Dataset
python
Copy code
data = {
    "context": [
        "The capital of France is Paris.",
        "The moon orbits the Earth."
    ],
    "question": [
        "What is the capital of France?",
        "What orbits the Earth?"
    ],
    "answers": [
        {"text": "Paris", "answer_start": 28},
        {"text": "The moon", "answer_start": 0},
    ]
}
dataset = Dataset.from_dict(data)
2️⃣ Fine-Tune the Model
python
Copy code
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

trainer.train()
🌈 LangChain Integration
Enable LangChain for enhanced pipelines:

python
Copy code
from langchain import LLMChain
from langchain.prompts import PromptTemplate

# Custom prompt template
prompt_template = PromptTemplate(input_variables=["question"], template="Question: {question}\nAnswer:")

# LangChain LLM integration
from transformers import pipeline
llm_pipeline = pipeline("text-generation", model=model_path, tokenizer=model_path)
llm_chain = LLMChain(prompt=prompt_template, llm=llm_pipeline)

# Example query
response = llm_chain.run({"question": "What is the capital of India?"})
print("Answer:", response)
📦 Save Model to Google Drive
Save your model for future use:

python
Copy code
from google.colab import drive
drive.mount('/content/drive')

model_path = '/content/drive/MyDrive/Llama3.2-1B/'
tokenizer.save_pretrained(model_path)
model.save_pretrained(model_path)
🧠 Example Interaction
vbnet
Copy code
🤖 Welcome to the Interactive Q&A System! Type 'exit' to quit.

You: What is the capital of India?  
💡 Model: The capital of India is New Delhi.

You: How are you?  
💡 Model: I'm an AI model here to assist you. How can I help?

You: Exit  
👋 Goodbye!
🔧 Customization Options
Fine-Tuning: Train the model on your custom dataset for domain-specific tasks.
LangChain Pipelines: Build complex workflows with ease.
Model Variants: Use different Llama versions based on your requirements.
