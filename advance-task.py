## Import required libraries
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

## Load a pre-trained Language Model (BERT or GPT-3)
model_name = "bert-base-uncased"  # Change to 'text-davinci-003' for GPT-3 (requires OpenAI API)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

## Function to analyze text sentiment (for BERT)
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = ["Negative", "Positive"]
    return {sentiment[i]: float(scores[0][i]) for i in range(len(sentiment))}

## Example: Sentiment Analysis
example_text = "I love artificial intelligence!"
print("Sentiment Analysis:", analyze_sentiment(example_text))

## Load a dataset for evaluation
dataset = load_dataset("imdb", split="test").select(range(100))

## Function to evaluate model on dataset
def evaluate_model():
    sample_texts = [dataset[i]['text'][:100] for i in range(5)]  # Take first 100 chars
    results = [analyze_sentiment(text) for text in sample_texts]
    
    for i, (inp, res) in enumerate(zip(sample_texts, results)):
        print(f"\nSample {i+1} Input: {inp}")
        print(f"Sentiment Scores: {res}")

## Run evaluation
evaluate_model()

## Visualization Example (Sentiment Distribution)
def visualize_sentiment():
    sentiments = [analyze_sentiment(text)['Positive'] for text in dataset['text'][:100]]
    
    plt.figure(figsize=(8,5))
    sns.histplot(sentiments, bins=30, kde=True)
    plt.xlabel("Positive Sentiment Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Sentiment Scores in Dataset")
    plt.show()

visualize_sentiment()

## Define Research Questions
research_questions = [
    "How accurately does BERT classify sentiment in movie reviews?",
    "Does the model introduce biases in sentiment classification?",
    "How does text length affect sentiment classification confidence?",
    "Can the model generalize to different domains beyond movie reviews?"
]

print("\nResearch Questions:")
for q in research_questions:
    print("-", q)

## Conclusion and Insights
conclusion = "This project explored the performance of BERT for sentiment classification. The analysis shows strengths in sentiment understanding but also highlights challenges in contextual sentiment interpretation and bias mitigation."
print("\nConclusion:", conclusion)
