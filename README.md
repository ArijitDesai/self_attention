# Self-Attention from Scratch Using PyTorch

This repository demonstrates a step-by-step implementation of the **self-attention mechanism** using PyTorch. It provides a simple example to illustrate how attention scores are calculated and how they help in understanding relationships between words in a sentence.

## Features

- **Word Embeddings**:  
  Utilizes Word2Vec to generate word embeddings for a sample sentence.

- **Query, Key, and Value Vectors**:  
  Explains how to create Query, Key, and Value vectors using learnable weight matrices, similar to how these concepts are used in the Transformer architecture.

- **Attention Score Calculation**:  
  Shows how to compute attention scores using the dot product of Query and Key vectors, then scale and normalize them using softmax.

- **Context Vectors**:  
  Computes context vectors for each word in the sentence, summarizing its understanding by integrating information from all the other words.

- **Intuitive Analogies**:  
  Uses database analogies and detailed comments throughout the code to explain the fundamentals of self-attention in natural language processing (NLP).

## How to Use

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/Self-Attention-Demo.git
   cd Self-Attention-Demo
   ```
2. **Run the Script: Execute the self_attention.py script to see how self-attention is applied step-by-step:
   ```bash
   python self_attention.py
   ``` 
3. **Analyze the Output: The script will print the context vectors for each word in the example sentence, showing how each word's representation changes after considering its relationship with other words.

## Understanding the Concepts
This project breaks down complex concepts like Query, Key, and Value in the self-attention mechanism, making it easier to understand how modern NLP models like Transformers work under the hood.

## What’s Inside?

1. A comprehensive implementation of self-attention using PyTorch.
2. Detailed comments and explanations to guide you through the process.
3. Analogies to help relate self-attention to everyday concepts, making it beginner-friendly.
