import os
import getpass
import gradio as gr
from ibm_watsonx_ai.foundation_models import ModelInference

def get_credentials():
    return {
        "url" : "https://au-syd.ml.cloud.ibm.com",
        "apikey" : "YOUR_API_KEY"
    }

model_id = "mistralai/mistral-large"

parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 900,
    "min_new_tokens": 0,
    "repetition_penalty": 1
}

project_id = os.getenv("PROJECT_ID")
space_id = os.getenv("SPACE_ID")

model = ModelInference(
    model_id = model_id,
    params = parameters,
    credentials = get_credentials(),
    project_id = project_id,
    space_id = space_id
)

prompt_input = """[INST] I want to learn about machine learning[/INST] Machine Learning Learning Guide

1. Definition
Machine Learning (ML) is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computers to perform tasks without explicit instructions. Instead, these systems learn from data, identifying patterns and making decisions with minimal human intervention. Machine Learning is widely used in various applications such as image and speech recognition, recommendation systems, and autonomous vehicles.

2. Learning Plan

Topic 1: Introduction to Machine Learning
- Sub-topics:
  - Definition and importance of Machine Learning
  - History and evolution of Machine Learning
  - Applications of Machine Learning

Topic 2: Types of Machine Learning
- Sub-topics:
  - Supervised Learning
  - Unsupervised Learning
  - Reinforcement Learning
  - Semi-supervised Learning

Topic 3: Key Algorithms
- Sub-topics:
  - Linear Regression
  - Logistic Regression
  - Decision Trees
  - Random Forests
  - Support Vector Machines (SVM)
  - K-Means Clustering
  - Neural Networks

Topic 4: Data Preprocessing
- Sub-topics:
  - Data cleaning
  - Feature scaling
  - Feature selection
  - Handling missing data

Topic 5: Evaluation Metrics
- Sub-topics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC Curve and AUC

Topic 6: Advanced Topics
- Sub-topics:
  - Deep Learning
  - Transfer Learning
  - Natural Language Processing (NLP)
  - Computer Vision

3. Exercises

Topic 1: Introduction to Machine Learning
- Exercise 1: Write a short essay explaining the difference between Artificial Intelligence, Machine Learning, and Deep Learning.
- Exercise 2: Research and list five real-world applications of Machine Learning.

Topic 2: Types of Machine Learning
- Exercise 1: Describe the difference between supervised and unsupervised learning with examples.
- Exercise 2: Implement a simple linear regression model using a dataset like the Boston Housing dataset.

Topic 3: Key Algorithms**
- Exercise 1**: Apply a decision tree classifier to the Iris dataset and evaluate the model's performance.
- Exercise 2: Implement a K-Means clustering algorithm on a dataset like the Mall Customers dataset.

Topic 4: Data Preprocessing
- Exercise 1: Clean a dataset by handling missing values and outliers.
- Exercise 2: Perform feature scaling on a dataset using standardization and normalization techniques.

Topic 5: Evaluation Metrics
- Exercise 1: Calculate the accuracy, precision, recall, and F1 score for a classification model.
- Exercise 2: Plot the ROC curve and calculate the AUC for a binary classification problem.

Topic 6: Advanced Topics
- Exercise 1: Implement a simple neural network using a framework like TensorFlow or PyTorch.
- Exercise 2: Perform sentiment analysis on a dataset of movie reviews using NLP techniques.

Resources
- Books:
  - "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron
  - "Pattern Recognition and Machine Learning" by Christopher M. Bishop

- Online Courses:
  - Coursera: "Machine Learning" by Andrew Ng
  - edX: "Introduction to Machine Learning" by Microsoft

- Websites:
  - Kaggle: A platform for data science competitions and datasets.
  - Towards: Data Science """


def generate_response(question):
    formattedQuestion = f"""<s>[INST] {question}[/INST]"""
    prompt = f"""{prompt_input}{formattedQuestion}"""
    generated_response = model.generate_text(prompt=prompt, guardrails=False)
    return generated_response

# Create a Gradio interface for the user to input a question
interface = gr.Interface(
    fn=generate_response, 
    inputs=gr.Textbox(label="Enter your question about Machine Learning"),
    outputs=gr.Textbox(label="AI Response"),
    title="Machine Learning Learning Guide",
    description="Ask any question related to Machine Learning and get a detailed answer based on the learning guide."
)

interface.launch(share=True)
