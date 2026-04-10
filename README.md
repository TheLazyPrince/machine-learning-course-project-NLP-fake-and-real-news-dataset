#  Fake News Classification: A First-Principles Machine Learning Approach

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Logistic%20Regression-orange.svg)
![NLP](https://img.shields.io/badge/NLP-Bag%20of%20Words-brightgreen.svg)
![Data Science](https://img.shields.io/badge/Data-Pandas%20%7C%20NumPy-lightgrey.svg)

##  Executive Summary
This repository features an end-to-end Natural Language Processing (NLP) pipeline designed to classify news articles as "Fake" or "True". 

Unlike standard projects that rely on high-level abstraction libraries (e.g., `scikit-learn`), this project was built **entirely from scratch** to demonstrate a profound mathematical and algorithmic understanding of Machine Learning concepts. Every component, from feature extraction to the gradient descent optimization, was manually implemented in pure Python and NumPy.

##  Core Algorithmic Implementations

### 1. Custom Logistic Regression via Gradient Descent
Instead of importing a pre-built model, I implemented a custom `LogisticRegression` class. The training loop calculates the **forward pass** (using a Sigmoid activation function) and the **backward pass** (computing gradients for weights and biases) to optimize the loss function iteratively over defined epochs.

### 2. Manual Hyperparameter Tuning & Validation
To ensure the model generalizes well and prevents overfitting, I developed a custom **K-Fold Cross-Validation (k=5)** algorithm and a **Grid Search** mechanism. This rigorously tests various learning rates to find the mathematical optimum before final training.

### 3. NLP Feature Engineering
* **Regex Preprocessing:** Developed a text-cleaning pipeline to strip punctuation, URLs, numbers, and artifacts.
* **Bag of Words (BoW):** Programmed a custom vectorizer to build a vocabulary of the top 3,000 contextual words and transform raw text into numerical matrices.

##  Dataset & Performance
* **Data Source:** The model was trained on a balanced dataset of ~45,000 articles (23.5K Fake, 21.4K True).
* **Evaluation Metric:** Implemented a custom `calculate_f1` function (measuring Precision and Recall via True Positives, False Positives, and False Negatives).
* **Final Result:** The raw algorithmic implementation successfully achieved an **F1-Score of 0.9736** on the unseen test set.

##  How to Run
1. Clone the repository.
2. Ensure you have `pandas` and `numpy` installed.
3. Run the Jupyter Notebook to view the step-by-step pipeline, from text sanitization to final predictions.

---

##  About the Author
**INON**
B.Sc. Computer Science student at HIT. 
Passionate about Computer Vision, Data Science, and translating complex mathematical theories into robust, production-ready code.
