## Fake Job Posting Detector

A Streamlit-based Machine Learning web application that detects whether a given job description is Real or Fake using Natural Language Processing (NLP) and Logistic Regression.

This project helps users avoid fraudulent job postings by analyzing common patterns found in fake job advertisements.

## Features

Paste any job description and analyze it

Classifies job postings as Real or Fake

Prevents unrelated or invalid input

Fast predictions using cached machine learning model

Clean and simple user interface

Deployed using Streamlit Community Cloud

## How It Works
Text Preprocessing

Converts text to lowercase

Removes special characters and punctuation

Removes English stopwords using NLTK

Input Validation

Checks whether the input is related to job descriptions

Blocks unrelated text before sending it to the ML model

Machine Learning

TF-IDF Vectorization for feature extraction

Logistic Regression for binary classification

Output labels:

0 → Real Job

1 → Fake Job

## Tech Stack

Programming Language: Python

Frontend Framework: Streamlit

Machine Learning: Scikit-learn

NLP: TF-IDF, NLTK

Data Processing: Pandas

## Project Structure
```
fake-job-posting-detector/
│
├── app.py
├── jobs.csv
├── requirements.txt
├── README.md
```
## Running the Project Locally

Clone the repository:
```bash
git clone https://github.com/prasannnna/fake-job-posting-detector.git
```
Move into the project directory
```
cd fake-job-posting-detector
```
Install required dependencies
```bash
pip install -r requirements.txt
```
Run the Streamlit application
```bash
streamlit run app.py
```
### Deployment

This application is deployed using Streamlit Community Cloud.

URL for Live App: https://fake-job-posting-detector.streamlit.app/

### Sample Inputs

Fake Job Example
Earn 50,000 per week from home. No experience required. Registration fee mandatory.

Real Job Example
We are hiring a Data Analyst with experience in SQL, Power BI, and data visualization tools.


