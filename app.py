import streamlit as st
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


@st.cache_resource
def download_nltk():
    nltk.download("stopwords")

download_nltk()
stop_words = set(stopwords.words("english"))


st.markdown(
    """
    <div style="display:flex; align-items:center; gap:12px;">
        <img src="https://em-content.zobj.net/source/apple/419/man-detective-light-skin-tone_1f575-1f3fb-200d-2642-fe0f.png" width="40">
        <h1 style="margin:0;">Fake Job Posting Detector</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<h2 style="color:light-gray; font-size:22px;">Check if the job opportunity is <em>Real or Fake</em></h2>',
    unsafe_allow_html=True
)


df = pd.read_csv("jobs.csv")
df["label"] = df["label"].map({"fake": 1, "real": 0})


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

df["clean_text"] = df["text"].apply(clean_text)


@st.cache_resource
def train_model(data):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data["clean_text"])
    y = data["label"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return vectorizer, model

vectorizer, model = train_model(df)


job_keywords = [
    "job", "role", "position", "company", "salary", "experience",
    "skills", "qualification", "responsibilities", "requirements",
    "hiring", "apply", "employment", "vacancy", "work", "location"
]

def is_job_related(text):
    text = text.lower()
    return any(word in text for word in job_keywords)



with st.form("job_form"):
    user_input = st.text_area(
        " ",
        height=300,
        placeholder="Enter Job Description Here..."
    )

    submitted = st.form_submit_button("Analyze")



if submitted:
    if user_input.strip() == "":
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:10px;">
                <img src="https://em-content.zobj.net/source/apple/419/warning_26a0-fe0f.png" width="30">
                <h3 style="color:orange;">Please enter a job description</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif not is_job_related(user_input):
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:10px;">
                <img src="https://em-content.zobj.net/source/apple/419/cross-mark_274c.png" width="30">
                <h3 style="color:red;">The provided text is not related to job description</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        cleaned_text = clean_text(user_input)
        vector = vectorizer.transform([cleaned_text])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.markdown(
                """
                <div style="display:flex; align-items:center; gap:10px;">
                    <img src="https://em-content.zobj.net/source/apple/419/police-car-light_1f6a8.png" width="30">
                    <h3 style="color:red;">Fake Job Posting</h3>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style="display:flex; align-items:center; gap:10px;">
                    <img src="https://em-content.zobj.net/source/apple/419/check-mark-button_2705.png" width="30">
                    <h3 style="color:green;">Real Job Posting</h3>
                </div>
                """,
                unsafe_allow_html=True
            )
