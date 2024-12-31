from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
import re
import os

app = Flask(__name__)

# Dataset and feedback file paths
dataset_file = 'questions_answers.csv'
feedback_file = 'feedback.csv'

# Function to load dataset and reinitialize TF-IDF vectorizer
def load_data():
    try:
        # Load the dataset
        questions_answers_df = pd.read_csv(dataset_file, encoding='utf-8', quotechar='"')
        questions_answers_df.columns = questions_answers_df.columns.str.strip()
        questions = questions_answers_df['Question'].tolist()
        answers = questions_answers_df['Answer'].tolist()

        # Initialize the vectorizer and fit_transform on the questions
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), analyzer='word', stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(questions)

        return questions, answers, vectorizer, tfidf_matrix
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return [], [], None, None

# Load initial data
questions, answers, vectorizer, tfidf_matrix = load_data()

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# Dynamic threshold
def get_dynamic_threshold(question_length):
    if question_length <= 5:
        return 0.2  # For short questions (5 letters or less), 20% match threshold
    elif question_length <= 10:
        return 0.5  # For questions with 6-10 letters, 50% match threshold
    else:
        return 0.7  # For longer questions, 70% match threshold

# Weighted score calculation
def get_weighted_score(tfidf_score, fuzzy_score, weight_tfidf=0.7, weight_fuzzy=0.3):
    return (tfidf_score * weight_tfidf) + (fuzzy_score / 100 * weight_fuzzy)

# Get answer from dataset
def get_dataset_answer(user_question):
    try:
        user_question = preprocess_text(user_question)
        question_length = len(user_question.split())
        threshold = get_dynamic_threshold(question_length)

        user_question_tfidf = vectorizer.transform([user_question])
        similarities = cosine_similarity(user_question_tfidf, tfidf_matrix).flatten()

        most_similar_idx = similarities.argmax()
        tfidf_score = similarities[most_similar_idx]

        best_fuzzy_score = 0
        best_fuzzy_idx = None
        for idx, question in enumerate(questions):
            fuzzy_score = fuzz.partial_ratio(user_question, preprocess_text(question))
            if fuzzy_score > best_fuzzy_score:
                best_fuzzy_score = fuzzy_score
                best_fuzzy_idx = idx

        weighted_score = get_weighted_score(tfidf_score, best_fuzzy_score)

        if weighted_score > threshold:
            if tfidf_score > (best_fuzzy_score / 100):
                return answers[most_similar_idx], weighted_score
            else:
                return answers[best_fuzzy_idx], weighted_score
        else:
            return None, 0
    except Exception as e:
        return str(e), 0

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.form.get('question', '').strip()
    if not user_question:
        return render_template('index.html', response="Please enter a valid question.")

    dataset_answer, confidence = get_dataset_answer(user_question)
    if confidence > 0.2:
        # If AI found an appropriate answer, do not save it to the feedback
        return render_template(
            'index.html',
            response=f"<strong>Answer:</strong> {dataset_answer}<br><strong>Source:</strong> dataset"
        )
    else:
        # If AI did not find an answer, save it to the feedback file
        try:
            feedback_exists = os.path.exists(feedback_file)
            with open(feedback_file, 'a', encoding='utf-8') as f:
                if not feedback_exists:
                    f.write("Question,Answer\n")
                f.write(f'"{user_question}","{dataset_answer}"\n')
        except Exception as e:
            print(f"Error saving to feedback: {e}")

        return render_template(
            'index.html',
            response="AI couldn't find an appropriate answer. Please teach AI below.",
            autofill_question=user_question
        )



@app.route('/teach', methods=['POST'])
def teach():
    new_question = request.form.get('new_question', '').strip()
    new_answer = request.form.get('new_answer', '').strip()

    if not new_question or not new_answer:
        return render_template('index.html', teach_response="Both question and answer are required.")

    try:
        # Add new question-answer pair to feedback
        feedback_exists = os.path.exists(feedback_file)
        with open(feedback_file, 'a', encoding='utf-8') as f:
            if not feedback_exists:
                f.write("Question,Answer\n")
            f.write(f'"{new_question}","{new_answer}"\n')

        # Reload the dataset and reinitialize TF-IDF model
        global questions, answers, vectorizer, tfidf_matrix
        questions, answers, vectorizer, tfidf_matrix = load_data()

        return render_template('index.html', teach_response="Feedback saved successfully. AI will now use this data.")
    except Exception as e:
        return render_template('index.html', teach_response=f"Error saving feedback: {e}")

if __name__ == '__main__':
    app.run(debug=True)
