from flask import Flask, render_template, request
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

# Weighted score calculation
def get_weighted_score(tfidf_score, fuzzy_score, weight_tfidf=0.7, weight_fuzzy=0.3):
    return (tfidf_score * weight_tfidf) + (fuzzy_score / 100 * weight_fuzzy)

# Get top 3 answers from dataset
def get_top_answers(user_question):
    try:
        user_question = preprocess_text(user_question)
        user_question_tfidf = vectorizer.transform([user_question])
        similarities = cosine_similarity(user_question_tfidf, tfidf_matrix).flatten()

        # Combine TF-IDF and fuzzy scores
        scores = []
        for idx, question in enumerate(questions):
            fuzzy_score = fuzz.partial_ratio(user_question, preprocess_text(question))
            weighted_score = get_weighted_score(similarities[idx], fuzzy_score)
            scores.append((idx, weighted_score))

        # Sort scores in descending order and get top 3
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:3]

        top_answers = [(answers[idx], score) for idx, score in scores if score > 0.2]  # Threshold = 0.2
        return top_answers if top_answers else None
    except Exception as e:
        return [(str(e), 0)]

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.form.get('question', '').strip()
    if not user_question:
        return render_template('index.html', response="Please enter a valid question.")

    top_answers = get_top_answers(user_question)
    if top_answers:
        return render_template(
            'index.html',
            response=[f"{i+1}. {answer}" for i, (answer, score) in enumerate(top_answers)],
            autofill_question=user_question
        )
    else:
        # Save the unanswered question to feedback
        try:
            feedback_exists = os.path.exists(feedback_file)
            with open(feedback_file, 'a', encoding='utf-8') as f:
                if not feedback_exists:
                    f.write("Question,Answer\n")
                f.write(f'"{user_question}",""\n')
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

@app.route('/feedback', methods=['POST'])
def feedback():
    question = request.form.get('question', '').strip()
    answer = request.form.get('answer', '').strip()

    if not question or not answer:
        return render_template(
            'index.html',
            teach_response="Both question and answer are required for feedback.",
            autofill_question=question
        )

    try:
        # Save the question-answer pair in feedback.csv
        feedback_exists = os.path.exists(feedback_file)
        with open(feedback_file, 'a', encoding='utf-8') as f:
            if not feedback_exists:
                f.write("Question,Answer\n")
            f.write(f'"{question}","{answer}"\n')

        return render_template(
            'index.html',
            teach_response="Thanks for Teach me.",
            autofill_question=question
        )
    except Exception as e:
        return render_template(
            'index.html',
            teach_response=f"Error saving feedback: {e}",
            autofill_question=question
        )


@app.route('/feedback', methods=['GET'])
def feedback_page():
    try:
        if os.path.exists(feedback_file):
            feedback_data = pd.read_csv(feedback_file)
        else:
            feedback_data = pd.DataFrame(columns=['Question', 'Answer'])

        # Add enumerated index for rendering in template
        feedback_data = feedback_data.reset_index()  # Ensure indexes are consistent
        return render_template('feedback.html', feedback=feedback_data.to_dict(orient='records'))
    except Exception as e:
        return f"Error loading feedback: {e}"

@app.route('/database', methods=['GET'])
def database_page():
    try:
        if os.path.exists(dataset_file):
            database_data = pd.read_csv(dataset_file)
        else:
            database_data = pd.DataFrame(columns=['Question', 'Answer'])

        # Add enumerated index for rendering in template
        database_data = database_data.reset_index()  # Ensure indexes are consistent
        return render_template('database.html', database=database_data.to_dict(orient='records'))
    except Exception as e:
        return f"Error loading database: {e}"



if __name__ == '__main__':
    app.run(debug=True)
