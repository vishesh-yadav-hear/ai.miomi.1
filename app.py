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
        
        # Check if question already exists in the dataset
        if user_question in [preprocess_text(q) for q in questions]:
            # Find the index of the existing question
            idx = [preprocess_text(q) for q in questions].index(user_question)
            return [(answers[idx], 1.0)]  # Return the existing answer directly
        
        user_question_tfidf = vectorizer.transform([user_question])
        similarities = cosine_similarity(user_question_tfidf, tfidf_matrix).flatten()

        scores = []
        for idx, question in enumerate(questions):
            fuzzy_score = fuzz.partial_ratio(user_question, preprocess_text(question))
            weighted_score = get_weighted_score(similarities[idx], fuzzy_score)
            scores.append((idx, weighted_score))

        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:3]

        top_answers = [(answers[idx], score) for idx, score in scores if score > 0.2]
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
            response=[f"{answer}" for answer, score in top_answers],  # No numbering here
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

@app.route('/update-feedback', methods=['POST'])
def update_feedback():
    try:
        old_question = request.form.get('old_question', '').strip()
        new_question = request.form.get('new_question', '').strip()
        new_answer = request.form.get('new_answer', '').strip()

        if not old_question or not new_question or not new_answer:
            return feedback_page(message="All fields are required for update.")

        df = pd.read_csv(feedback_file)
        if old_question not in df['Question'].values:
            return feedback_page(message="The question to update does not exist.")

        # Update entry
        df.loc[df['Question'] == old_question, ['Question', 'Answer']] = [new_question, new_answer]
        df.to_csv(feedback_file, index=False)
        return feedback_page(message="Feedback updated successfully.")
    except Exception as e:
        return feedback_page(message=f"Error updating feedback: {e}")


@app.route('/delete-feedback', methods=['POST'])
def delete_feedback():
    try:
        question_to_delete = request.form.get('question_to_delete', '').strip()

        if not question_to_delete:
            return feedback_page(message="Question is required for deletion.")

        df = pd.read_csv(feedback_file)
        if question_to_delete not in df['Question'].values:
            return feedback_page(message="The question to delete does not exist.")

        # Delete entry
        df = df[df['Question'] != question_to_delete]
        df.to_csv(feedback_file, index=False)
        return feedback_page(message="Feedback deleted successfully.")
    except Exception as e:
        return feedback_page(message=f"Error deleting feedback: {e}")



@app.route('/update-database', methods=['POST'])
def update_database():
    try:
        old_question = request.form.get('old_question', '').strip()
        new_question = request.form.get('new_question', '').strip()
        new_answer = request.form.get('new_answer', '').strip()

        if not old_question or not new_question or not new_answer:
            return database_page(message="All fields are required for update.")

        df = pd.read_csv(dataset_file)
        if old_question not in df['Question'].values:
            return database_page(message="The question to update does not exist.")

        # Update entry
        df.loc[df['Question'] == old_question, ['Question', 'Answer']] = [new_question, new_answer]
        df.to_csv(dataset_file, index=False)

        # Reload TF-IDF model
        global questions, answers, vectorizer, tfidf_matrix
        questions, answers, vectorizer, tfidf_matrix = load_data()

        return database_page(message="Database updated successfully.")
    except Exception as e:
        return database_page(message=f"Error updating database: {e}")


@app.route('/delete-database', methods=['POST'])
def delete_database():
    try:
        question_to_delete = request.form.get('question_to_delete', '').strip()

        if not question_to_delete:
            return database_page(message="Question is required for deletion.")

        df = pd.read_csv(dataset_file)
        if question_to_delete not in df['Question'].values:
            return database_page(message="The question to delete does not exist.")

        # Delete entry
        df = df[df['Question'] != question_to_delete]
        df.to_csv(dataset_file, index=False)

        # Reload TF-IDF model
        global questions, answers, vectorizer, tfidf_matrix
        questions, answers, vectorizer, tfidf_matrix = load_data()

        return database_page(message="Database deleted successfully.")
    except Exception as e:
        return database_page(message=f"Error deleting database: {e}")


def feedback_page(message=None):
    if os.path.exists(feedback_file):
        feedback_data = pd.read_csv(feedback_file).to_dict(orient='records')
    else:
        feedback_data = []
    return render_template('feedback.html', feedback=feedback_data, message=message)


def database_page(message=None):
    if os.path.exists(dataset_file):
        database_data = pd.read_csv(dataset_file).to_dict(orient='records')
    else:
        database_data = []
    return render_template('database.html', database=database_data, message=message)




def load_database():
    """Utility function to load the database for rendering."""
    if os.path.exists(dataset_file):
        return pd.read_csv(dataset_file).to_dict(orient='records')
    return []


@app.route('/add-to-database', methods=['POST'])
def add_to_database():
    try:
        # Question and answer from feedback
        question = request.form.get('question', '').strip()
        answer = request.form.get('answer', '').strip()

        if not question or not answer:
            return feedback_page(message="Both question and answer are required to add to the database.")

        # Load feedback and database files
        feedback_df = pd.read_csv(feedback_file)
        database_df = pd.read_csv(dataset_file) if os.path.exists(dataset_file) else pd.DataFrame(columns=['Question', 'Answer'])

        # Check if the question already exists in the database
        if question in database_df['Question'].values:
            return feedback_page(message="This question already exists in the database.")

        # Add to database
        database_df = pd.concat([database_df, pd.DataFrame([[question, answer]], columns=['Question', 'Answer'])], ignore_index=True)
        database_df.to_csv(dataset_file, index=False)

        # Remove from feedback
        feedback_df = feedback_df[feedback_df['Question'] != question]
        feedback_df.to_csv(feedback_file, index=False)

        # Reload TF-IDF model
        global questions, answers, vectorizer, tfidf_matrix
        questions, answers, vectorizer, tfidf_matrix = load_data()

        return feedback_page(message="Entry successfully added to the database.")
    except Exception as e:
        return feedback_page(message=f"Error adding to database: {e}")


def feedback_page(message=None):
    if os.path.exists(feedback_file):
        feedback_data = pd.read_csv(feedback_file).to_dict(orient='records')
    else:
        feedback_data = []
    return render_template('feedback.html', feedback=feedback_data, message=message)



if __name__ == '__main__':
    app.run(debug=True)
