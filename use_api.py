import pyttsx3
import speech_recognition as sr
import requests

# Initialize the TTS engine
engine = pyttsx3.init()

# Function to speak text (TTS)
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Function to recognize speech (STT)
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for a question...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        question = recognizer.recognize_google(audio)
        print(f"You said: {question}")
        return question
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError:
        print("Could not request results from Google Speech Recognition service.")
        return None

# Function to ask the API for an answer
def ask_question(api_url, question):
    payload = {"question": question}
    headers = {"Content-Type": "application/json"}

    try:
        # API call to fetch the answer
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()

        # Display the answer or error
        if "answer" in data:
            answer = data['answer']
            print(f"Answer: {answer}")
            speak_text(answer)  # Speak the answer using TTS
        elif "error" in data:
            error_message = f"Error: {data['error']}"
            print(error_message)
            speak_text(error_message)  # Speak the error message
        else:
            error_message = "No response received from the API."
            print(error_message)
            speak_text(error_message)  # Speak the error message
    except requests.exceptions.RequestException as e:
        error_message = f"An error occurred: {e}"
        print(error_message)
        speak_text(error_message)  # Speak the error message

# API URL (replace with your API URL)
api_url = "http://127.0.0.1:5000/ask"  # Replace this with your actual API URL

# Main loop to interact with user
while True:
    # Recognize speech (STT)
    question = recognize_speech()
    if question:
        # Ask the API with the recognized question
        ask_question(api_url, question)
    else:
        speak_text("Sorry, I couldn't hear anything. Please try again.")
   