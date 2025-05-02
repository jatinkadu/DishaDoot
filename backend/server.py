from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import traceback

app = Flask(__name__)
CORS(app)  # Allow React frontend to access this API

# Root route to confirm server is running
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ Server is running! Use /api/get-questions to fetch questions."})

# Function to load and process questions from the Excel file
def load_questions():
    try:
        df = pd.read_excel("QuizData.xlsx", engine="openpyxl")

        # Ensure required columns exist
        required_columns = {"Section", "Question", "Option A", "Option B", "Option C", "Option D", "Correct Answer"}
        if not required_columns.issubset(df.columns):
            error_message = "❌ ERROR: Invalid column names in the Excel file."
            print(error_message)  # ✅ Print error in terminal
            return {"error": error_message}

        sections = df["Section"].unique()
        quiz_questions = []

        for section in sections:
            section_questions = df[df["Section"] == section]

            # Ensure at least 10 questions exist per section
            if len(section_questions) < 10:
                continue  # Skip sections with less than 10 questions

            selected_questions = section_questions.sample(n=10, random_state=42)

            for _, row in selected_questions.iterrows():
                quiz_questions.append({
                    "section": row["Section"],
                    "question": row["Question"],
                    "options": [row["Option A"], row["Option B"], row["Option C"], row["Option D"]],
                    "correct_answer": row["Correct Answer"]
                })

        return quiz_questions

    except Exception as e:
        error_message = f"❌ ERROR: {str(e)}"
        traceback.print_exc()  # ✅ Print error in terminal
        return {"error": error_message}

# API route to get random questions
@app.route("/api/get-questions", methods=["GET"])
def get_questions():
    questions = load_questions()
    
    if "error" in questions:  # ✅ Return error message in JSON response
        return jsonify(questions), 400  # HTTP 400 = Bad Request
    
    return jsonify(questions)

# Run the Flask server
if __name__ == "__main__":
    print("✅ Flask server is running at http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
