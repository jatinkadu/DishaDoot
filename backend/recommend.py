from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import time
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")

# Load dataset
output_columns = ['Course_1', 'Course_2', 'Course_3', 'Course_4', 'Course_5']
df = pd.read_csv("final_course_recommendations.csv", dtype=str).fillna("")
unique_courses = sorted(set(df[output_columns].values.flatten()))

# Define model class
class MultiLabelBERT(nn.Module):
    def __init__(self, num_labels):
        super(MultiLabelBERT, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(pooled_output))

# Load trained model
num_labels = len(unique_courses)
model = MultiLabelBERT(num_labels).to(device)
model.load_state_dict(torch.load("bert_course_recommendation.pt", map_location=device))
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Function to determine SSC and aptitude conditions
def get_conditions(ssc_percent, aptitude_percent):
    return ("less than 75" if ssc_percent < 75 else "more than 75"), \
           ("less than 80" if aptitude_percent < 80 else "more than 80")

# Predict function
def predict_courses(hobbies, ssc_percent, aptitude_percent):
    condition1, condition2 = get_conditions(ssc_percent, aptitude_percent)
    input_text = " ".join(hobbies) + f" {condition1} {condition2}"
    print(f"📝 Input Text: {input_text}")

    with torch.no_grad():
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        start_time = time.time()
        outputs = model(**inputs)
        end_time = time.time()
        probabilities = torch.sigmoid(outputs).squeeze()
        top_indices = torch.argsort(probabilities, descending=True)[:5]
        recommended_courses = [unique_courses[i] for i in top_indices if i < len(unique_courses)]

        print(f"🚀 Inference Time: {end_time - start_time:.4f} sec on {device}")
        return recommended_courses

# API route to fetch recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    print(f"📩 Received Data: {data}")

    hobbies = data.get("hobbies", [])
    ssc_percent = data.get("ssc_percent", 0)
    aptitude_percent = data.get("aptitude_percent", 0)

    if not hobbies or ssc_percent is None or aptitude_percent is None:
        return jsonify({"error": "Invalid data"}), 400

    recommendations = predict_courses(hobbies, ssc_percent, aptitude_percent)
    return jsonify({"recommendations": recommendations})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
