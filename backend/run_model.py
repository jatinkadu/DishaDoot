import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

# Load dataset
df = pd.read_csv("course_recommendations.csv", dtype=str)  # Ensure courses remain as strings
df.fillna("", inplace=True)

# Define input and output columns
input_columns = ['Hobby_1', 'Hobby_2', 'Hobby_3', 'Hobby_4', 'Hobby_5', 'SSC Percent', 'Aptitude Percent']
output_columns = ['Course_1', 'Course_2', 'Course_3', 'Course_4', 'Course_5']

# Combine input columns into a single string per row
df['combined_input'] = df[input_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
df['combined_output'] = df[output_columns].apply(lambda x: ', '.join(x.astype(str)), axis=1)

# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(df['combined_input'], df['combined_output'], test_size=0.2, random_state=42)

# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class MultiLabelBERT(nn.Module):
    def __init__(self, num_labels):
        super(MultiLabelBERT, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(pooled_output)
        return self.classifier(x)

# Extract all unique courses exactly as they appear in the dataset
unique_courses = sorted(set(df[output_columns].values.flatten()))  # Preserve exact names
num_labels = len(unique_courses)

# Create a strict mapping without modifying names
course_to_idx = {course: idx for idx, course in enumerate(unique_courses)}

# Function to convert labels into multi-hot vectors
def get_label_vector(label_row):
    labels = torch.zeros(num_labels)
    for c in label_row.split(', '):  # Keep original formatting
        if c in course_to_idx:  # Avoid stripping spaces
            labels[course_to_idx[c]] = 1
    return labels

# Tokenize input texts
def tokenize_function(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")

train_encodings = tokenize_function(train_texts.tolist())
test_encodings = tokenize_function(test_texts.tolist())

# Convert output labels to multi-hot encoding
train_labels_idx = torch.stack([get_label_vector(row) for row in train_labels])
test_labels_idx = torch.stack([get_label_vector(row) for row in test_labels])

# Custom dataset class
class CourseDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

train_dataset = CourseDataset(train_encodings, train_labels_idx)
test_dataset = CourseDataset(test_encodings, test_labels_idx)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiLabelBERT(num_labels).to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.BCEWithLogitsLoss()  # Multi-label classification loss

from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

def train_model(model, train_loader, epochs=5):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(**inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")
    print("Training complete!")

train_model(model, train_loader)

# Save model
torch.save(model.state_dict(), "bert_course_recommendation.pt")

# Predict function
def predict_courses(input_text):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        outputs = model(**inputs)
        probabilities = torch.sigmoid(outputs).squeeze()
        top_indices = torch.argsort(probabilities, descending=True)[:5]
        recommended_courses = [unique_courses[i] for i in top_indices]
        return recommended_courses

# Example prediction
sample_input = "Reading more than 75 more than 80"
predictions = predict_courses(sample_input)
print("Recommended Courses:", predictions)
