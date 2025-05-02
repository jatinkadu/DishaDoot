import pandas as pd
import re

# Load the Excel file
file_path = "Unique_MCQ_Question_Bank.xlsx"  # Change this to your file path
df = pd.read_excel(file_path)

# Function to remove only numbers inside parentheses
def remove_numbers_in_parentheses(text):
    return re.sub(r"\(\d+\)", "", str(text)).strip()  # Removes (numbers) but keeps other text

# Apply function to a specific column (Change 'Column_Name' to your actual column name)
df['Question'] = df['Question'].apply(remove_numbers_in_parentheses)

# Save the cleaned data back to a new Excel file
output_file = "cleaned_file.xlsx"
df.to_excel(output_file, index=False)

print(f"Cleaned file saved as {output_file}")
