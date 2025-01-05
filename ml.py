import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv('data/sinhala_grammar_checker_large_dataset.csv')

# Extract features and target
df['verb_last_letter'] = df['verb'].apply(lambda x: x[-1])  # Extract last letter
X = df[['subject', 'verb_last_letter']]
y = df['label']  # Assuming the column name for labels is 'label'

# Encode categorical data
le_subject = LabelEncoder()
le_verb_last = LabelEncoder()
le_label = LabelEncoder()

X['subject'] = le_subject.fit_transform(X['subject'])
X['verb_last_letter'] = le_verb_last.fit_transform(X['verb_last_letter'])
y = le_label.fit_transform(y)  # Converts 'Correct' to 1, 'Incorrect' to 0

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Optional: Test with a new subject-verb pair
test_subject = "ඔහු"
test_verb = "කියනවා"
test_verb_last_letter = test_verb[-1]

test_data = pd.DataFrame({
    'subject': [le_subject.transform([test_subject])[0]],
    'verb_last_letter': [le_verb_last.transform([test_verb_last_letter])[0]]
})

prediction = model.predict(test_data)
predicted_label = le_label.inverse_transform(prediction)
print("Prediction for new input:", predicted_label[0])
