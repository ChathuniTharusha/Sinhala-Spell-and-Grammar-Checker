import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

# Load and shuffle the dataset
data = pd.read_csv("Datasets/test_set_for_supervised.csv")
data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the data

# Separate features and labels
X = data['word']
y = data['label']

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
X_vectorized = vectorizer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Define models to train
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC()
}

# Train and evaluate each model
results = {}
print("Training models...\n")

for name, model in tqdm(models.items(), desc="Training Progress"):
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

    # Display classification report
    print(f"\nModel: {name}")
    print(classification_report(y_test, y_pred))

# Display final results
print("\nFinal Accuracy Scores:")
for model_name, accuracy in results.items():
    print(f"{model_name}: {accuracy:.4f}")
