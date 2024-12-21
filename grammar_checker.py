import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

class GrammarChecker:
    def __init__(self, dataset_path="data/sinhala_grammar_checker_dataset.csv"):
        # Load the dataset
        self.data = pd.read_csv(dataset_path, encoding='utf-8-sig')
        self.vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2))  # Word-level n-grams


        # Split data into training and testing sets
        X = self.data['Sentence']
        y = self.data['Label'].apply(lambda x: 1 if x == "Correct" else 0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Transform the text data to feature vectors
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # Train a Logistic Regression model
        self.model = LogisticRegression()
        self.model.fit(X_train_vec, y_train)

        # Evaluate the model
        y_pred = self.model.predict(X_test_vec)
        print("Grammar Checker Model Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

    def check_grammar(self, sentence):
        sentence_vec = self.vectorizer.transform([sentence])
        prediction = self.model.predict(sentence_vec)
        probabilities = self.model.predict_proba(sentence_vec)
        print(f"Sentence: {sentence}")
        print(f"Probabilities: {probabilities}")
        if prediction[0] == 1:
            print(f"Prediction: Correct")
            return "Correct"
        else:
            print(f"Prediction: Incorrect")
            return "Incorrect"


