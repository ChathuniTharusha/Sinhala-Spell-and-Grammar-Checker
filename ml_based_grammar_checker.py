import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class MLBasedGrammarChecker:
    def __init__(self, dataset_path="data/sinhala_grammar_checker_large_dataset.csv"):
        # Load the dataset
        self.data = pd.read_csv(dataset_path, encoding='utf-8-sig')
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))  # Use TF-IDF Vectorizer

        # Prepare data
        X = self.data['Sentence']
        y = self.data['Label'].apply(lambda x: 1 if x == "Correct" else 0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Transform the text data into feature vectors
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # Train a Logistic Regression model with class weights
        self.model = LogisticRegression(class_weight='balanced', random_state=42)
        self.model.fit(X_train_vec, y_train)

        # Evaluate model accuracy on the test set
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"ML Model Accuracy: {accuracy * 100:.2f}%")

    def ml_grammar_checker(self, sentence):
        """
        Use ML-based Logistic Regression model to check the grammar of a sentence.
        """
        sentence_vec = self.vectorizer.transform([sentence])
        prediction = self.model.predict(sentence_vec)
        return "Correct" if prediction[0] == 1 else "Incorrect (ML-Based)"

    def check_grammar(self, sentence):
        """
        Check grammar using ML-based approach.
        """
        ml_result = self.ml_grammar_checker(sentence)
        return {
            "ML-Based Result": ml_result
        }


# Sample Usage
if __name__ == "__main__":
    ml_checker = MLBasedGrammarChecker()
    sentences = [
        "අපි ගමනට ගියේය.",  # Predicted result
        "අපි ගමට ගියමු.",  # Predicted result
        "මම පොත කියවයි.",  # Predicted result
        "මම ගමට ගියමි."  # Predicted result
    ]

    for sentence in sentences:
        result = ml_checker.check_grammar(sentence)
        print(f"Input Sentence: {sentence}")
        print(f"Result: {result}")
        print("-" * 50)
