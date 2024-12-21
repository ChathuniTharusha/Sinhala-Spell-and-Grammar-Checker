from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd


class GrammarChecker:
    def __init__(self, dataset_path="data/sinhala_grammar_checker_dataset.csv"):
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

        # Evaluate the model
        y_pred = self.model.predict(X_test_vec)
        print("Grammar Checker Model Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

    def apply_rules(self, sentence):
        """
        Apply predefined grammar rules to the sentence and suggest corrections.
        """
        rules = [
            # Rule: If the sentence starts with 'අපි', it should end with 'මු'
            (
                lambda s: s.startswith("අපි") and not s.endswith("මු."),
                "If the sentence starts with 'අපි', it should end with 'මු.'",
                lambda s: s.rstrip(".") + " මු."
            ),
        ]

        for condition, message, correction in rules:
            if condition(sentence):
                print(f"Rule Triggered: {message}")
                suggested_correction = correction(sentence)
                return False, message, suggested_correction

        return True, None, None

    def check_grammar(self, sentence):
        """
        Check grammar using both rule-based and ML approaches, and suggest corrections.
        """
        # Apply rule-based checks first
        is_valid, rule_message, suggestion = self.apply_rules(sentence)
        if not is_valid:
            return f"Incorrect (Rule Violation): {rule_message}\nSuggested Correction: {suggestion}"

        # ML-based grammar checking
        sentence_vec = self.vectorizer.transform([sentence])
        prediction = self.model.predict(sentence_vec)
        if prediction[0] == 1:
            return "Correct"
        else:
            return "Incorrect (ML-Based)"
