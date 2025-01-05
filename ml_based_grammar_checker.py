import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


class MLBasedGrammarChecker:
    def __init__(self, dataset_path="data/sinhala_grammar_checker_large_dataset.csv"):
        # Load and prepare dataset
        try:
            self.data = pd.read_csv(dataset_path, encoding='utf-8-sig')
        except FileNotFoundError:
            print(f"Dataset not found at {dataset_path}")
            exit(1)

        # Prepare data
        X = self.data['Sentence']
        y = self.data['Label'].apply(lambda x: 1 if x == "Correct" else 0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Transform the text data into feature vectors
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # Train a Logistic Regression model with class weights
        self.model = LogisticRegression(class_weight='balanced', random_state=42)
        self.model.fit(X_train_vec, y_train)

        # Evaluate model accuracy
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"ML Model Accuracy: {accuracy * 100:.2f}%")
        print("Classification Report:\n", classification_report(y_test, y_pred))

    def ml_grammar_checker(self, sentence):
        """
        Use ML-based Logistic Regression model to check the grammar of a sentence.
        """
        sentence_vec = self.vectorizer.transform([sentence])
        prediction = self.model.predict(sentence_vec)
        confidence = self.model.predict_proba(sentence_vec).max()
        return prediction[0], confidence

    def suggest_correction(self, sentence):
        """
        Suggest corrections for invalid sentences based on subject-verb agreement rules.
        """
        # Define rules for subject-verb agreement
        rules = {
            "අපි": "මු.",
            "මම": "මි.",
            "ඔහු": "යි.",
            "ඇය": "යි.",
            "ඔවුන්": "යි."
        }

        # Split sentence into subject and verb
        words = sentence.split()
        if len(words) < 2:
            return "Sentence too short to analyze."

        subject, verb = words[0], words[-1]
        if subject in rules:
            correct_ending = rules[subject]
            if not verb.endswith(correct_ending):
                corrected_verb = verb[:-1] + correct_ending  # Replace only the last character(s) of the verb
                corrected_sentence = " ".join(words[:-1] + [corrected_verb])
                return corrected_sentence
            else:
                return "Sentence is grammatically correct."
        else:
            return "No suggestion available for this subject."

    def check_grammar(self, sentence):
        """
        Check grammar using ML-based approach and provide corrections if needed.
        """
        prediction, confidence = self.ml_grammar_checker(sentence)
        if prediction == 1:
            return {
                "ML-Based Result": "Correct",
                "Suggestion": "No correction needed."
            }
        else:
            correction = self.suggest_correction(sentence)
            return {
                "ML-Based Result": f"Incorrect (Confidence: {confidence:.2f})",
                "Suggestion": correction
            }


# Sample Usage
if __name__ == "__main__":
    ml_checker = MLBasedGrammarChecker()
    sentences = [
        "අපි ගමට ගියෙය",
        "ඔහු පාසලට ගියෙය",
        "මම පොතක් කියවයි",
        "ඇය ගෙදරට ආවා",
        "ඔවුන් ගමනක් සූදානම් කළා",
        "අපි ආහාර සකසා ගත්තෙමු",
        "මගේ මිතුරා කාමරයේ සිටියා",
        "ඔවුන් ගමක නතර විය",
        "අපි වත්තෙ කටයුත්තක් කළා",
        "ඇය පොතක් උගන්වමින් සිටියා"
    ]

    for sentence in sentences:
        result = ml_checker.check_grammar(sentence)
        print(f"Input Sentence: {sentence}")
        print(f"Result: {result}")
        print("-" * 50)
