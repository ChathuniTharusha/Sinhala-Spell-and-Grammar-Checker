from grammar_checker import GrammarChecker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from sklearn.metrics import accuracy_score

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def apply_basic_rules(sentence):
    """
    Apply basic grammar rules to the sentence.
    """
    # Define rules for specific subjects and their correct endings
    subject_rules = {
        "අපි": "මු.",
        "මම": "මි.",
        "ඔහු": "යි.",
        "ඇය": "යි.",
        "ඔවුන්": "යි."
    }

    # Check if the sentence starts with a defined subject
    for subject, correct_ending in subject_rules.items():
        if sentence.startswith(subject):
            words = sentence.split()
            if len(words) > 1:  # Ensure there's a verb to process
                verb = words[-1].rstrip(".")  # Remove period for processing
                if not verb.endswith(correct_ending.rstrip(".")):
                    corrected_verb = verb.rstrip("මුමියි.") + correct_ending.rstrip(".")
                    corrected_sentence = " ".join(words[:-1] + [corrected_verb]) + "."
                    return False, (
                        f"If the sentence starts with '{subject}', it should end with '{correct_ending}'."
                    ), corrected_sentence

    return True, None, sentence


class GrammarChecker:
    def __init__(self, dataset_path="data/sinhala_grammar_checker_large_dataset.csv"):
        # Load the dataset for ML-based approach
        self.data = pd.read_csv(dataset_path, encoding='utf-8-sig')
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))  # Use TF-IDF Vectorizer

        # Prepare data for Logistic Regression model
        X = self.data['Sentence']
        y = self.data['Label'].apply(lambda x: 1 if x == "Correct" else 0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Transform the text data into feature vectors
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # Train a Logistic Regression model with class weights
        self.model = LogisticRegression(class_weight='balanced', random_state=42)
        self.model.fit(X_train_vec, y_train)

        # Store the test data for accuracy evaluation
        self.X_test = X_test
        self.y_test = y_test
        self.X_test_vec = X_test_vec

    def ml_grammar_checker(self, sentence):
        """
        Use ML-based Logistic Regression model to check the grammar of a sentence.
        """
        sentence_vec = self.vectorizer.transform([sentence])
        prediction = self.model.predict(sentence_vec)
        return "Correct" if prediction[0] == 1 else "Incorrect (ML-Based)"

    def check_grammar(self, sentence):
        """
        Check grammar using rule-based and ML-based approaches.
        """
        # Step 1: Apply rules
        valid, message, corrected_sentence = apply_basic_rules(sentence)
        if not valid:
            return f"Rule Violation: {message}\nSuggested Correction: {corrected_sentence}"

        # Step 2: Apply ML-based grammar checking
        ml_result = self.ml_grammar_checker(corrected_sentence)

        return {
            "Rule-Based Result": "Valid" if valid else "Invalid",
            "ML-Based Result": ml_result,
            "Corrected Sentence": corrected_sentence
        }

    def evaluate_accuracy(self):
        """
        Evaluate and print accuracy for both rule-based and ML-based grammar checking.
        """
        rule_based_correct = 0
        ml_based_correct = 0
        total_sentences = len(self.X_test)

        # Evaluate Rule-based approach
        for sentence in self.X_test:
            rule_valid, _, _ = apply_basic_rules(sentence)
            if rule_valid:
                rule_based_correct += 1

        # Evaluate ML-based approach
        ml_predictions = self.model.predict(self.X_test_vec)
        ml_based_correct = sum(ml_predictions == self.y_test)

        # Calculate accuracies
        rule_based_accuracy = rule_based_correct / total_sentences * 100
        ml_based_accuracy = ml_based_correct / total_sentences * 100

        print(f"Rule-Based Accuracy: {rule_based_accuracy:.2f}%")
        print(f"ML-Based Accuracy: {ml_based_accuracy:.2f}%")

# Sample sentences for testing
sentences = [
    "අපි ගමනට ගියේය.",  # Rule should trigger
    "අපි ගමට ගියමු.",  # Rule should pass
    "මම පොත කියවයි.",  # Rule should trigger
    "මම ගමට ගියමි.",  # Rule should pass
    "ඇය පොත කියවමි."  # Should be processed by ML grammar checker
]

# Initialize grammar checker
grammar_checker = GrammarChecker(dataset_path="data/sinhala_grammar_checker_large_dataset.csv")

# Process sentences
for sentence in sentences:
    print(f"Input Sentence: {sentence}")

    # Step 1: Apply Basic Rules
    rule_valid, rule_message, rule_corrected = apply_basic_rules(sentence)
    if not rule_valid:
        print(f"Rule Violation: {rule_message}")
        print(f"Suggested Correction (Rule-Based): {rule_corrected}")
        print("-" * 50)
        continue  # Skip further checks if rule is violated

    # Step 2: Check Grammar Using All Approaches
    results = grammar_checker.check_grammar(rule_corrected)

    print(f"Rule-Based Result: {results['Rule-Based Result']}")
    print(f"ML-Based Grammar Checker Result: {results['ML-Based Result']}")
    print("-" * 50)

# Calculate and print accuracy
grammar_checker.evaluate_accuracy()
