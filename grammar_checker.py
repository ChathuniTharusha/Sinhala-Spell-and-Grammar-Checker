from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForSequenceClassification, AutoTokenizer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd


class GrammarChecker:
    def __init__(self, dataset_path="data/sinhala_grammar_checker_large_dataset.csv", model_path=None):
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

        # Load FLAN-T5 model for grammar checking
        self.tokenizer_flan_t5 = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.flan_t5_model = AutoModelForSequenceClassification.from_pretrained("google/flan-t5-base")

    def apply_advanced_rules(self, sentence):
        """
        Apply grammar rules based on subject and verb endings.
        """
        # Define rules for specific subjects
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

    def ml_grammar_checker(self, sentence):
        """
        Use ML-based Logistic Regression model to check the grammar of a sentence.
        """
        sentence_vec = self.vectorizer.transform([sentence])
        prediction = self.model.predict(sentence_vec)
        return "Correct" if prediction[0] == 1 else "Incorrect (ML-Based)"

    def flan_t5_grammar_checker(self, sentence):

        inputs = self.tokenizer_flan_t5.encode("correct grammar: " + sentence, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        outputs = self.flan_t5_model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
        corrected_sentence = self.tokenizer_flan_t5.decode(outputs[0], skip_special_tokens=True)
        return corrected_sentence.strip()

    def check_grammar(self, sentence):
        """
        Check grammar using rule-based, ML-based, and FLAN-T5-based approaches.
        """
        # Step 1: Apply rules
        valid, message, corrected_sentence = self.apply_advanced_rules(sentence)
        if not valid:
            return f"Rule Violation: {message}\nSuggested Correction: {corrected_sentence}"

        # Step 2: Apply ML-based grammar checking
        ml_result = self.ml_grammar_checker(corrected_sentence)

        # Step 4: Apply FLAN-T5-based grammar checking
        flan_t5_result = self.flan_t5_grammar_checker(corrected_sentence)

        return {
            "Rule-Based Result": "Valid" if valid else "Invalid",
            "ML-Based Result": ml_result,
            "FLAN-T5-Based Result": flan_t5_result
        }
