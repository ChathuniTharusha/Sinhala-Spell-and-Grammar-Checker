from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForSequenceClassification, AutoTokenizer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd


class RuleBasedGrammarChecker:
    def __init__(self):
        # Define rules for specific subjects
        self.subject_rules = {
            "අපි": "මු.",
            "මම": "මි.",
            "ඔහු": "යි.",
            "ඇය": "යි.",
            "ඔවුන්": "යි."
        }

    def apply_advanced_rules(self, sentence):
        """
        Apply grammar rules based on subject and verb endings.
        """
        # Check if the sentence starts with a defined subject
        for subject, correct_ending in self.subject_rules.items():
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

    def check_grammar(self, sentence):
        """
        Check grammar using rule-based approach.
        """
        valid, message, corrected_sentence = self.apply_advanced_rules(sentence)
        return {
            "Rule-Based Result": "Valid" if valid else "Invalid",
            "Suggested Correction": corrected_sentence if not valid else None
        }


# Sample Usage
if __name__ == "__main__":
    rule_checker = RuleBasedGrammarChecker()
    sentences = [
        "අපි ගමනට ගියේය.",  # Rule Violation
        "අපි ගමට ගියමු.",  # Valid
        "මම පොත කියවයි.",  # Rule Violation
        "මම ගමට ගියමි."  # Valid
    ]

    for sentence in sentences:
        result = rule_checker.check_grammar(sentence)
        print(f"Input Sentence: {sentence}")
        print(f"Result: {result}")
        print("-" * 50)
