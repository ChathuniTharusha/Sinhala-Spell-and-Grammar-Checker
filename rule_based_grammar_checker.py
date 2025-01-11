from sklearn.metrics import accuracy_score, classification_report
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
        for subject, correct_ending in self.subject_rules.items():
            if sentence.startswith(subject):
                words = sentence.split()
                if len(words) > 1:  # Ensure there's a verb to process
                    verb = words[-1].rstrip(".")  # Remove period for processing
                    if not verb.endswith(correct_ending.rstrip(".")):
                        corrected_verb = verb[:-len(correct_ending.rstrip("."))] + correct_ending.rstrip(".")
                        corrected_sentence = " ".join(words[:-1] + [corrected_verb]) + "."
                        return False, (
                            f"If the sentence starts with '{subject}', the verb should end with '{correct_ending}'."
                        ), corrected_sentence

        return True, None, sentence

    def check_grammar(self, sentence):
        """
        Check grammar using rule-based approach.
        Processes multi-sentence inputs by splitting and validating each sentence.
        """
        # Split the input into individual sentences
        sentences = [s.strip() for s in sentence.split(".") if s.strip()]
        results = []

        for s in sentences:
            valid, message, corrected_sentence = self.apply_advanced_rules(s)
            results.append({
                "Original Sentence": s,
                "Rule-Based Result": "Valid" if valid else "Invalid",
                "Violation Message": message if not valid else None,
                "Suggested Correction": corrected_sentence if not valid else s
            })

        return results

    def evaluate_grammar_checker(self, dataset_path):
        """
        Evaluate the rule-based grammar checker on a labeled dataset.
        """
        # Load dataset
        data = pd.read_csv(dataset_path, encoding='utf-8-sig')

        # Initialize lists for ground truth and predictions
        ground_truth = data['Label'].apply(lambda x: "Valid" if x == "Correct" else "Invalid").tolist()
        predictions = []

        # Run the checker on each sentence
        for sentence in data['Sentence']:
            result = self.check_grammar(sentence)
            # Check if all sentences in the result are valid
            overall_result = "Valid" if all(r["Rule-Based Result"] == "Valid" for r in result) else "Invalid"
            predictions.append(overall_result)

        # Calculate accuracy and classification report
        accuracy = accuracy_score(ground_truth, predictions)
        report = classification_report(ground_truth, predictions, target_names=["Invalid", "Valid"])

        print(f"Rule-Based Accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:\n", report)


# Sample Usage
if __name__ == "__main__":
    rule_checker = RuleBasedGrammarChecker()

    # Test grammar checker with multi-sentence input
    input_text = (
        "අපි ගමට ගියෙය. ඔහු පාසලට ගියෙය. මම පොතක් කියවයි. "
        "අපි ආහාර සකසා ගත්තෙමු."
    )

    results = rule_checker.check_grammar(input_text)
    for result in results:
        print(f"Original Sentence: {result['Original Sentence']}")
        print(f"Rule-Based Result: {result['Rule-Based Result']}")
        if result['Violation Message']:
            print(f"Violation Message: {result['Violation Message']}")
            print(f"Suggested Correction: {result['Suggested Correction']}")
        print("-" * 50)

    # Evaluate rule-based grammar checker on labeled dataset
    dataset_path = "data/sinhala_grammar_checker_large_dataset.csv"  # Replace with your dataset path
    rule_checker.evaluate_grammar_checker(dataset_path)
