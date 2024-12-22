from spell_checker import check_spelling_and_print
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


# Sample Usage
if __name__ == "__main__":
    rule_checker = RuleBasedGrammarChecker()
    input_text = ("අපි ගමට ගියෙය. ඔහු පාසලට ගියෙය. මම පොතක් කියවයි. ඇය ගෙදරට ආවා. ඔවුන් ගමනක් සූදානම් කළා. අපි ආහාර සකසා ගත්තෙමු. මගේ මිතුරා කාමරයේ සිටියා. ඔවුන් ගමක නතර විය. අපි වත්තෙ කටයුත්තක් කළා. ඇය පොතක් උගන්වමින් සිටියෙය.")

    results = rule_checker.check_grammar(input_text)

    for result in results:
        print(f"Original Sentence: {result['Original Sentence']}")
        print(f"Rule-Based Result: {result['Rule-Based Result']}")
        if result['Violation Message']:
            print(f"Violation Message: {result['Violation Message']}")
            print(f"Suggested Correction: {result['Suggested Correction']}")
        print("-" * 50)
check_spelling_and_print(input_text)