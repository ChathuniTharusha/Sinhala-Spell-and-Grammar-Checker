from spell_checker import SpellChecker
from grammar_checker import GrammarChecker


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


# Sample sentences for testing
sentences = [
    "අපි ගමනට ගියේය.",  # Rule should trigger
    "අපි ගමට ගියමු.",  # Rule should pass
    "මම පොත කියවයි.",  # Rule should trigger
    "මම ගමට ගියමි.",  # Rule should pass
    "ඇය පොත කියවමි."  # Should be processed by ML-based grammar checker
]

# Initialize the grammar checker
grammar_checker = GrammarChecker("data/sinhala_grammar_checker_large_dataset.csv")



# Process sentences
for sentence in sentences:
    print(f"Input Sentence: {sentence}")
    result = grammar_checker.check_grammar(sentence)
    print(f"Output: {result}\n")
