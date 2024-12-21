from spell_checker import SpellChecker
from grammar_checker import GrammarChecker

# Initialize the spell checker and grammar checker
spell_checker = SpellChecker("data/sinhala_spell_checker_dataset.csv")
grammar_checker = GrammarChecker("data/sinhala_grammar_checker_dataset.csv")

# Sample sentences for testing
sample_sentences = [
    "ඔහු පාසලට යනවය.",  # Incorrect spelling and grammar
    "ඔවුන් ගෙදරට පැමිනියි.",  # Correct spelling and grammar
    "ඇය පොතක් කියවන්නෙය.",  # Incorrect grammar
    "අපි ගමනකට ගියෙය.",  # Correct spelling and grammar
    "ඔහු පාසලට ගියේය."  # Correct spelling and grammar
]

# Process each sentence
for sentence in sample_sentences:
    print(f"Original: {sentence}")

    # Step 1: Correct spelling
    corrected_spelling = spell_checker.correct_text(sentence)
    print(f"After Spell Correction: {corrected_spelling}")

    # Step 2: Check grammar
    grammar_result = grammar_checker.check_grammar(corrected_spelling)
    print(f"Grammar Check Result: {grammar_result}")

    print("-" * 50)
