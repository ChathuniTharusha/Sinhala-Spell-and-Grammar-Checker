from spell_checker import SpellChecker
from grammar_checker import GrammarChecker

# Sample sentences for testing
sentences = [
    "අපි ගමනට ගියේය.",  # Should trigger the rule
    "අපි ගමට ගියමු.",  # Should pass both rule-based and ML-based checks
    "මම පොතක් කියවයි.",  # Should pass rule-based, evaluated by ML
    "ඇය පොතක් උගන්වමින් සිටියෙය.",  # Should be evaluated
    "අපි ආහාරය සකසා ගත්තෙමු."  # Should pass rule-based and ML-based checks
]

# Initialize spell and grammar checkers
spell_checker = SpellChecker("data/sinhala_spell_checker_dataset.csv")
grammar_checker = GrammarChecker("data/sinhala_grammar_checker_large_dataset.csv")

# Process each sentence
for sentence in sentences:
    # Correct spelling (if applicable)
    corrected_sentence = spell_checker.correct_text(sentence)
    print(f"Corrected Sentence: {corrected_sentence}")

    # Check grammar
    result = grammar_checker.check_grammar(corrected_sentence)
    print(f"Input: {corrected_sentence}\nResult: {result}\n{'-' * 50}")
