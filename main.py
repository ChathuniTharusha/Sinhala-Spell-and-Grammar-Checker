from spell_checker import SpellChecker
from grammar_checker import GrammarChecker


# Sample sentences for testing
sentences = [
    "අපි ගමනට ගියේය.",  # Should trigger the rule
    "අපි ගමනට ගියමු.",  # Should pass both rule-based and ML-based checks
    "ඔහු පාසලට ගියෙය."  # Should pass rule-based, evaluated by ML
]



# Initialize spell and grammar checkers
spell_checker = SpellChecker("data/sinhala_spell_checker_dataset.csv")
grammar_checker = GrammarChecker("data/sinhala_grammar_checker_dataset.csv")

# Process each sentence
for sentence in sentences:
    result = grammar_checker.check_grammar(sentence)
    print(f"Input: {sentence}\nResult: {result}\n{'-' * 50}")
