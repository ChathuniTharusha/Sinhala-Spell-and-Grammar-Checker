from transformers import T5Tokenizer, T5ForConditionalGeneration


class TransformerGrammarChecker:
    def __init__(self, model_name="t5-small", fine_tuned_model_path=None):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(
            fine_tuned_model_path if fine_tuned_model_path else model_name
        )

    def check_grammar(self, sentence):
        input_text = f"correct grammar: {sentence}"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        outputs = self.model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
        corrected_sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {
            "Transformer-Based Result": corrected_sentence
        }


# Sample Usage
if __name__ == "__main__":
    transformer_checker = TransformerGrammarChecker()
    sentences = [
        "අපි ගමනට ගියේය.",  # Incorrect
        "අපි ගමට ගියමු.",  # Correct
        "මම පොත කියවයි.",  # Incorrect
        "මම ගමට ගියමි."  # Correct
    ]

    for sentence in sentences:
        result = transformer_checker.check_grammar(sentence)
        print(f"Input Sentence: {sentence}")
        print(f"Transformer Result: {result['Transformer-Based Result']}")
        print("-" * 50)
