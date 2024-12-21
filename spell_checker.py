import pandas as pd
from nltk.metrics.distance import edit_distance


class SpellChecker:
    def __init__(self, dataset_path):
        # Load the dataset
        self.data = pd.read_csv(dataset_path, encoding='utf-8-sig')
        # Extract correct words
        self.correct_words = self.data[self.data['Label'] == 1]['Word'].tolist()

    def correct_word(self, word):
        if word in self.correct_words:
            return word  # Word is already correct
        suggestions = sorted(self.correct_words, key=lambda w: edit_distance(word, w))
        if suggestions and edit_distance(word, suggestions[0]) <= 2:  # Threshold for correction
            return suggestions[0]
        return word  # Return the original word if no suitable correction is found

    def correct_text(self, sentence):
        words = sentence.split()
        corrected_words = [self.correct_word(word) for word in words]
        return ' '.join(corrected_words)
