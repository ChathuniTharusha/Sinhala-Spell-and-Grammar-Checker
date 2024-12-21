import pandas as pd
from nltk.metrics.distance import edit_distance

class SpellChecker:
    def __init__(self, dataset_path):
        # Load the dataset
        self.data = pd.read_csv(dataset_path, encoding='utf-8-sig')
        # Split into correct and incorrect words
        self.correct_words = self.data[self.data['Label'] == 1]['Word'].tolist()

    def correct_word(self, word):
        if word in self.correct_words:
            return word  # Do not modify if the word is already correct
        suggestions = sorted(self.correct_words, key=lambda w: edit_distance(word, w))
        if suggestions and edit_distance(word, suggestions[0]) <= 2:  # Adjust threshold as needed
            return suggestions[0]
        return word  # Return the original word if no close match is found
    def correct_text(self, sentence):
        # Correct each word in the sentence
        words = sentence.split()
        corrected_words = [self.correct_word(word) for word in words]
        return ' '.join(corrected_words)
