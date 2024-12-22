import re
import pandas as pd
from difflib import get_close_matches


# Load correct words from CSV file (replace this with the actual path to your correct words dataset)
def load_correct_words(correct_words_path):
    correct_words_df = pd.read_csv(correct_words_path, header=None)
    return set(correct_words_df[0].tolist())  # Assuming the words are in the first column


# Define spelling correction rules (example)
def apply_rules(word, correct_words):
    # Rule 1: Remove duplicate consonants (e.g., "අංකන" → "අංකන")
    word = re.sub(r'(.)\1', r'\1', word)  # Removes consecutive duplicate characters

    # Rule 2: Remove incorrect vowel shortening or add missing vowels (e.g., "අංකන" → "අංකය")
    word = re.sub(r'අංකන$', 'අංකය', word)  # Example of a specific correction

    # Rule 3: Fix common phonetic errors (e.g., "අංකු" → "අංක")
    word = re.sub(r'අංකු$', 'අංක', word)

    # Rule 4: Handle common typographical errors (e.g., "අංකස" → "අංක")
    word = re.sub(r'අංකස$', 'අංක', word)

    # If the word is still incorrect, use the correct words dataset to suggest a close match
    if word not in correct_words:
        word = get_best_match(word, correct_words)

    return word


# Function to get the best match for an incorrect word from the correct words list
def get_best_match(word, correct_words):
    # Get close matches to the incorrect word based on the correct words list
    closest_matches = get_close_matches(word, correct_words, n=1, cutoff=0.8)  # Adjust cutoff for sensitivity
    if closest_matches:
        return closest_matches[0]
    else:
        return word  # If no close match, return the word as is


# Function to load dataset and apply spell correction
def process_dataset(dataset_path, correct_words):
    with open(dataset_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    corrected_words = []
    original_words = []
    labels = []

    # Skip the header row (if there is one) and process the rest
    for line in lines[1:]:  # Skip the first line (header)
        word, label = line.strip().split(",")
        try:
            labels.append(int(label))  # Assuming label is '1' for correct and '0' for incorrect
        except ValueError:
            print(f"Skipping line with invalid label: {line.strip()}")
            continue  # Skip any lines where the label is not an integer

        corrected_word = apply_rules(word, correct_words)
        corrected_words.append(corrected_word)
        original_words.append(word)

    return original_words, corrected_words, labels


# Function to calculate accuracy
def calculate_accuracy(original_words, corrected_words, labels):
    correct_count = 0
    total_count = len(original_words)

    for original, corrected, label in zip(original_words, corrected_words, labels):
        if label == 1:  # Word is originally correct
            if original == corrected:
                correct_count += 1
        else:  # Word is originally incorrect
            if original != corrected:
                correct_count += 1

    accuracy = correct_count / total_count * 100
    return accuracy


# Main
if __name__ == "__main__":
    # Path to your correct words dataset CSV file (replace with your actual file path)
    correct_words_path = "Datasets/dictionary_extracted_words.csv"

    # Load the correct words dataset
    correct_words = load_correct_words(correct_words_path)

    # Sample dataset: Replace this with the path to your dataset with labels (words to be corrected)
    dataset_path = "Datasets/test_set_for_supervised.csv"

    # Process the dataset and apply the spell checker
    original_words, corrected_words, labels = process_dataset(dataset_path, correct_words)

    # Print corrected words
    for original, corrected in zip(original_words, corrected_words):
        print(f"Original: {original} -> Corrected: {corrected}")

    # Calculate accuracy
    accuracy = calculate_accuracy(original_words, corrected_words, labels)
    print(f"Accuracy: {accuracy:.2f}%")
