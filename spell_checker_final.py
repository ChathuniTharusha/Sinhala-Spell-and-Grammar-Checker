import csv
import difflib

# Function to load the dictionary from a CSV file
def load_dictionary_from_csv(file_path):
    dictionary = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header if there is one
        for row in reader:
            word = row[0].strip()  # Assuming the word is in the first column
            dictionary.add(word)
    return dictionary

# Function to suggest corrections for a misspelled word
def suggest_correction(misspelled_word, dictionary):
    # Use difflib to find the closest matches in the dictionary
    suggestions = difflib.get_close_matches(misspelled_word, dictionary, n=5, cutoff=0.8)
    return suggestions

# Function to process the paragraph and suggest corrections
def process_paragraph(paragraph, dictionary):
    # Split the paragraph into words using space as the separator
    words = paragraph.split()

    misspelled_words = {}

    for word in words:
        if word not in dictionary:
            suggestions = suggest_correction(word, dictionary)
            if suggestions:
                misspelled_words[word] = suggestions

    return misspelled_words

# Function to print the suggestions for a given paragraph
def check_spelling_and_print(paragraph, dictionary_file='Datasets/preprocessed_dictionary.csv'):
    # Load the dictionary from the CSV file
    dictionary = load_dictionary_from_csv(dictionary_file)

    # Process the paragraph and get misspelled words with suggestions
    misspelled_words = process_paragraph(paragraph, dictionary)

    if misspelled_words:
        print("Misspelled words and suggestions:")
        for word, suggestions in misspelled_words.items():
            print(f"Word: {word}")
            print("Did you mean:")
            for suggestion in suggestions:
                print(f"  - {suggestion}")
    else:
        print("No misspelled words found. The paragraph seems correct.")
