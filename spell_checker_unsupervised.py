import csv
from difflib import get_close_matches
from tqdm import tqdm  # Progress bar library

def load_dictionary(dictionary_csv):
    """Load Sinhala words from a CSV dictionary file."""
    with open(dictionary_csv, 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip the header
        return [row[0].strip() for row in reader]

def load_test_data(test_file):
    """Load test words from a file."""
    with open(test_file, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]

def spell_checker(word, dictionary, n_suggestions=3):
    """Find the closest matches for a word using edit distance."""
    return get_close_matches(word, dictionary, n=n_suggestions)

def process_test_data(test_file, dictionary, output_file):
    """Process test words, check their spelling, and save results."""
    test_words = load_test_data(test_file)
    results = []

    print("Processing test data...")
    for word in tqdm(test_words, desc="Spell checking progress", unit="word"):
        suggestions = spell_checker(word, dictionary)
        results.append({
            "Word": word,
            "Is Correct": word in dictionary,
            "Suggestions": ", ".join(suggestions) if suggestions else "None"
        })

    # Save results to a CSV file
    with open(output_file, 'w', encoding='utf-8', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["Word", "Is Correct", "Suggestions"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Spell check results saved to: {output_file}")
    print(f"Total words processed: {len(test_words)}")
    print(f"Words correctly spelled: {sum(1 for r in results if r['Is Correct'])}")
    print(f"Words requiring suggestions: {sum(1 for r in results if not r['Is Correct'])}")

# Paths to input files
dictionary_csv_file = "Datasets/dictionary_extracted_words.csv"  # Replace with your dictionary CSV file path
test_file = "Datasets/test_set_for_supervised.csv"  # Replace with your test data file path
output_file = "path_to_results.csv"  # Replace with your desired results CSV file path

# Main script workflow
# Step 1: Load the dictionary
sinhala_words = load_dictionary(dictionary_csv_file)
print(f"Dictionary loaded. Total words: {len(sinhala_words)}")

# Step 2: Process test data and spell check
process_test_data(test_file, sinhala_words, output_file)
