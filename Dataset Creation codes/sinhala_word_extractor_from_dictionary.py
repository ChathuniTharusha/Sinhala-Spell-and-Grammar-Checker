import re
import csv

def extract_sinhala_words_from_file(input_file, output_file):
    sinhala_words = set()

    # Read the .txt file line by line
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            # Split by delimiters like '-' and '|'
            parts = re.split(r'[-|]', line)
            for part in parts:
                # Check if part contains Sinhala characters (Unicode range)
                if re.search(r'[\u0D80-\u0DFF]', part):
                    # Clean and normalize
                    word = part.strip()
                    sinhala_words.add(word)

    # Write the unique Sinhala words to a CSV file
    with open(output_file, 'w', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Sinhala Words"])  # Column header
        for word in sorted(sinhala_words):
            writer.writerow([word])

    print(f"Extraction complete. Sinhala words saved to: {output_file}")

# Input and output file locations
input_txt_file = "english-sinhala_dictionary.txt"  # Replace with your .txt file path
output_csv_file = "Datasets/dictionary_extracted_words.csv"  # Replace with your .csv file path

extract_sinhala_words_from_file(input_txt_file, output_csv_file)
