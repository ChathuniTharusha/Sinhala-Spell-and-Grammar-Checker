import pandas as pd
import random


# Function to generate random correct and incorrect Sinhala sentences
def generate_sinhala_sentences(num_correct, num_incorrect):
    correct_sentences = [
        "අපි ගමට ගියෙමු.",
        "ඔහු පාසලට ගියෙය.",
        "මම පොතක් කියවමි.",
        "ඇය ගෙදරට ආවා.",
        "ඔවුන් ගමක නතර විය.",
        "අපි ආහාර සකසා ගත්තෙමු.",
        "මගේ මිතුරා කාමරයේ සිටියේය.",
        "ඔවුන් ගමනක් සූදානම් කළා.",
        "අපි වත්තෙ කටයුත්තක් කළා.",
        "ඇය පොතක් උගන්වමින් සිටියා."
    ]

    incorrect_sentences = [
        "අපි ගමට ගියෙය.",
        "ඔහු පාසලට ගියා.",
        "මම පොතක් කියවයි.",
        "ඇය ගෙදරට යනව.",
        "ඔවුන් ගමනක නතරවිය.",
        "අපි ආහාර සකසා ගත්තෙය.",
        "මගේ මිතුරා කාමරයේ සිටියා.",
        "ඔවුන් ගමනක් සූදානම් කළේය.",
        "අපි වත්තෙ කටයුත්තක් කරා.",
        "ඇය පොතක් උගන්වමින් සිටියෙය."
    ]

    # Expand to the required number of sentences by random sampling
    correct = random.choices(correct_sentences, k=num_correct)
    incorrect = random.choices(incorrect_sentences, k=num_incorrect)

    # Combine and shuffle the sentences
    sentences = correct + incorrect
    labels = ["Correct"] * num_correct + ["Incorrect"] * num_incorrect
    combined = list(zip(sentences, labels))
    random.shuffle(combined)
    return combined


# Generate 1000 sentences: 500 correct and 500 incorrect
dataset = generate_sinhala_sentences(500, 500)

# Create a DataFrame
df_large = pd.DataFrame(dataset, columns=["Sentence", "Label"])

# Save the DataFrame as a CSV file
file_path_large = "sinhala_grammar_checker_large_dataset.csv"
df_large.to_csv(file_path_large, index=False, encoding="utf-8-sig")

print(f"Dataset saved as {file_path_large}")
