import pandas as pd
import ace_tools as tools
# Create a spell checker dataset with examples of correct and incorrect words
spell_checker_data = {
    "Word": [
        "කියවති", "කියවමි", "ගමට", "ගමනක්", "ගමන්ක්",
        "පාසල", "පාසලේ", "පාසල්", "අපි", "අපිම",
        "මම", "මමම", "ගියෙය", "ගියමි", "ගියමු"
    ],
    "Label": [
        0, 1, 1, 1, 0,  # 'කියවති' is incorrect, 'කියවමි' is correct, etc.
        1, 1, 1, 1, 0,  # 'අපිම' is incorrect but 'අපි' is correct.
        1, 0, 1, 1, 1  # 'මමම' is incorrect while others are correct.
    ],
    "Correction": [
        "කියවමි", "-", "-", "-", "ගමනක්",
        "-", "-", "-", "-", "අපි",
        "-", "මම", "-", "-", "-"
    ]
}

# Convert to DataFrame
spell_checker_dataset = pd.DataFrame(spell_checker_data)

# Save the dataset for future use
spell_checker_dataset_path = 'data/sinhala_spell_checker_dataset.csv'
spell_checker_dataset.to_csv(spell_checker_dataset_path, index=False, encoding='utf-8-sig')

# Display the dataset to the user


tools.display_dataframe_to_user(name="Sinhala Spell Checker Dataset", dataframe=spell_checker_dataset)
