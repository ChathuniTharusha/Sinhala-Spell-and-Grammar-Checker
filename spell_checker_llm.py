import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
import torch

# Step 1: Load the labeled dataset from CSV
labeled_data_path = "Datasets/test_set_for_supervised.csv"  # Provide the path to your labeled CSV file
labeled_data = pd.read_csv(labeled_data_path)

# Load the unlabeled dataset from CSV
unlabeled_data_path = "Datasets/dictionary_extracted_words.csv"  # Provide the path to your unlabeled CSV file

unlabeled_data = pd.read_csv(unlabeled_data_path)

# Step 2: Convert the labeled data to Hugging Face dataset
labeled_dataset = Dataset.from_pandas(labeled_data)

# Create the unlabeled dataset with pseudo-labels (all labels are 1 since the words are correct)
# Assuming the unlabeled data CSV has a column named 'word' that contains the words
unlabeled_dataset = Dataset.from_dict({
    'word': unlabeled_data['word'].tolist(),
    'label': [1] * len(unlabeled_data)  # All labeled as correct
})

# Combine both datasets (labeled and pseudo-labeled)
combined_dataset = labeled_dataset.concatenate(unlabeled_dataset)

# Step 3: Load a pre-trained model and tokenizer
model_name = "bert-base-multilingual-cased"  # You can replace with a Sinhala model if available
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 4: Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['word'], padding="max_length", truncation=True)

tokenized_datasets = combined_dataset.map(tokenize_function, batched=True)

# Format the dataset for PyTorch
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
train_dataset = tokenized_datasets.shuffle(seed=42).select([i for i in range(0, len(tokenized_datasets))])

# Step 5: Initialize the model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Step 6: Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=16,   # batch size for evaluation
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch",     # evaluate after each epoch
    save_strategy="epoch",           # save model after each epoch
)

# Step 7: Initialize the Trainer
trainer = Trainer(
    model=model,                         # the model to be trained
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    tokenizer=tokenizer,                 # tokenizer
)

# Step 8: Train the model
trainer.train()

# Step 9: Evaluate the model (optional, for evaluation purposes)
eval_results = trainer.evaluate()

# Step 10: Make predictions on new words
new_words = ["අංකම", "අංක එක", "අංක මුහුණත"]  # Example words to test the model
input_encodings = tokenizer(new_words, truncation=True, padding=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(**input_encodings)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

# Display predictions (1 = Correct, 0 = Incorrect)
for word, pred in zip(new_words, predictions):
    print(f"Word: {word}, Prediction: {'Correct' if pred.item() == 1 else 'Incorrect'}")
