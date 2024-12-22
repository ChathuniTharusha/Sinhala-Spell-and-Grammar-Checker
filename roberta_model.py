import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class RoBERTaGrammarChecker:
    def __init__(self, dataset_path="data/sinhala_grammar_checker_large_dataset.csv"):
        # Load the dataset
        self.data = pd.read_csv(dataset_path, encoding='utf-8-sig')

        # Prepare the dataset
        self.X = self.data['Sentence']
        self.y = self.data['Label'].apply(lambda x: 1 if x == "Correct" else 0)

        # Initialize tokenizer and model
        self.tokenizer_roberta = RobertaTokenizer.from_pretrained("roberta-base")
        self.model_roberta = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

        # Fine-tune RoBERTa model on the grammar correction dataset
        self.train_roberta_model(self.X, self.y)

    def train_roberta_model(self, X_train, y_train):
        """
        Fine-tune RoBERTa for grammar checking.
        """

        class GrammarDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_len=128):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_len = max_len

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, item):
                text = str(self.texts[item])
                label = self.labels[item]

                encoding = self.tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    pad_to_max_length=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )

                return {
                    "input_ids": encoding["input_ids"].flatten(),
                    "attention_mask": encoding["attention_mask"].flatten(),
                    "label": torch.tensor(label, dtype=torch.long),
                }

        # Prepare the DataLoader
        train_dataset = GrammarDataset(X_train, y_train, self.tokenizer_roberta)
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        # Training Arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
        )

        # Initialize the Trainer
        trainer = Trainer(
            model=self.model_roberta,
            args=training_args,
            train_dataset=train_dataset,
        )

        # Train the model
        trainer.train()

    def roberta_grammar_checker(self, sentence):
        """
        Use RoBERTa to check the grammar of a sentence.
        """
        inputs = self.tokenizer_roberta.encode_plus(
            sentence, add_special_tokens=True, max_length=128, pad_to_max_length=True, return_tensors="pt"
        )
        outputs = self.model_roberta(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

        return "Correct" if predicted_class == 1 else "Incorrect"

    def evaluate_accuracy(self, X_test, y_test):
        """
        Evaluate and print accuracy for RoBERTa grammar checker.
        """
        correct = 0
        total_sentences = len(X_test)

        for sentence, label in zip(X_test, y_test):
            result = self.roberta_grammar_checker(sentence)
            correct += 1 if (result == "Correct" and label == 1) or (result == "Incorrect" and label == 0) else 0

        accuracy = correct / total_sentences * 100
        print(f"RoBERTa Grammar Checker Accuracy: {accuracy:.2f}%")
