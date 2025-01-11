import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_scheduler
from tqdm import tqdm

# Define Dataset Class
class SinhalaGrammarDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = str(self.sentences[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

# Load Dataset
def load_dataset(path):
    data = pd.read_csv(path, encoding='utf-8-sig')
    sentences = data['Sentence'].tolist()
    labels = data['Label'].apply(lambda x: 1 if x == "Correct" else 0).tolist()
    return sentences, labels

# Fine-Tune Transformer Model
def fine_tune_transformer(dataset_path, model_name="bert-base-multilingual-cased", epochs=3, batch_size=16, learning_rate=2e-5):
    sentences, labels = load_dataset(dataset_path)
    train_size = int(0.8 * len(sentences))
    train_sentences = sentences[:train_size]
    train_labels = labels[:train_size]
    test_sentences = sentences[train_size:]
    test_labels = labels[train_size:]

    tokenizer = BertTokenizer.from_pretrained(model_name)
    train_dataset = SinhalaGrammarDataset(train_sentences, train_labels, tokenizer)
    test_dataset = SinhalaGrammarDataset(test_sentences, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(train_loader) * epochs
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            attention_mask = batch["attention_mask"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            labels = batch["label"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            progress_bar.set_postfix(loss=loss.item())

    model.save_pretrained("sinhala_grammar_nlp_model")
    tokenizer.save_pretrained("sinhala_grammar_nlp_model")

    evaluate_model(model, test_loader, test_labels)

# Evaluate Model
def evaluate_model(model, test_loader, test_labels):
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            attention_mask = batch["attention_mask"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(test_labels, all_predictions)
    report = classification_report(test_labels, all_predictions, target_names=["Incorrect", "Correct"])

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:\n", report)

# NLP Grammar Checker with Rule-Based Suggestions
class NLPGrammarChecker:
    def __init__(self, model_path="sinhala_grammar_nlp_model"):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model = self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.subject_rules = {
            "අපි": "මු.",
            "මම": "මි.",
            "ඔහු": "යි.",
            "ඇය": "යි.",
            "ඔවුන්": "යි."
        }

    def predict(self, sentence):
        encoding = self.tokenizer(
            sentence,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        attention_mask = encoding["attention_mask"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
        return "Correct" if prediction == 1 else "Incorrect"

    def suggest_correction(self, sentence):
        for subject, correct_ending in self.subject_rules.items():
            if sentence.startswith(subject):
                words = sentence.split()
                if len(words) > 1:
                    verb = words[-1].rstrip(".")
                    if not verb.endswith(correct_ending.rstrip(".")):
                        corrected_verb = verb[:-len(correct_ending.rstrip("."))] + correct_ending.rstrip(".")
                        return " ".join(words[:-1] + [corrected_verb]) + "."
        return "No suggestion available."

    def check_sentences(self, sentences):
        results = []
        for sentence in sentences:
            prediction = self.predict(sentence)
            suggestion = self.suggest_correction(sentence) if prediction == "Incorrect" else "No correction needed."
            results.append({
                "Sentence": sentence,
                "Prediction": prediction,
                "Suggestion": suggestion
            })
        return results

# Main Script
if __name__ == "__main__":
    dataset_path = "data/sinhala_grammar_checker_large_dataset.csv"
    fine_tune_transformer(dataset_path)

    nlp_checker = NLPGrammarChecker()
    test_sentences = [
        "අපි ගමට ගියෙය.",
        "ඔහු පාසලට ගියෙය.",
        "මම පොතක් කියවයි.",
        "ඇය ගෙදරට ආවා."
    ]
    results = nlp_checker.check_sentences(test_sentences)
    for res in results:
        print(res)
