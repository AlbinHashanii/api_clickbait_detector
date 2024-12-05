from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the tokenizer and model
model_path = '/home/webintel/Desktop/InsightAnalyze/clickbait_detector_api/data/clickbait_data/results/checkpoint-5370'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Example test data
test_headlines = [
    "You Won't Believe What Happened Next!",  # Likely clickbait
    "Local Man Discovers Rare Species in His Backyard",  # Not clickbait
    "Breaking: Major Earthquake Strikes City Center",  # Neutral
    "At least 24 killed by the bomb blast at the train station",
    "A bag with explosives was found at a station"
]

# Tokenize the test data
test_encodings = tokenizer(test_headlines, truncation=True, padding=True, max_length=128, return_tensors="pt")

# Define a custom threshold
threshold = 0.5

# Predict
model.eval()
with torch.no_grad():
    outputs = model(**test_encodings)
    logits = outputs.logits  # Raw output scores from the model

# Custom logic for close predictions
for headline, logit in zip(test_headlines, logits):
    logit_list = logit.tolist()
    score_diff = abs(logit_list[0] - logit_list[1])
    prediction = (
        1 if logit_list[1] > logit_list[0] or score_diff < threshold else 0
    )  # Classify as 1 if logits are close or class 1 has a higher score

    print(f"Headline: {headline}")
    print(f"Logits: {logit_list}")
    print(f"Prediction (custom logic): {prediction}")
    print("-" * 50)
