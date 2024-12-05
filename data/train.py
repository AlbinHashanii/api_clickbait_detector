import pandas as pd
import nltk
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from nltk.corpus import stopwords
import nltk

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load data from CSV
df = pd.read_csv('clickbait_data/train.csv')

# Preview the data
print("Data Sample:")
print(df.head())

# Preprocess titles and texts (remove punctuation, lowercase, remove stopwords)
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Ensure the text is a string before processing
    if not isinstance(text, str):
        text = str(text)  # Convert to string if it's not
    # Convert text to lowercase and remove punctuation
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    # Tokenize and remove stopwords
    words = nltk.word_tokenize(text)
    return ' '.join([word for word in words if word not in stop_words])


# Preprocess the title and text together
df['processed_text'] = df['title'] + " " + df['text']
df['processed_text'] = df['processed_text'].apply(preprocess_text)

# Split data into features (X) and target (y)
X = df['processed_text']
y = df['label'].apply(lambda x: 1 if x == 'clickbait' else 0)  # Assuming 'clickbait' and 'news' labels

# Vectorizing text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Limit the number of features for simplicity
X_tfidf = vectorizer.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train the model using Logistic Regression
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print(classification_report(y_test, y_pred))

# Save the model and vectorizer for future use
joblib.dump(model, 'clickbait_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("\nModel and Vectorizer have been saved.")

# Example function to predict if a new title is clickbait
def predict_clickbait(new_title):
    # Preprocess the new title
    processed_title = preprocess_text(new_title)
    # Transform the title into the same format used during training
    title_tfidf = vectorizer.transform([processed_title])
    # Predict if the title is clickbait
    prediction = model.predict(title_tfidf)
    return "Clickbait" if prediction == 1 else "Not Clickbait"

# Test the prediction function
new_title = "Why this Chinese economic policy will change everything!"
print("\nPrediction for new title:")
print(predict_clickbait(new_title))
