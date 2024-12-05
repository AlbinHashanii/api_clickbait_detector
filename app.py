import os
from flask import Flask, request, jsonify, session
from flask.cli import load_dotenv
from flask_cors import CORS
import joblib
import requests
from transformers import pipeline
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
import tensorflow as tf
from flask_bcrypt import Bcrypt
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import random
import json
import logging
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import warnings
import string
from nltk.corpus import stopwords
import nltk
import string
warnings.filterwarnings("ignore", category=UserWarning)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
load_dotenv() 



device = 0 if torch.cuda.is_available() else -1 

app = Flask(__name__)
CORS(app, origins="*")


app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your_default_secret_key')


# Firebase initialization
firebase_sdk_path = os.environ.get('FIREBASE_ADMIN_SDK_PATH', './insightanalyze-firebase-adminsdk-ddwvh-57c48acc2b.json')
firebase_cred = credentials.Certificate(firebase_sdk_path)
firebase_admin.initialize_app(firebase_cred)
db = firestore.client()

# model_path = '/home/webintel/Desktop/InsightAnalyze/api_insight_analyze/data/clickbait_data/results/checkpoint-5370'
# tokenizer = BertTokenizer.from_pretrained(model_path)
# model = BertForSequenceClassification.from_pretrained(model_path)

model_path = os.path.join('data', 'clickbait_data', 'clickbait_detector_model.pkl')
vectorizer_path = os.path.join('data', 'clickbait_data', 'clickbait_vectorizer.pkl')

# Load the trained model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

nltk.download('stopwords')

def predict_the_clickbait(headline):
    """
    Predict the probability that a headline is clickbait.

    Args:
        headline (str): The headline to analyze.

    Returns:
        float: Probability that the headline is clickbait (between 0 and 1).
    """
    try:
        # Transform the input using the vectorizer
        headline_tfidf = vectorizer.transform([headline])
        
        # Predict the probability
        probability = model.predict_proba(headline_tfidf)[0][1]

        return probability
    except Exception as e:
        print(f"Error: {e}")
        return None





# OpenAI API key from environment




# # Clickbait tokenizer
# clickbait_tokenizer = Tokenizer(num_words=20000)
# clickbait_tokenizer.fit_on_texts(["Sample data to initialize"])

# BCRYPT
bcrypt = Bcrypt(app)

# List of cities and towns in Kosovo
kosovo_cities = [
    "Prishtina", "Prizren", "Peja", "Mitrovica", "Gjakova", "Gjilan",
    "Ferizaj", "Vushtrri", "Suharekë", "Lipjan", "Kaçanik", "Podujevë",
    "Rahovec", "Malishevë", "Skenderaj", "Drenas", "Obiliq", "Shtime",
    "Shtërpcë", "Viti", "Deçan", "Dragash", "Istog", "Kamenicë", "Klinë",
    "Zubin Potok", "Zvečan", "Novobërda", "Glogovac", "Leposaviq",
    "Graçanica", "Partesh", "Ranilug", "Pristina", "Kosove", "Kosovo"
]

@app.after_request
def add_cache_control(response):
    response.cache_control.no_cache = True
    response.cache_control.no_store = True
    response.cache_control.must_revalidate = True
    return response

# Pre-trained NER model
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)

def generate_password_hash(password):
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    return hashed_password

def check_password_hash(hashed_password, password):
    return bcrypt.check_password_hash(hashed_password, password)

def preprocess_text(text):
    sequences = clickbait_tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=200)
    return padded

def translate_text(text, target_language='en'):
    try:
        if len(text) > 5000:
            translated_text = ''
            chunks = [text[i:i+5000] for i in range(0, len(text), 5000)]
            translator = GoogleTranslator(source='auto', target=target_language)
            for chunk in chunks:
                translated_chunk = translator.translate(chunk)
                translated_text += translated_chunk
            return translated_text
        else:
            translator = GoogleTranslator(source='auto', target=target_language)
            return translator.translate(text)
    except Exception as e:
        raise Exception(f"Translation error: {e}")

def is_same_meaning(title, description):
    vectorizer = TfidfVectorizer().fit_transform([title, description])
    vectors = vectorizer.toarray()
    similarity = cosine_similarity(vectors)[0][1]
    return similarity > 0.4

def translate_and_identify_locations(title, description):
    translated_title = translate_text(title)
    translated_description = translate_text(description)
    locations_title = identify_locations(translated_title)
    locations_description = identify_locations(translated_description)
    combined_locations = list(set(locations_title + locations_description))
    same_meaning = is_same_meaning(translated_title, translated_description)
    return translated_title, translated_description, combined_locations, same_meaning

def scrape_it(url):

    USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Mozilla/5.0 (Windows NT 10.0; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:45.0) Gecko/20100101 Firefox/45.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_4) AppleWebKit/601.5.17 (KHTML, like Gecko) Version/9.1 Safari/601.5.17'
]
    try:
        response = requests.get(url,  headers = {
            'User-Agent': random.choice(USER_AGENTS)
        })
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        title = soup.find('title').get_text() if soup.find('title') else 'No title found'
        description = soup.find('meta', {'name': 'description'})['content'] if soup.find('meta', {'name': 'description'}) else 'No description available'
        image_url = soup.find('meta', {'property': 'og:image'})['content'] if soup.find('meta', {'property': 'og:image'}) else 'https://via.placeholder.com/1920x1080'
        author = 'Unknown Author'
        source = url.split('/')[2]
        post_date = 'Unknown Date'

        return {
            'title': title,
            'description': description,
            'author': author,
            'source': source,
            'post_date': post_date,
            'image_url': image_url,
            'url': url
        }
    except Exception as e:
        print(f"Error scraping URL: {e}")
        return None

def scrape_url(url):
    try:
        # Request the page
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract the title (falling back to OG title or first h1 tag)
        title = soup.find('title').get_text() if soup.find('title') else None
        if not title:
            og_title = soup.find('meta', {'property': 'og:title'})
            title = og_title['content'] if og_title else None
        if not title:
            h1_tag = soup.find('h1')
            title = h1_tag.get_text().strip() if h1_tag else 'No title found'

        # Extract the description (fallback to meta description, paragraphs, or OG description)
        description = ''
        meta_description = soup.find('meta', attrs={'name': 'description'})
        if meta_description:
            description = meta_description['content']
        elif soup.find('meta', {'property': 'og:description'}):
            description = soup.find('meta', {'property': 'og:description'}).get('content')
        else:
            paragraphs = soup.find_all('p')
            description = " ".join([p.get_text() for p in paragraphs[:5]])

        # Extract the image URL (fallback to OG image or placeholder)
        image_url = soup.find('meta', {'property': 'og:image'})
        image_url = image_url['content'] if image_url else 'https://via.placeholder.com/1920x1080'

        # Extract the author (fallbacks for common patterns)
        author = 'Unknown Author'
        author_meta = soup.find('meta', attrs={'name': 'author'})
        if author_meta:
            author = author_meta.get('content')
        else:
            author_tag = soup.find(class_='author') or soup.find('a', {'rel': 'author'}) or soup.find('span', {'class': 'byline'})
            if author_tag:
                author = author_tag.get_text().strip()

        # Extract the source (fallbacks to meta site_name or domain)
        source = 'Unknown Source'
        if soup.find('meta', {'property': 'og:site_name'}):
            source = soup.find('meta', {'property': 'og:site_name'})['content']
        elif soup.find('meta', {'name': 'application-name'}):
            source = soup.find('meta', {'name': 'application-name'})['content']
        else:
            source = url.split('/')[2]

        # Extract the publication date (fallback to time/date tags or unknown)
        post_date = 'Unknown Date'
        date_meta = soup.find('meta', {'property': 'article:published_time'})
        if date_meta:
            post_date = date_meta.get('content')
        else:
            date_tag = soup.find('time') or soup.find(class_='published') or soup.find('span', {'class': 'date'})
            if date_tag:
                post_date = date_tag.get_text().strip()

        # Return the extracted data
        return {
            'title': title,
            'description': description,
            'author': author,
            'source': source,
            'post_date': post_date,
            'image_url': image_url,
            'url': url
        }

    except Exception as e:
        print(f"Error scraping URL: {e}")
        return {
            'title': 'No title found',
            'description': description,
            'author': author,
            'source': 'Unknown Source',
            'post_date': post_date,
            'image_url': 'https://via.placeholder.com/1920x1080',
            'url': url
        }
    
def chat_with_custom_api(conversation_history, retries=3):
    try:
        # Ensure conversation history is below 256 characters
        conversation_text = ''.join([msg['content'] for msg in conversation_history])
        
        if len(conversation_text) > 256:
            print("Trimming conversation history to fit within 256 characters.")
            while len(''.join([msg['content'] for msg in conversation_history])) > 256:
                conversation_history.pop(0)  # Remove the oldest message
        
        # Correcting the header to properly include the Authorization token
        headers = {
            "Authorization": "Bearer XVPJOU2Z6WHLE7IZNKEIAMY7C2XADC7X",  # Replace with actual token
            "Content-Type": "application/json",
        }

        data = {
            "model": "gpt-4o",
            "messages": conversation_history,
            "temperature": 0.7,
            "max_tokens": 128,
        }
        
        # URL for the external API
        url = "https://api.wit.ai/message?q=Whats+the+weather+today"

        # Retry logic
        for attempt in range(retries):
            try:
                # Send the request
                response = requests.post(url, headers=headers, json=data)
                response.raise_for_status()  # Check for HTTP errors
                
                response_json = response.json()

                if 'choices' in response_json and len(response_json['choices']) > 0:
                    return response_json['choices'][0]['message']['content']
                else:
                    return "Unexpected API response format."

            except requests.exceptions.HTTPError as http_err:
                if 500 <= http_err.response.status_code < 600:
                    print(f"Server error: {http_err}. Retrying in 2 seconds...")
                    time.sleep(2)  # Wait for 2 seconds before retrying
                else:
                    print(f"HTTP error occurred: {http_err}")
                    print(f"Response content: {response.text}")
                    return f"HTTP error occurred: {http_err}"

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return f"An error occurred: {str(e)}"


def identify_locations(text):
    entities = ner(text)
    locations = [entity['word'] for entity in entities if entity['entity_group'] == 'LOC']
    return locations

def calculate_similarity(title, description):
    vectorizer = TfidfVectorizer().fit_transform([title, description])
    vectors = vectorizer.toarray()
    similarity = cosine_similarity(vectors)[0][1]
    return similarity > 0.7

def is_location_in_kosovo(locations):
    return any(location in kosovo_cities for location in locations)

def predict_clickbaitt(headline):
    preprocessed_headline = preprocess_text(headline)
    print("Preprocessed Headline:", preprocessed_headline)  # Debug preprocessed text
    
    # Predict using the clickbait model
    prediction = clickbait_model.predict(preprocessed_headline)
    print("Model Prediction:", prediction)  # Debug model output
    
    # Using threshold of 0.5 instead of rounding
    return 1 if prediction >= 0.575 else 0

def predict_clickbait(headline, threshold=0.5):
    # Tokenize the input headline
    test_encodings = tokenizer([headline], truncation=True, padding=True, max_length=128, return_tensors="pt")

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(**test_encodings)
        logits = outputs.logits  # Raw output scores from the model
        probabilities = torch.sigmoid(logits)  # Apply sigmoid to get probabilities

        # Check if the probability for "clickbait" (index 1) is greater than the threshold
        if probabilities[0][1] >= threshold:
            return 1  # Clickbait
        else:
            return 0  # Not clickbait




@app.route('/news_articles/<article_id>', methods=['GET'])
def get_news_article(article_id):
    try:
        doc_ref = db.collection('news_articles').document(article_id)
        doc = doc_ref.get()

        if doc.exists:
            article = doc.to_dict()
            return jsonify({
                'id': doc.id,
                'title': article.get('title'),
                'description': article.get('description'),
                'translated_title': article.get('translated_title'),
                'translated_description': article.get('translated_description'),
                'clickbait': 'Yes' if article.get('clickbait') else 'No',
                'locations': article.get('locations', []),
                'is_in_kosovo': 'Yes' if article.get('in_kosovo') else 'No',
                'similarity': 'Yes' if article.get('title_description_similarity') else 'No',
                'author': article.get('author', 'Anonymous'),
                'postDate': article.get('postDate', 'Unknown Date'),
                'imageUrl': article.get('imageUrl', 'https://via.placeholder.com/1920x1080'),
                'original_url': article.get('original_url', 'None')
            }), 200
        else:
            return jsonify({'error': 'Article not found'}), 404

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    article_url = data.get('article_url')
    true_classification = data.get('true_classification')

    if not article_url or not true_classification:
        return jsonify({'error': 'Article URL and true classification are required'}), 400

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(article_url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract Title
        title = soup.find('h1').get_text() if soup.find('h1') else 'No title found'

        # Extract Description
        description = soup.find('meta', {'name': 'description'})['content'] if soup.find('meta', {'name': 'description'}) else None
        if not description:
            paragraphs = soup.find_all('p')
            description = ' '.join([p.get_text() for p in paragraphs if p.get_text()])
        description = description.strip() if description else 'No description available'

        # Extract Image URL
        image_url = soup.find('meta', {'property': 'og:image'})['content'] if soup.find('meta', {'property': 'og:image'}) else 'https://via.placeholder.com/1920x1080'

        # Extract Author
        author = soup.find('meta', {'name': 'author'})['content'] if soup.find('meta', {'name': 'author'}) else 'Anonymous'

        # Extract Publication Date
        post_date = soup.find('meta', {'property': 'article:published_time'})['content'] if soup.find('meta', {'property': 'article:published_time'}) else 'Unknown Date'

        # Translate Title and Description
        translated_title = translate_text(title)
        translated_description = translate_text(description)

        # Identify Locations
        locations_title = identify_locations(translated_title)
        locations_description = identify_locations(translated_description)
        locations = list(set(locations_title + locations_description))

        # Check if the article is in Kosovo
        in_kosovo = is_location_in_kosovo(locations)

        # Calculate similarity between title and description
        title_description_similarity = calculate_similarity(translated_title, translated_description)

        # Clickbait Prediction Using the Model
        clickbait_probability = predict_the_clickbait(translated_title)

        # Determine if it's clickbait based on probability and other factors
        is_clickbait = False  # Default to not clickbait
        if in_kosovo:
            is_clickbait = False  # Not clickbait if in Kosovo
        elif title_description_similarity:
            is_clickbait = False  # Not clickbait if title and description are similar
        elif clickbait_probability >= 0.5:
            is_clickbait = True  # Clickbait if model predicts high probability

        # Prepare Article Data
        article_data = {
            'original_url': article_url,
            'title': title,
            'translated_title': translated_title,
            'description': description,
            'translated_description': translated_description,
            'clickbait_probability': round(clickbait_probability, 2),
            'clickbait': 'Yes' if is_clickbait else 'No',
            'locations': locations,
            'in_kosovo': 'Yes' if in_kosovo else 'No',
            'title_description_similarity': 'Yes' if title_description_similarity else 'No',
            'imageUrl': image_url,
            'author': author,
            'postDate': post_date,
            'true_classification': true_classification
        }

        # Log the Article Data
        logger.info("Processed Article Data:")
        logger.info(json.dumps(article_data, indent=4))

        # Add article data to the database
        article_ref = db.collection('news_articles').add(article_data)
        article_id = article_ref[1].id

        # Return Success Response
        return jsonify({
            'message': 'Article processed successfully',
            'article_id': article_id,
            'title': title,
            'description': description,
            'translated_title': translated_title,
            'clickbait_probability': round(clickbait_probability, 2),
            'clickbait': 'Yes' if is_clickbait else 'No',
            'locations': locations,
            'is_in_kosovo': 'Yes' if in_kosovo else 'No',
            'similarity': 'Yes' if title_description_similarity else 'No',
            'imageUrl': image_url,
            'author': author,
            'postDate': post_date,
            'true_classification': true_classification
        }), 200

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500



from urllib.parse import urlparse
from collections import Counter
from flask import jsonify

@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        # Query the database for all articles
        articles_ref = db.collection('news_articles')
        articles = articles_ref.stream()

        total_articles = 0
        correct_classifications = 0
        incorrect_classifications = 0
        domains = []

        for article in articles:
            article_data = article.to_dict()
            
            # Only process articles with a true classification available
            true_class = article_data.get('true_classification')
            if not true_class:
                continue  # Skip articles without true classification
            
            total_articles += 1
            predicted_class = 'clickbait' if article_data.get('clickbait') else 'not_clickbait'

            # Compare true classification and predicted classification
            if predicted_class == true_class:
                correct_classifications += 1
            else:
                incorrect_classifications += 1

            # Extract domain from original_url and store it
            original_url = article_data.get('original_url')
            if original_url:
                domain = urlparse(original_url).hostname
                domain = domain.replace('www.', '')  # Normalize the domain (remove www.)
                domains.append(domain)

        # Find top 3 most common domains
        domain_counts = Counter(domains)
        top_3_domains = domain_counts.most_common(3)

        # Calculate percentages
        if total_articles > 0:
            accuracy_percentage = (correct_classifications / total_articles) * 100
            incorrect_percentage = (incorrect_classifications / total_articles) * 100
        else:
            accuracy_percentage = 0
            incorrect_percentage = 0

        # Stats data
        stats_data = {
            'total_articles': total_articles,
            'correct_classifications': correct_classifications,
            'incorrect_classifications': incorrect_classifications,
            'accuracy_percentage': accuracy_percentage,
            'incorrect_percentage': incorrect_percentage,
            'top_3_sources': top_3_domains  # Include the top 3 most common sources
        }

        # Return stats
        return jsonify(stats_data), 200

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500




@app.route('/news_articles', methods=['GET'])
def get_news_articles():
    query = request.args.get('query', '')
    try:
        articles_ref = db.collection('news_articles')
        docs = articles_ref.stream()

        articles = []
        for doc in docs:
            article = doc.to_dict()
            if query.lower() in article.get('title', '').lower() or query.lower() in article.get('description', '').lower():
                articles.append({
                    'id': doc.id,
                    'title': article.get('title'),
                    'description': article.get('description'),
                    'imageUrl': article.get('imageUrl', 'https://via.placeholder.com/1920x1080'),
                    'link': article.get('url'),
                    'prediction': 'Clickbait' if article.get('clickbait') else 'Not Clickbait',
                    'source': article.get('source', 'Unknown Source'),
                    'author': article.get('author', 'Anonymous'),
                    'postDate': article.get('postDate', 'Unknown Date')
                })

        articles.sort(key=lambda x: x['postDate'], reverse=True)
        return jsonify(articles), 200

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

import re

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'error': 'Message is required'}), 400

    if 'conversation_history' not in session:
        session['conversation_history'] = [
            {"role": "system", "content": "You are a helpful assistant capable of everything."}
        ]
    session['conversation_history'].append({"role": "user", "content": user_message})

    # Use regex to extract URL from the message
    url = re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', user_message)
    
    if url:
        url = url.group(0)  # Extract the actual URL string
        article_data = scrape_url(url)  # Call your existing scrape function
        
        if article_data:
            title = article_data.get('title', 'No title found')
            description = article_data.get('description', 'No description available')

            # Translate the title and description and identify locations
            translated_title, translated_description, locations, same_meaning = translate_and_identify_locations(title, description)
            in_kosovo = is_location_in_kosovo(locations)

            # Determine if the article is clickbait
            is_clickbait = False if in_kosovo else predict_clickbait(translated_title)
            analysis = "Clickbait" if is_clickbait else "Not Clickbait"
            reason = "No location in Kosovo was found." if not in_kosovo else "Location in Kosovo was found."
            identified_locations = ', '.join(locations) if locations else 'None'

            # Generate detailed analysis
            detailed_analysis = generate_detailed_response(
                title=title,
                translated_title=translated_title,
                translated_description=translated_description,
                locations=identified_locations,
                is_in_kosovo=in_kosovo,
                clickbait_analysis=analysis,
                reason=reason
            )

            return jsonify({
                'message': user_message,
                'response': detailed_analysis,
                'details': {
                    'title': title,
                    'translated_title': translated_title,
                    'translated_description': translated_description,
                    'locations': locations,
                    'is_in_kosovo': bool(in_kosovo),
                    'clickbait_analysis': analysis if not in_kosovo else None,
                    'same_meaning': bool(same_meaning)
                }
            })
    
    # If no URL is detected or other paths, respond via custom API
    chatbot_response = chat_with_custom_api(session['conversation_history'])
    session['conversation_history'].append({"role": "assistant", "content": chatbot_response})

    return jsonify({'message': user_message, 'response': chatbot_response})


def generate_detailed_response(title, translated_title, translated_description, locations, is_in_kosovo, clickbait_analysis, reason):
    conversation_history = [
        {"role": "system", "content": "You are a helpful assistant capable of everything."},
        {"role": "user", "content": f"Analyze the following article title: '{title}'."},
        {"role": "assistant", "content": f"The title was translated as: '{translated_title}'."},
        {"role": "assistant", "content": f"Translated description: {translated_description}"},
        {"role": "assistant", "content": f"Identified locations: {locations}."},
        {"role": "assistant", "content": f"Is in Kosovo: {'Yes' if is_in_kosovo else 'No'}."},
        {"role": "assistant", "content": f"Clickbait analysis: {clickbait_analysis}."},
        {"role": "assistant", "content": f"Reason: {reason}."}
    ]
    return chat_with_custom_api(conversation_history)

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    first_name = data.get('first_name')  # Get first_name from request
    last_name = data.get('last_name')     # Get last_name from request
    email = data.get('email')             # Get email from request
    password = data.get('password')       # Get password from request

    if not email or not password or not first_name or not last_name:
        return jsonify({'error': 'Missing fields'}), 400

    user_ref = db.collection('Users').where('email', '==', email).get()

    if len(user_ref) > 0:
        return jsonify({'error': 'User already exists'}), 400

    hashed_password = generate_password_hash(password)
    user_data = {
        'email': email,
        'first_name': first_name,  # Correctly set first_name
        'last_name': last_name,     # Correctly set last_name
        'password': hashed_password,
        'created_at': firestore.SERVER_TIMESTAMP,
        'updated_at': firestore.SERVER_TIMESTAMP,
        'role': 'user'
    }

    # Add the user data to Firestore
    db.collection('Users').add(user_data)

    # Prepare the user object to return after registration
    user_info = {
        'email': email,
        'first_name': first_name,
        'last_name': last_name,
        'role': 'user'
    }

    return jsonify({'message': 'Registration successful', 'user': user_info}), 200


@app.route('/signin', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'error': 'Missing fields'}), 400

    # Query Firestore for user by email
    user_ref = db.collection('Users').where('email', '==', email).get()

    if not user_ref:
        return jsonify({'error': 'User does not exist'}), 400

    user = user_ref[0].to_dict()

    # Check if the password is correct
    if check_password_hash(user['password'], password):
        session['current_user'] = user
        return jsonify({'message': 'Login successful', 'user': user}), 200
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/profile', methods=['POST'])
def update_profile():
    print("Received request to update profile")  # Debug log
    try:
        # Get the profile data sent from the frontend
        data = request.get_json()
        print(f"Data received: {data}")  # Log the received data

        email = data.get('email')  # Get email from the frontend
        first_name = data.get('first_name')
        last_name = data.get('last_name')
        profile_picture_url = data.get('profile_picture_url')

        # Check for missing fields
        if not email or not first_name or not last_name:
            print("Missing fields in the request")  # Log if fields are missing
            return jsonify({'error': 'Missing fields'}), 400

        # Update Firestore document with the given email
        user_ref = db.collection('Users').document(email)
        user_ref.update({
            'first_name': first_name,
            'last_name': last_name,
            'profile_picture_url': profile_picture_url
        })

        # Fetch the updated user to return it to the frontend
        updated_user_snapshot = user_ref.get()
        if not updated_user_snapshot.exists:
            print("User not found after update")  # Log if user isn't found
            return jsonify({'error': 'User not found after update'}), 404

        updated_user = updated_user_snapshot.to_dict()
        print(f"Updated user data: {updated_user}")  # Log updated user data

        return jsonify({
            'message': 'Profile updated successfully',
            'updated_user': updated_user
        }), 200
    except Exception as e:
        print(f"Error updating profile: {str(e)}")  # Log the error
        return jsonify({'error': f'Could not update profile: {str(e)}'}), 500


    
@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')

    if not query:
        return jsonify({'error': 'Query is required'}), 400

    articles_ref = db.collection('news_articles')
    articles = articles_ref.stream()

    matching_articles = []
    for article in articles:
        article_data = article.to_dict()
        if query.lower() in article_data['title'].lower():
            matching_articles.append({
                'id': article.id,
                'title': article_data['title'],
                'description': article_data['description']
            })

    return jsonify(matching_articles), 200

@app.route('/sources', methods=['GET', 'POST'])
def manage_sources():
    try:
        if request.method == 'GET':
            sources_ref = db.collection('sources')
            sources = [doc.to_dict() for doc in sources_ref.stream()]
            return jsonify(sources), 200

        if request.method == 'POST':
            data = request.get_json()
            source_name = data.get('name')
            base_url = data.get('base_url')

            if not source_name or not base_url:
                return jsonify({'error': 'Source name and base URL are required'}), 400

            new_source = {
                'name': source_name,
                'base_url': base_url,
                'created_at': firestore.SERVER_TIMESTAMP
            }

            db.collection('sources').add(new_source)
            return jsonify({'message': 'Source added successfully'}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sources/<source_id>', methods=['DELETE'])
def delete_source(source_id):
    try:
        db.collection('sources').document(source_id).delete()
        return jsonify({'message': 'Source deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)