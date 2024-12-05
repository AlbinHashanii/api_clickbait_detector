import os

class Config:
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', 'your_default_secret_key')
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'your_default_openai_key')
    BASE_URL = os.environ.get('BASE_URL', 'https://api.openai.com')
    FIREBASE_ADMIN_SDK_PATH = os.environ.get('FIREBASE_ADMIN_SDK_PATH', './firebase-adminsdk.json')
