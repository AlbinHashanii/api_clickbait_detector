import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase Admin
cred = credentials.Certificate('./insightanalyze-firebase-adminsdk-ddwvh-57c48acc2b.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

def get_next_id(collection_name):
    # Get all documents in the collection to find the highest ID
    collection_ref = db.collection(collection_name)
    docs = collection_ref.stream()
    max_id = 0
    for doc in docs:
        doc_id = int(doc.id)
        if doc_id > max_id:
            max_id = doc_id
    return max_id + 1

def create_document_with_integer_id(collection_name, data):
    next_id = get_next_id(collection_name)
    document_ref = db.collection(collection_name).document(str(next_id))
    document_ref.set(data)
    print(f"Document with ID {next_id} created in '{collection_name}' collection.")

def initialize_collections():
    # Create some example data for Users collection
    create_document_with_integer_id('Users', {
        'first_name': 'Alice', 
        'last_name': 'Smith', 
        'email': 'alice@example.com', 
        'password': 'hashed_password_here',
        'created_at': firestore.SERVER_TIMESTAMP, 
        'updated_at': firestore.SERVER_TIMESTAMP,
        'role': 'admin',
        'profile_picture_url': 'http://example.com/profile/alice.jpg',
        'phone_number': '+123456789',
        'country': 'Kosovo',
        'language': 'en',
        'timezone': 'Europe/Pristina',
        'is_active': True,
        'email_verified': True
    })

    create_document_with_integer_id('Users', {
        'first_name': 'Bob', 
        'last_name': 'Johnson', 
        'email': 'bob@example.com', 
        'password': 'hashed_password_here',
        'created_at': firestore.SERVER_TIMESTAMP, 
        'updated_at': firestore.SERVER_TIMESTAMP,
        'role': 'user',
        'profile_picture_url': 'http://example.com/profile/bob.jpg',
        'phone_number': '+987654321',
        'country': 'Kosovo',
        'language': 'en',
        'timezone': 'Europe/Pristina',
        'is_active': True,
        'email_verified': True
    })

    # Example data for Articles collection
    create_document_with_integer_id('Articles', {
        'title': 'Introduction to Firestore', 
        'content': 'Content here...', 
        'authorId': 1,
        'created_at': firestore.SERVER_TIMESTAMP,
        'updated_at': firestore.SERVER_TIMESTAMP,
        'tags': ['Firestore', 'Database', 'NoSQL'],
        'status': 'published',
        'views': 0,
        'likes_count': 0,
        'comments_count': 0,
        'featured_image_url': 'http://example.com/images/firestore.jpg',
        'language': 'en',
        'is_featured': False
    })

    create_document_with_integer_id('Articles', {
        'title': 'Advanced Firestore Queries', 
        'content': 'Content here...', 
        'authorId': 2,
        'created_at': firestore.SERVER_TIMESTAMP,
        'updated_at': firestore.SERVER_TIMESTAMP,
        'tags': ['Firestore', 'Queries', 'Advanced'],
        'status': 'draft',
        'views': 0,
        'likes_count': 0,
        'comments_count': 0,
        'featured_image_url': 'http://example.com/images/firestore_advanced.jpg',
        'language': 'en',
        'is_featured': False
    })

    # Example data for Notifications collection
    create_document_with_integer_id('Notifications', {
        'userId': 1, 
        'content': 'You have a new comment on your article.', 
        'timestamp': firestore.SERVER_TIMESTAMP,
        'is_read': False,
        'type': 'comment',
        'related_resource_id': 1
    })

    create_document_with_integer_id('Notifications', {
        'userId': 2, 
        'content': 'Your article was liked.', 
        'timestamp': firestore.SERVER_TIMESTAMP,
        'is_read': False,
        'type': 'like',
        'related_resource_id': 1
    })

if __name__ == '__main__':
    initialize_collections()  # Run this to initialize the collections and add example data
