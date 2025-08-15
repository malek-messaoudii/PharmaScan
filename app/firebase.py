from firebase_admin import credentials, initialize_app, db
from app.config import FIREBASE_CREDENTIALS, FIREBASE_DB_URL

cred = credentials.Certificate(FIREBASE_CREDENTIALS)
initialize_app(cred, {'databaseURL': FIREBASE_DB_URL})

def store_quantity(quantity: int):
    db.reference('number/value').push(quantity)
