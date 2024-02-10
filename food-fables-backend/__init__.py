from flask import Flask
from flask_cors import CORS
from .routes import main  # Adjust the import path as necessary

def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/process_request/*": {"origins": "http://localhost:3000"}})
    app.register_blueprint(main)  # Register the Blueprint
    return app