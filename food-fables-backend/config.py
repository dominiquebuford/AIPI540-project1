import os
from dotenv import load_dotenv

class Config:
    DEBUG = False

    # Get the current directory
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # Specify the path to the .env file
    env_path = os.path.join(current_directory, '.env')
    # Load the environment variables from the .env file
    load_dotenv(dotenv_path=env_path)

    # Sensitive configs
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

class DevelopmentConfig(Config):
    DEBUG = True
    # Other development-specific configs