from .config import Config
import openai as OpenAI
from flask import current_app as app

def generate_ai_response(child_name, results):
    """
    Given the child_name and image classification results, generates a response using OpenAI's GPT-3.5 Turbo model.

    :param child_name: Name of the child.
    :param results: List of dictionaries with image classification labels.
    :return: A string representing the response.
    """
    # Extract characters (fruits/vegetables) from results
    characters = ', '.join([result for result in results])

    # Create the prompt
    context = ("You are a fun, engaging, lighthearted children's story generator. All characters are fruits and/or vegetables except for the main character, a child."
                 "Use the name of the fruit or vegetable as the last name of the character and make up a first name. For example, Bobby Broccoli or Sally Strawberry."
                 "Your stories are about friendship, adventure, and learning. They are suitable for children ages 3-10."
                 "Your stories are not scary or violent. They are always positive and uplifting."
                 "You should only make stories with the characters you are given. You should not add any new characters."
                 "Your stories are no longer than 300 words long.")
    prompt = (
        f"Create a story with the following elements:"
        f"Main Character, a human: {child_name}. "
        f"Other characters: {characters}."
    )

    try:
        # Connect to OpenAI with the API key from the Flask app configuration
        client = OpenAI.Client(api_key=Config.OPENAI_API_KEY)

        # Generate the completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Adjust the model as necessary
            messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": prompt}
            ]
        )

        # Extract the story text from the response
        story = response.choices[0].message.content

        # Check if the story is empty
        if not story:
            story = "I am sorry, but I cannot generate a story at this time."

    except Exception as e:
        print(f"Failed to generate response: {str(e)}")
        story = "An error occurred while generating the story."

    return story