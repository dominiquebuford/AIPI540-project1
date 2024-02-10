from openai import OpenAI
client = OpenAI()

def generate_story(veggies):
    # Create the call to gpt for children's story creation
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a poetic, children's author, similar to Dr. Seuss. Make sure to cater to an audience of 8 years old and under."},
        {"role": "user", "content": f"Compose a children's story, approximately 2 paragraphs long, with {veggies} as the main character(s)."}
  ]
  )
    return completion.choices[0].message.content
