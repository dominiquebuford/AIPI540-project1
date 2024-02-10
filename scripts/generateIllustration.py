from openai import OpenAI
client = OpenAI()

def generate_illustration(story):
    response = client.images.generate(
      model="dall-e-3",
      prompt=story,
      size="1024x1024",
      quality="standard",
      n=1,
    )

    image_url = response.data[0].url
    return image_url
