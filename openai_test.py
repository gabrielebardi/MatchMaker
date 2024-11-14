from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import Optional

# Load environment variables from .env file
load_dotenv()

# Retrieve your API key
api_key = os.getenv("OPENAI_API_KEY")

def get_api_response(prompt: str) -> Optional[str]:
    text: Optional[str] = None
    try:
        # Instantiate the OpenAI client
        client = OpenAI(api_key=api_key)

        # Use the client to create a chat completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=1.2,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.7,
            stop=['Human:', 'AI:']
        )

        # Extract the response text
        text = response.choices[0].message.content
    except Exception as e:
        print('ERROR:', e)
    return text

# Example usage
prompt = "Describe a sunset over a mountain range and generate a pickup line based on it."
print(get_api_response(prompt))