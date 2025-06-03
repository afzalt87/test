import logging
from openai import OpenAI
import os
import base64
import json
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
OPENAI_MODEL = "gpt-4.1"


class Llm:
    def __init__(self):
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully.")
        except Exception as e:
            logger.exception(
                f"Failed to initialize OpenAI client: {e}. API calls will fail."
            )
            self.client = None

    def call_with_text(self, system_prompt, user_prompt):
        # TODO add call with text
        return

    def call_with_image(self, system_prompt, user_prompt, img_path):
        if not self.client:
            logger.error("OpenAI client not initialized.")
            return None
        try:
            with open(img_path, "rb") as image_file:
                base64_image = base64.b64encode(
                    image_file.read()).decode("utf-8")

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                },
            ]
            completion = self.client.chat.completions.create(
                model=OPENAI_MODEL, messages=messages, temperature=0.1, max_tokens=1000
            )
            response = completion.choices[0].message.content
            print(response)
            response_json = json.loads(response)

            # TODO return in format like {"condition1": "response1", "condition2": "response_2"}
            return response_json
        except Exception as e:
            logger.exception(f"Image evaluation failed: {e}")
            return None
