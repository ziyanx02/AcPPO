import pickle
from openai import AzureOpenAI

from reward_tuning.API_key import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT
from reward_tuning.example import RESPONSE_SAMPLE

class Client:
    def __init__(self, temperature=0.5, disable=False):
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY, 
            api_version="2024-02-01",
            azure_endpoint=AZURE_OPENAI_ENDPOINT, # os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        self.deployment_name = "gpt-4o"
        self.history = []
        self.temperature = temperature
        self.disable = disable

    def response(self, messages):    
        if self.disable:
            return RESPONSE_SAMPLE
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            temperature=self.temperature,
        )
        self.history.append({
            "messages": messages,
            "response_dict": response.model_dump_json(indent=2),
            "response_content": response.choices[0].message.content,
        })
        return response.choices[0].message.content
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.history, f)
