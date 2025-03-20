import pickle
from openai import AzureOpenAI

from reward_tuning.API_key import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT

import logging

CLIENT_LOG_TEMPLATE = '''
Calling LLM. 
Model: {model}
Temperature: {temp}
Message: {message}
Response: {response}
'''

def messages_to_string(messages: list):
    result = ''
    for message in messages:
        result += f"{message['role']} : {message['content']}\n"
    return result

class Client:
    def __init__(self, temperature=0.5, disable=False, template=None):
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY, 
            api_version="2024-02-01",
            azure_endpoint=AZURE_OPENAI_ENDPOINT, # os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        self.deployment_name = "gpt-4o"
        self.history = []
        self.temperature = temperature
        self.disable = disable
        self.template = template

    def response(self, messages, log=None):
        if log != None:
            logging.info(f'{log}    Request for response')
        if self.disable:
            logging.debug(f'Calling LLM. Cheat with fixed response:\n{self.template}')
            return self.template
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            temperature=self.temperature,
        )

        logging.debug(CLIENT_LOG_TEMPLATE.format(
            model=self.deployment_name,
            temp=self.temperature,
            message=messages_to_string(messages),
            response=response.choices[0].message.content
        ))
        self.history.append({
            "messages": messages,
            "response_dict": response.model_dump_json(indent=2),
            "response_content": response.choices[0].message.content,
        })
        return response.choices[0].message.content
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.history, f)
