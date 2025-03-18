import os
from openai import AzureOpenAI
from reward_tuning.API_key import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT
from reward_tuning.prompt import INITIAL_USER, INITIAL_SYSTEM

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY, #os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-02-01",
    azure_endpoint=AZURE_OPENAI_ENDPOINT, # os.getenv("AZURE_OPENAI_ENDPOINT"),
)

deployment_name='gpt-4o' #This will correspond to the custom name you chose for your deployment when you deployed a model. Use a gpt-35-turbo-instruct deployment. 

# Send a completion call to generate an answer
print('Sending a test completion job')
response = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {"role": "system", "content": INITIAL_USER},
        {"role": "user", "content": INITIAL_SYSTEM}
    ]
)
# print(response)
print(response.model_dump_json(indent=2))
print(response.choices[0].message.content)