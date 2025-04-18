#run phi model in python code using openai library
import openai
client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="nokeyneeded")#hosted like chatgpt api key needed where as locally like phi and llama no api needed
#setting up a connection to an AI model (phi) using the OpenAI library and specifying the base URL and API key. 
#url webaddress to find resources online,localhost means locally 
#tell your Python code where Phi is running
response = client.chat.completions.create(
    model="phi:latest", 
    temperature=0.3,
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "what is the meaning of table."}
    ]
)
#create method sends a request to the model to generate a response based on the messages provided.

print(f'Response: {response.choices[0].message.content}')
#accessing the first response from the model

#When you're using GPT (like GPT-4 or GPT-3.5)from open ai:
#You → send a message to GPT using the API key → GPT → sends back a response
#so no need for baseurl because it is not locally in my machine like phi and llama it is hosted in cloud
