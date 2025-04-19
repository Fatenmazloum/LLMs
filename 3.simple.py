
"""
OpenAI is python library gives you access to  ai models like ChatGPT,PHI via an API, so you can use its language abilities for tasks like building a chatbot, answering questions, and more.

You can customize ChatGPT to fit your needs (like specific knowledge about your products).
OpenAI's API (GPT) helps me build my own chatbot. I can ask GPT any question and it replies. But for new things like my own product, it doesn’t know. So I give it my own data, and then it can answer questions about my products too — even common questions people usually ask. This way, I don’t have to build the chatbot from scratch.
In this case we are not fine tunning the GPT model by updating its weights and parameters instead customizing its responses.

ChatGPT through the OpenAI API , you need an API key
"""

#run phi model in python code using openai library
import openai
client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="nokeyneeded")#hosted like chatgpt api key needed where phi and llama no api needed
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
