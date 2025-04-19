"""
1. Meta is the company behind Facebook and Instagram. They created the LLaMA models, which are open-source AI models that can generate text and be customized or fine-tuned. 

2. OpenAI is the company that built ChatGPT, GPT-3, and GPT-4. Their models are not open-source, but they are very powerful and easy to use through cloud-based APIs. 

3. Ollama is a newer company that builds a tool for running large language models like LLaMA,gemma, Phi, Mistral, and others locally on your computer. 

4. Phi is LLM model used to generate a text developed by Microsoft it is also opensource like llama can be runned locally like llama via ollama and can be finetunned 

5. GPT: Hosted on OpenAI’s servers, requires an API key and API requests.

"""
#Build my own chatbot that interacts with phi model to get answers for my questions using gardio framework
import gradio as gr
import openai

# Connect to local model (like Ollama running phi model)
client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="noipeneeded")

# Chat function
def chatwithmodel(history, new_message):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for user_message, assistant_response in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_response})
    messages.append({"role": "user", "content": new_message})

    response = client.chat.completions.create(
        model="phi:latest",
        temperature=0.3,
        messages=messages
    )

    assistant_message = response.choices[0].message.content
    history.append((new_message, assistant_message))
    return history, ""

# Create Gradio UI
def gradiochat():
    with gr.Blocks() as app:#for web application
        gr.Markdown("## Ollama Phi Model Chat Interface")
        chatbot = gr.Chatbot(label="Chat")
        user_input = gr.Textbox(label="Your Message", placeholder="Type something...", lines=1)
        send_button = gr.Button("Send")
        clear_button = gr.Button("Clear Chat")

        def clearchat():
            return [], ""

        send_button.click(
            fn=chatwithmodel,#process the chat
            inputs=[chatbot, user_input],#input comes from history of chatbot and ne message
            outputs=[chatbot, user_input]#Output will update both chat history and reset the input box
        )

        clear_button.click(
            fn=clearchat,
            inputs=[],
            outputs=[chatbot, user_input]
        )

    return app

# Run the app
if __name__ == "__main__":
    app = gradiochat()
    app.launch()#sets up a local web server to interact with the chatbot
