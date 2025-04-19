import os#operating system
import numpy as np#for embeddings
from langchain_community.llms import HuggingFaceHub #to answer like human
from langchain_community.embeddings import HuggingFaceEmbeddings#embedd files to be saved in vector database
from langchain.text_splitter import RecursiveCharacterTextSplitter#chunking texts inside documents into smaller parts
from langchain_community.vectorstores import FAISS #vector database 
from langchain_core.prompts import PromptTemplate #tells llms how to answer
from langchain_community.document_loaders import PyPDFDirectoryLoader#convert pdfs into langchain document objects
from langchain.chains import RetrievalQA#build chains to answer(generate answer)

#load pdfs from url (internet source or website)

from urllib.request import urlretrieve #retrieves pdfs from url
files = [
    "https://www.census.gov/content/dam/Census/library/publications/2022/demo/p70-178.pdf",
    "https://www.census.gov/content/dam/Census/library/publications/2023/acs/acsbr-017.pdf",
    "https://www.census.gov/content/dam/Census/library/publications/2023/acs/acsbr-016.pdf",
    "https://www.census.gov/content/dam/Census/library/publications/2023/acs/acsbr-015.pdf",
]
#create folder to save files inside
os.makedirs("faten",exist_ok=True)
#create path for each pdf 
for file in files:
    path=os.path.join("faten",file.rpartition("/")[2])#path inside faten folder so os.join.path creates path by joining folder name and file name you want to save
    urlretrieve(file,path)#urlretrive takes url and path
#now all pdfs are dowloaded and saved in folder named faten 
#convert pdfs into langchain document objects
loader=PyPDFDirectoryLoader("faten")#takes folder name 
beforesplit=loader.load()
#now each page in pdf is converted to langchain document object so before split length is 63 
#each document object contains dictionary of metdata(author,pagenumber,title,source..) and page content(text inside the page)
#now each pagecontent contains 5000 words we cannot embed them
#chunk the text in the document into smaller parts in this case number of documents will increase and in each document we can indicate how many caharcter we wants
#chunking using text splitter
splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)#i need 500 word in each document
aftersplit=splitter.split_documents(beforesplit)#split documents(chunking documents)
#now after split length will be greater than 63 but number of caharacters inside each document is maximum 500
#now they are ready for embedding in order to be saved in vector database
embed=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device":"cpu"},
    encode_kwargs={"normalize_embeddings":True}

)#model name used to embedd text into vectors , and load it into cpu or cuda
#now create vectordatabase to store embeddings inside
vectorstore=FAISS.from_documents(aftersplit,embedding=embed)
#now the user will ask  question and the question will be embedded
#the vector database will search for documnets similar to question and retrieve them
#now the retrieved document will be entered to LM with the question to generate answer like human

#so retrievalqa.chain is to create chain (steps) to generate tha answer(which model you want to use,prompt,retreiver(how the documnets will be retrieved)..)
#we should create prompt template to tell LM how to answer 
access_token="**"#to access the model in hugginface
LM=HuggingFaceHub(
    repo_id='mistralai/Mixtral-8x7B-Instruct-v0.1',
    model_kwargs = {"temperature" : 0.1 , "max_length" : 500},
    huggingfacehub_api_token=access_token

)
temp="""
Use the following pieces of context to answer the question.
- If you don't know the answer, say: "I couldn't find a definitive answer."
- Be clear and concise. Answer in 3–5 sentences max.

{context}

{question}

Answer here:

"""

PROMPT = PromptTemplate(
 template=temp, input_variables=["context", "question"]
)

ret=vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":3})

generation=RetrievalQA.from_chain_type(
    llm=LM,
    retriever=ret,
    chain_type="stuff",
    chain_type_kwargs={"prompt":PROMPT}

)
#retrieval documents are stuffed in one block
#prompt is filled
#filled prompt will enter to LLM 
#query="How poverty in measured"
#output=generation.invoke(query)

import streamlit as st
st.set_page_config(page_title="chatbot",layout="centered")
st.title("Hello my chatbot")
input=st.text_input("Type your answer")
if "history" not in st.session_state:
    st.session_state.history=[]
if input:
    output=generation.invoke(input)
    answer=output['result']
    st.session_state.history.append({"question":input,"answer":answer})
for chat in st.session_state.history:
    st.markdown(f"**You:**{chat['question']}")
    st.markdown(f"**Bot:**{chat['answer']}")
    st.markdown("----------")

#need own API if:frontend (user interface) and backend (logic with LLM, vector DB) are not running in the same place.
#but here they are running in same place where user enters question and the python code is invoked to get the answer
#streamlit handles both backend and frontend

