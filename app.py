import os
from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# Initialize Flask app
app = Flask(__name__)

# Initialize models and chain
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-v0.1",
    model_kwargs={"temperature": 0.1, "max_length": 750},
    huggingfacehub_api_token="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"
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Load and process the PDF
loader = PyPDFLoader("my.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
retriever = vectorstore.as_retriever()

template = """
Answer this question using the provided context only.
{question}
Context:
{context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

def format_response(response, question):
    answer_start = response.find("Answer:")
    if answer_start != -1:
        clean_response = response[answer_start + len("Answer:"):].strip()
    else:
        clean_response = response.strip()
    return clean_response

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        question = request.json['question']
        response = chain.invoke(question)
        formatted_response = format_response(response, question)
        return jsonify({'answer': formatted_response})
    except Exception as e:
        return jsonify({'answer': f'Error processing question: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)