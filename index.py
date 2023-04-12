import os, streamlit as st

# Uncomment to specify your OpenAI API key here (local testing only, not in production!), or add corresponding environment variable (recommended)
# os.environ['OPENAI_API_KEY']= ""

from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper
from langchain import OpenAI

# This example uses text-davinci-003 by default; feel free to change if desired
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-002"))

# Configure prompt parameters and initialise helper
max_input_size = 2048
num_output = 128
max_chunk_overlap = 20

prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

# Load documents from the 'data' directory
documents = SimpleDirectoryReader('data').load_data()
index = GPTSimpleVectorIndex(
    documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
)

# Define a simple Streamlit app
st.title("Ask Llama")
query = st.text_input("What would you like to ask?", "")

if st.button("Submit"):
    response = index.query(query)
    st.write(response)
