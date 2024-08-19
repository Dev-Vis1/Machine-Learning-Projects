import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline('text-generation', model='gpt2-xl', device='cuda')

model = load_model()

st.title("GPT-2 XL Text Generation")
prompt = st.text_input("Enter your prompt:", "Once upon a time")

if st.button("Generate"):
    with st.spinner('Generating...'):
        result = model(prompt, max_length=500, num_return_sequences=1)
        st.success(result[0]['generated_text'])
