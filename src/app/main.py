import streamlit as st
import pandas as pd


from src.data_preparation import load_dataset

dataset_path = "data/dataset.jsonl"

dataset = load_dataset(dataset_path)

st.title("Axialent Test Challeng task")

st.markdown(
    "In this work I tried to analyse dataset with user communications with usage of LLM and langchain for finding patterns and insights."
)

st.markdown("## Dataset")


col1, col2 = st.columns(2)

with col1:
    sample = pd.DataFrame(dataset["inputs"][1]["messages"])
    st.dataframe(sample, hide_index=True, width=500)

with col2:
    st.write("Dataset contains 19 conversation between user and assistant.")
    st.write("First sample of dataset was removed due to api errors in assistant messages.")
    st.write("Dataset was preprocessed in following way:")
    st.write("- Duplicate messages were removed")
    st.write("- Assistant api error messages were removed")
    st.write("- Text feedback was converted to numerical scores (from 1 to 5)")
    st.write("- Conversation length and user words were calculated")
