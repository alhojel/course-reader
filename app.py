import ast
import numpy as np
import pandas as pd
import streamlit as st
import openai
import os
from openai.embeddings_utils import get_embedding

openai.api_key =  st.secrets["OPENAI_API_KEY"]
DATA_URL = "final_shrinked.csv"


@st.cache_data
def blending_wonders():
    df = pd.read_csv(DATA_URL)
    df['embeddings'] = df['embeddings'].apply(ast.literal_eval)
    return df

@st.cache_data
def brewing_magic(search_query, df):
    search_embedding = get_embedding(search_query, engine='text-embedding-ada-002')
    embeddings = np.array(df["embeddings"].tolist())
    similarities = np.dot(embeddings, search_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(search_embedding))
    df["similarities"] = similarities
    df.sort_values(by="similarities", ascending=False, inplace=True)
    return df.head(15)

def display_result_card(result):
    card_style = """
    <style>
        .card {
            background-color: #222222;
            border: 1px solid #ccc;
            border-radius: 4px;
            box-shadow: 1px 1px 4px rgba(0, 0, 0, 0.1);
            padding: 15px;
            margin-bottom: 15px;
        }
    </style>
    """
    class_name = f"{result['name']}"
    #class_code = f"Code: <a href='{result['url']}'>{result['url']}</a>"
    #class_id = f"Class ID: {result['Class ID']}"
    #dept_link = f"Dept: <a href='{result['Department URL']}'>{result['Department']}</a>" if pd.notnull(result['Department URL']) else result['Department']
   #instruction_mode = f"{result['Instruction Mode']}"
    #location = f"Location: <a href='{result['Building URL']}'>{result['Location']}</a>" if pd.notnull(result['Building URL']) else ""

    #if isinstance(result['Location'], float):
    #    location = ""


            #<p>{result['page']} unit{'s' if result['Units'] != "1" else ''}  |  {result['Time']}  |  {result['Meets Days']} | {class_code}</p>
    #<p>{result['text']}</p>
    card_content = f"""
    <a href='{result['url']}#page={result['page']+1}' style='text-decoration: none; color: inherit;'>
        <div class="card">
            <h3>{class_name}</h3>
            <p >Click to open...</a>
            <p>Page Numer: {result['page']+1} | Type: {result['type']} | Vector Similarity: {result['similarities']}</p>
        </div>
    </a>
    """

    st.markdown(card_style, unsafe_allow_html=True)
    st.markdown(card_content, unsafe_allow_html=True)

def main():
    st.markdown("<h1 style='text-align: center;'><a href='https://berkeley.streamlit.app/' style='text-decoration: none; color: inherit;'>EECS16B CourseReader 🚀</a></h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-top: -10px; color: #ccc;'>Find that note you're looking for by searching for concepts using AI</p>", unsafe_allow_html=True)

    with st.expander('Add Filters'):
        st.write("More filters coming soon 👀")
        st.write('Resource Type:')
        col1, col2, col3  = st.columns(3)
        type_filters = {
         'notes': False,
            'homeworks': False,
            'discussions': False,
        }
        type_filters['notes'] = col1.checkbox('Notes', value=type_filters['notes'])
        type_filters['homeworks'] = col2.checkbox('Homeworks', value=type_filters['homeworks'])
        type_filters['discussions'] = col3.checkbox('Discussions', value=type_filters['discussions'])
        
    search_query = st.text_input("✨ Search for a note, homework, or discussion:", placeholder="What is a Full SVD?", key='search_input')
        
    if search_query:
        df = blending_wonders()
        
        # # Filter by the selected units
        selected_type_filters = [unit[0] for unit, value in type_filters.items() if value]
        if selected_type_filters:
            df = df[df['type'].apply(lambda x: any(val in selected_type_filters for val in x))]

        results = brewing_magic(search_query, df)
        
        for i in range(3): # Always display the first 7 entries
            if i < len(results):
                display_result_card(results.iloc[i])
    st.markdown("<p style='text-align: center; margin-top: 10px; color: #ccc;'>🚨 If the text is illegible, set the theme to DARK: 3 lines on the top right > settings > theme: dark 🚨</p>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; margin-top: 5px;'><a href='https://forms.gle/NsvJXWy1x43Jdz9y7'>Leave feedback</a></div>", unsafe_allow_html=True)
    #st.markdown("<p style='text-align: center; margin-top: 20px; color: #ccc;'>Currently in beta with upcoming features</p>", unsafe_allow_html=True)
    st.markdown("<hr margintop: 20px>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-top: 25;'>Made with ♥︎ by Alberto Hojel</p>", unsafe_allow_html=True)
    #st.markdown("<div style='text-align: center; margin-top: 10px;'><a href='https://www.buymeacoffee.com/jaysethi' target='_blank'><img src='https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png' alt='Buy Me A Coffee' width='150' ></a></div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()