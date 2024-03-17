import streamlit as st
import base64
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer

with open('c.png', 'rb') as f:
    background_image = base64.b64encode(f.read()).decode()

home = tensorflow.keras.models.load_model('gen2.keras')

bag = [
    'avengers logo',
    'nbkrist logo'
]
token = tensorflow.keras.preprocessing.text.Tokenizer(num_words = 10)
token.fit_on_texts(bag)

background = f"""
<style>

.appview-container {{
    background-image: url('data:image/png;base64, {background_image}');
    background-repeat: no-repeat;
    background-position: center;
    background-size: cover;
}}

.row-widget.stTextInput > div {{
    border-width: 0;
}}

.row-widget.stTextInput > div > div {{
    background-color: white;
}}

.row-widget.stTextInput > div > div > input {{
    background-color: lightgrey;
    border: 2px solid red;
    border-radius: 10px;
    color: red;
    caret-color: red;
}}

.row-widget.stButton > button{{
    background-color: lightyellow;
    color: red;
    border: 2px solid red;
}}

.row-widget.stDownloadButton > button{{
    background-color: lightyellow;
    color: red;
    border: 2px solid red;
}}
</style>
"""

st.markdown(background, unsafe_allow_html = True)

col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.markdown("<h3 style = 'color:red; margin-left: 0px;'>Image Generator</h3>", unsafe_allow_html = True)
    text = st.text_input('Generate', placeholder = 'Enter Text Prompt................', label_visibility = 'collapsed')
    submit = st.button('Generate')
    clear = st.button('Clear')

if submit:
    noise = tensorflow.random.normal([1, 50], stddev = 0.2)
    sequences = token.texts_to_sequences([text])
    emb = tensorflow.convert_to_tensor([sequences])
    img = home.predict([noise, emb])
    img = img[0]
    img = (img * 127.5) + 127.5
    img = img.astype('uint8')
    plt.imsave('img.jpg', img)
    with col2:
        st.image(img)
    with open('img.jpg', 'rb') as file:
        st.download_button('Download', data = file, file_name = 'image.jpg', mime = 'image/png')

if clear:
    with col2:
        st.empty()
