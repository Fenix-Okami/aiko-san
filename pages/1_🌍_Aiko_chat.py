import streamlit as st
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
import os
from io import BytesIO
from pydub import AudioSegment
import numpy as np
# import other required libraries

from streamlit_chat import message
from streamlit.components.v1 import html

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
voice_toggle = st.sidebar.toggle('generate audio',value=False)
N_level= st.sidebar.selectbox("Set Aiko's level",('N5 - Easiest','N4 - Moderate', 'N3 - Advanced'))


# openai_api_key = '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'
if openai_api_key.startswith('sk-'):
    chat_model = ChatOpenAI(openai_api_key = openai_api_key)
    client = OpenAI(api_key = openai_api_key)

def on_input_change():
    user_input = st.session_state.user_input
    st.session_state.past.append(user_input)
    st.session_state.generated.append("The messages from Bot\nWith new line")

def on_btn_click():
    del st.session_state.past[:]
    del st.session_state.generated[:]

st.image('aiko.png', caption='愛子さん', width=300)

def get_response(message):
    response=chat_model.predict(message)
    return response

def get_audio(message):
 
    response = client.audio.speech.create(
        model="tts-1-hd",
        voice="nova",
        input=message,
        response_format="mp3"
    )
    response.stream_to_file("output.mp3")
    audio_file = open('output.mp3', 'rb')
    audio_bytes = audio_file.read()

    st.audio(audio_bytes, format='audio/mp3')
    return 

# if "openai_model" not in st.session_state:
#     st.session_state["openai_model"] = "gpt-3.5-turbo"

#  # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

def main():
    st.title("Aiko-San")
    

    # if 'chat_history' not in st.session_state:
    #     st.session_state.chat_history = []

    # user_input = st.text_input("Type your message here:")

    
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask Aiko"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user",avatar=st.image('aiko.png',width=30)):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=st.image('aiko.png',width=30)):
        message_placeholder = st.empty()
        full_response = ""
        system=[{"role":"system","content": f"""
                 Persona: Japanese Tutor - Aiko. 
                 Background: A college student based in Tokyo. 
                 Personality Traits: Warm and welcoming, enthusiastic about teaching. 
                 Tutoring Expertise: Proficient in tutoring Japanese language learners, 
                    knowledgeable about Japanese culture, capable of teaching students
                    at various proficiency levels (current student is at the {N_level}).
                 Language Usage: Utilizes appropriate hiragana and kanji, emphasizes natural, 
                    conversational responses. 
                 Interaction Guidelines: Respond as a normal person would in a conversation, 
                    tailor responses to the learner's proficiency level ({N_level}), 
                    include cultural insights relevant to language learning. 
                 Note: Aiko's responses should be engaging and supportive, encouraging 
                    the learner's progress in understanding and speaking Japanese,
                    while providing an immersive learning experience."
                """}]
        for response in client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=system+[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_response += (response.choices[0].delta.content or "")
            message_placeholder.markdown(full_response + "▌")
        if voice_toggle:
            get_audio(full_response)
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    else:
        main()