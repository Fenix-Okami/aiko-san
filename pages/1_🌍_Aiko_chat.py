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
voice_toggle = st.sidebar.toggle('Generate audio',value=False) 
image_toggle = st.sidebar.toggle('Generate illustrations',value=False) 
immersion_toggle = st.sidebar.toggle('Immersion mode',value=False)
if immersion_toggle:
    immersion="The student wants a fully immersive experience, so ONLY respond in Japanese"
else:
    immersion=""
n_level= st.sidebar.selectbox("Set Aiko's level",('N5 - Basic',
                                                  'N4 - Elementary', 
                                                  'N3 - Intermediate',
                                                  'N3 - Advanced',
                                                  'N1 - Fluent'))

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

def get_image(prompt):
     gpt_response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt="""the following is the AI response from a chat. 
            Take this prompt and output a different prompt, in english, to pass
            to the DALL-E-3 text-to-image generator to create an appropriate illustration.
            Add instructions to make the image output a handdrawn, anime style.
            be as detailed as possible.
            : """+ prompt
            )
     image_prompt=gpt_response.choices[0].text
     response = client.images.generate(
                model="dall-e-3",
                prompt=image_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
                )
     return response.data[0].url

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
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        system=[{"role":"system","content": f"""
                 Persona: Japanese Tutor - Aiko. 
                 Background: A college student based in Tokyo. 
                 Personality Traits: Warm and welcoming, enthusiastic about teaching. 
                 Tutoring Expertise: Proficient in tutoring Japanese language learners, 
                    knowledgeable about Japanese culture, capable of teaching students
                    at various proficiency levels (current student is at the {n_level}).
                 Language Usage: Utilizes appropriate hiragana and kanji, emphasizes natural, 
                    conversational responses. 
                 Interaction Guidelines: Respond as a normal person would in a conversation, 
                    tailor responses to the learner's proficiency level ({n_level}), 
                    include cultural insights relevant to language learning. 
                    {immersion}
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
        if image_toggle:
            st.image(get_image(full_response))


        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    else:
        main()