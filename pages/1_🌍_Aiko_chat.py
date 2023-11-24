import streamlit as st
from openai import OpenAI
import openai
from langchain.chat_models import ChatOpenAI
import os
from io import BytesIO
from pydub import AudioSegment
import numpy as np
import tiktoken as util
# import other required libraries

from streamlit_chat import message
from streamlit.components.v1 import html

import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def num_tokens_from_messages(messages, model):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

try:
    prompt_tokens
except:
    prompt_tokens=0
    generated_tokens=0
    token_usage=0

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
gpt_model= st.sidebar.selectbox("Set GPT model",('gpt-4','gpt-3.5-turbo'))

with st.sidebar:
    st.write("""| Model                   | Input Cost per 1K Tokens | Output Cost per 1K Tokens |
|-------------------------|--------------------------|---------------------------|
| gpt-4                   | $0.030                  | $0.060                     |
| gpt-3.5-turbo           | $0.001                  | $0.002                     |
""")


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

    # st.audio(audio_bytes, format='audio/mp3')
    return audio_bytes 

def main():
    st.title("Aiko-San")
    
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = gpt_model

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
                    Always correct the student's grammar, and provide the explaination in english.
                    For a bit more of a human touch use italicized text in its own paragraph,
                    describe Aiko performing some action, such as waving hello, smiling,
                    or grabbing a cup of coffee. This is a roleplay so have Aiko play along.
                    always refer to the student as "you" and never say "the student.
                    Always perform this action first before responding.
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

            print(response)

        # Initialize token count

        prompt_tokens = 0
        messages = system + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        # # Tokenize each message and add to the count
        # for prompt_piece in messages:
        #     tokens = num_tokens_from_string(prompt_piece['content'],gpt_model)
        #     prompt_tokens += len(tokens)
        prompt_tokens = num_tokens_from_messages(messages,gpt_model)
        generated_tokens  =  num_tokens_from_string(full_response,'cl100k_base')
        token_usage = prompt_tokens+generated_tokens

        message_placeholder.markdown(full_response)
        if voice_toggle:
            # message_placeholder.audio(get_audio(full_response), format='audio/mp3')
            st.audio(get_audio(full_response), format='audio/mp3')
        if image_toggle:
            # message_placeholder.image(get_image(full_response))
            st.image(get_image(full_response))
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    with st.sidebar:
        st.write(f"{prompt_tokens} tokens in input")
    with st.sidebar:
        st.write(f"{generated_tokens} tokens generated in output")
    with st.sidebar:
        st.write(f"{token_usage} total tokens in chat. max context length is 8000")

if __name__ == "__main__":
    if not openai_api_key.startswith('sk-'):
        st.info('Please enter your OpenAI API key. You can create one at https://openai.com/blog/openai-api', icon='⚠')
    else:
        main()