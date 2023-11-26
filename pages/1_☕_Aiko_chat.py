import streamlit as st
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
import os
from io import BytesIO
from pydub import AudioSegment
import numpy as np
# import other required libraries

from streamlit.components.v1 import html
from PIL import Image

import tiktoken

image = Image.open('aiko.png')
response_template= """
  persona_and_goal: Aiko is a warm and welcoming college student from Tokyo, enthusiastic about teaching Japanese. She is skilled in tutoring learners of various proficiency levels and knowledgeable in Japanese culture. The goal is to provide engaging and supportive responses, fostering the user's understanding and speaking of Japanese. You have been engaging with a student. Whenever he response in Japanese, they are practicing or making an attempt.

  Strictly respond in two parts in this order:
  "action": *italicized* text of a simple, relatable action performed by Aiko at the start of each response, described in {response_lang}. For example, *Aiko nods thoughtfully* or *Aiko gestures to a Japanese text*. This element sets the stage for a friendly and interactive learning environment. have this be its own paragraph.

  "response": Aiko's main response to the user's query or statement. {difficulty}. NEVER USE ROMANJI. {translation}
"""

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
translation_toggle = st.sidebar.toggle('Include translation',value=True)
voice_toggle = st.sidebar.toggle('Generate audio (additional fees apply)',value=False) 
image_toggle = st.sidebar.toggle('Generate illustrations (additional fees apply)',value=False) 
# immersion_toggle = st.sidebar.toggle('Immersive mode',value=False)
immersion_toggle = False
if immersion_toggle:
    immersion="Aiko responds exclusively in Japanese, suitable for the user's JLPT level, to maximize immersion."
    response_lang="Japanese"
else:
    immersion="Respond with an approprimate mix Japanese and English, suitable for the user's JLPT level, balancing language practice and clear communication"
    response_lang="English"
if translation_toggle:
    translation="Include the full translation of Aiko's response in English"
else:
    translation="DO NOT simply provide a full English translation"

n_level = st.sidebar.selectbox("Set Aiko's difficulty", ('Absolute Beginner',
                                                        'Basic', 
                                                        'Intermediate',
                                                        'Advanced',
                                                        'Fluent'))
def get_instructions(level):
    instructions = {
        'Absolute Beginner': "Use simple English predominantly, with very basic Japanese phrases interspersed. Focus on familiarizing the user with common expressions and encourage attempts at Japanese, while primarily explaining concepts in English for clarity.",
        'Basic': "Balance English and Japanese in your responses. Include basic Japanese phrases with furigana. Offer explanations in English, but encourage the use of Japanese in conversation. Provide challenges in English and give feedback in a mix of both languages.",
        'Intermediate': "Use Japanese primarily, but provide occasional explanations or check-ins in English. Include more complex grammar and vocabulary. Encourage the user to express themselves in Japanese, offering corrections and suggestions in a mix of English and Japanese.",
        'Advanced': "Focus mainly on Japanese, using English only for complex explanations or when addressing misunderstandings. Emphasize advanced grammar and nuanced expressions. Encourage sophisticated conversation in Japanese, correcting subtle mistakes and providing feedback.",
        'Fluent': "Conduct sessions almost entirely in Japanese. Use English minimally, primarily for clarifications. Focus on fluency, idiomatic expressions, and cultural nuances, engaging in advanced conversations and discussing complex topics in Japanese."
    }
    return instructions.get(level)


gpt_model= st.sidebar.selectbox("Set GPT model",('gpt-4','gpt-3.5-turbo'),index=1)

reset_button = st.sidebar.button("Reset Chat")

with st.sidebar:
    st.write("""| Model                   | Input Cost per 1K Tokens | Output Cost per 1K Tokens |
|-------------------------|--------------------------|---------------------------|
| gpt-4                   | $0.030                  | $0.060                     |
| gpt-3.5-turbo           | $0.001                  | $0.002                     |
             
""")

left_co, right_co = st.columns(2)
with left_co:
    st.image('aiko.png', caption='愛子さん', width=300)
with right_co:
    st.markdown("""*Seated in the cozy corner of a quaint café, 
                Aiko fits perfectly into the warm, inviting ambiance. 
                With black hair framing her thoughtful expression and 
                brown eyes peering through glasses, she radiates a 
                sense of quiet intellect. She's engrossed in a book of 
                Japanese poetry, occasionally sipping green tea. 
                Her attire is casual yet stylish, blending seamlessly 
                with the café's relaxed vibe. As she looks up, 
                her eyes meet yours. 
                A gentle, welcoming smile crosses her face, reflecting her 
                eagerness to share her knowledge of language and culture.* """)

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

if 'prompt_tokens' not in locals():
    prompt_tokens=0
    generated_tokens=0
    token_usage=0

if openai_api_key.startswith('sk-'):
    chat_model = ChatOpenAI(openai_api_key = openai_api_key)
    client = OpenAI(api_key = openai_api_key)

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

    if reset_button:
        st.session_state.messages = []
        st.markdown('')
 
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

        system=[{"role":"system","content":response_template.format(n_level=n_level,response_lang=response_lang,difficulty=get_instructions(n_level),translation=translation)}]
        with st.chat_message("assistant",avatar=np.array(image)):
            message_placeholder = st.empty()
            full_response = ""

            for response in client.chat.completions.create(
                                                        model=st.session_state["openai_model"],
                                                        messages=system+[
                                                            {"role": m["role"], "content": m["content"]}
                                                            for m in st.session_state.messages],
                                                        stream=True,
                                                        ):
                full_response += (response.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")
                
            ###Track token usage for awareness
            messages =          system+[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            prompt_tokens =     num_tokens_from_messages(messages,gpt_model)
            generated_tokens  = num_tokens_from_string(full_response,'cl100k_base')
            token_usage =       prompt_tokens + generated_tokens

            message_placeholder.markdown(full_response)
            if voice_toggle:
                st.audio(get_audio(full_response), format='audio/mp3')
            if image_toggle:
                st.image(get_image(full_response))
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        with st.sidebar:
            cost=prompt_tokens/1000*.001+generated_tokens/1000*.002
            max_context=4096
            if gpt_model=='gpt-4':
                cost=cost*30
                max_context=max_context*2
            st.code(f"""
                    --last response--
                    input: {prompt_tokens} tokens
                    output: {generated_tokens} tokens
                    estimated cost: ${cost:.5f} dollars
                    {token_usage}/{max_context} tokens in full context
                    """)

if __name__ == "__main__":
    if not openai_api_key.startswith('sk-'):
        st.info('Please enter your OpenAI API key on the left sidebar. You can create one at https://openai.com/blog/openai-api', icon='⚠')
    else:
        main()