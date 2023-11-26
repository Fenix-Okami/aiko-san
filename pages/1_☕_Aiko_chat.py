import streamlit as st
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
import os
from io import BytesIO
from pydub import AudioSegment
import numpy as np
# import other required libraries

from streamlit.components.v1 import html

from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

import tiktoken

response_template= """
  persona_and_goal: Aiko is a warm and welcoming college student from Tokyo, enthusiastic about teaching Japanese. She is skilled in tutoring learners of various proficiency levels and knowledgeable in Japanese culture. The goal is to provide engaging and supportive responses, fostering the user's understanding and speaking of Japanese. Responses should be tailored to the user's JLPT level {n_level}, incorporating cultural insights and language corrections in a friendly manner.
  difficulty: JLPT {n_level}

  Strictly respond in two parts in this order:
  "action": *italicized* text of a simple, relatable action performed by Aiko at the start of each response, described in {response_lang}. For example, *Aiko nods thoughtfully* or *Aiko gestures to a Japanese text*. This element sets the stage for a friendly and interactive learning environment. have this be its own paragraph.

  "response": Aiko's main response to the user's query or statement. {immersion}. NEVER USE ROMANJI unless asked about hiragana and katakana. {translation}
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

n_level= st.sidebar.selectbox("Set Aiko's level",('N5 - Basic',
                                                  'N4 - Elementary', 
                                                  'N3 - Intermediate',
                                                  'N3 - Advanced',
                                                  'N1 - Fluent'))
gpt_model= st.sidebar.selectbox("Set GPT model",('gpt-4','gpt-3.5-turbo'),index=1)

reset_button = st.sidebar.button("Reset Chat")

with st.sidebar:
    st.write("""| Model                   | Input Cost per 1K Tokens | Output Cost per 1K Tokens |
|-------------------------|--------------------------|---------------------------|
| gpt-4                   | $0.030                  | $0.060                     |
| gpt-3.5-turbo           | $0.001                  | $0.002                     |
             
""")

persona_and_guidelines="""persona_and_goal: Aiko is a warm and welcoming college student from Tokyo, enthusiastic about teaching Japanese. She is skilled in tutoring learners of various proficiency levels and knowledgeable in Japanese culture. The goal is to provide engaging and supportive responses, fostering the user's understanding and speaking of Japanese. Responses should be tailored to the user's JLPT level {n_level}, incorporating cultural insights and language corrections in a friendly manner.difficulty: JLPT {n_level}"""

action_schema = ResponseSchema(name="action",
                             description="A simple, relatable action performed by Aiko at the start of each response, described in {response_lang}. For example, 'Aiko nods thoughtfully' or 'Aiko gestures to a Japanese text'. This element sets the stage for a friendly and interactive learning environment.")
response_schema = ResponseSchema(name="response",
                                      description="Aiko's main response to the user's query or statement. {immerson}")
explanation_schema = ResponseSchema(name="explanation",
                                    description="An explanation and breakdown in {response_lang}, particularly focusing on any corrections to the user's Japanese grammar or language use. This part aims to provide clear and helpful insights to aid the user's learning, explaining language points or cultural references mentioned in the response.")

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

    if reset_button:
        st.session_state.messages = []
        st.markdown('')
 
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = gpt_model

    if "messages" not in st.session_state:
        st.session_state.messages = []
        # st.session_state.messages = [SystemMessage(content=response_template)]
        # st.session_state.messages =[{"role":"system","content": f"""
        #             Persona: assistant is Aiko, a Japanese Tutor
        #             Background: College student in Tokyo
        #             Personality Traits: Warm, welcoming, enthusiastic about teaching Japanese
        #             Tutoring Expertise: Skilled in tutoring Japanese language learners of 
        #                 various proficiency levels, knowledgeable in Japanese culture
        #             Language Level: The current user is at JLPT level {n_level}, and knows English as their first language
        #             Language Usage: Use of appropriate hiragana, kanji, and conversational Japanese
        #             Interaction Style:
        #                 ALWAYS write an initial paragraph with Aiko performing a contextually apropriate action in the third person and in ENGLISH (e.g., *Aiko smiles warmly, Aiko takes a sip of her tea, etc*) use italicized text and dedicate a paragraph to this.
        #                 Always perform this action first before responding.
        #                 Always address the user directly as "you" in the first paragraph.
        #                 Tailor ALL Japanese responses to the user's JLPT {n_level}, using simpler language for lower levels and gradually introducing complexity for higher levels.
        #                 Integrate cultural insights relevant to the conversation topic, enhancing the immersive learning experience.
        #                 ALWAYS Provide corrections to the user's Japanese grammar in a supportive manner, followed by clear explanations in ENGLISH.
        #                 Encourage user engagement through questions and interactive dialogue.
        #                 {immersion}
        #             Objective: Aiko's responses should foster an engaging, supportive environment, 
        #                 focusing on the user's progress in understanding and speaking Japanese. 
        #                 The approach should be mindful of the user's proficiency level, ensuring not to overwhelm them.
        #             """}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask Aiko"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # persona_and_guidelines.format(n_level=n_level)
        # response_template.format(n_level=n_level,response_lang=response_lang,immersion=immersion)
        
        # response_schemas = [action_schema.format(response_lang=response_lang), 
        #                     response_schema.format(immersion=immersion),
        #                     explanation_schema.format(response_lang=response_lang)]
        # output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        # format_instructions = output_parser.get_format_instructions()

        # prompt = ChatPromptTemplate.from_template(template=response_template)

        # messages = prompt.format_messages(text=customer_review, 
        #                                   format_instructions=format_instructions)
        
        # chat = ChatOpenAI(model=gpt_model)
        system=[{"role":"system","content":response_template.format(n_level=n_level,response_lang=response_lang,immersion=immersion,translation=translation)}]
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            # with st.spinner("Thinking..."):
            #     response = chat(st.session_state.messages)
            #     st.session_state.messages.append(
            #         AIMessage(content=response.content))
            for response in client.chat.completions.create(
                                                        model=st.session_state["openai_model"],
                                                        messages=system+[
                                                            {"role": m["role"], "content": m["content"]}
                                                            for m in st.session_state.messages],
                                                        stream=True,
                                                        ):
                full_response += (response.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")
                
            # output_dict = output_parser.parse(response.content)
            ###Track token usage for awareness
            messages =          system+[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            prompt_tokens =     num_tokens_from_messages(messages,gpt_model)
            generated_tokens  = num_tokens_from_string(full_response,'cl100k_base')
            token_usage =       prompt_tokens + generated_tokens

            message_placeholder.markdown(full_response)
            if voice_toggle:
                # message_placeholder.audio(get_audio(full_response), format='audio/mp3')
                st.audio(get_audio(full_response), format='audio/mp3')
            if image_toggle:
                # message_placeholder.image(get_image(full_response))
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