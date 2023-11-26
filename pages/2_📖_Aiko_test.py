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

response_template = """
persona_and_goal: You are Aiko. She is a warm and welcoming college 
student from Tokyo, enthusiastic about teaching Japanese. 
She is skilled in tutoring learners of various proficiency
levels and knowledgeable in Japanese culture. 
NEVER use romanji anywhere. Always use kanji that you would find at JLPT {n_level}. 
you are testing your student in this interaction, provide these items:

question: challenge your student with a question with a difficulty similar to the JLPT {n_level}. Provide the instructions in english unless the difficulty is N3 or above

A: First multiple choice answer.
B: Second multiple choice answer.
C: Third multiple choice answer.
D: Fourth multiple choice answer.

{format_instructions}
"""

# multiple choices: mulitiple choice of 4 possible answers. separate each with its own line.

answer_template = """
persona_and_goal: You are Aiko. She is a warm and welcoming college 
student from Tokyo, enthusiastic about teaching Japanese. 
She is skilled in tutoring learners of various proficiency
levels and knowledgeable in Japanese culture. 
Your student just provided you an answer to a JLPT question. give a response how Aiko would. NEVER use romanji.
If they are correct, start with "You got it!". 
If they state they're unsure start with "That's alright. let's go through each of the options".
Otherwise, start with "Not quite".
Then, ALWAYS state what the question is asking for.
Then, and AFTER stating what the question is asking for, primarily using english, provide explanations for each possible answer choice and the etymology of the kanji components if applicable. 
always conclude with "let me know when you are ready for the next question":

{response_instructions}
"""
left_co, right_co = st.columns(2)
with left_co:
    st.image('aiko_class.png', caption='愛子さん', width=300)
with right_co:
    st.markdown("""*In the vibrantly decorated classroom, Aiko, sporting her black running jacket, exudes energy and spontaneity. Her black hair sways rhythmically as she darts back and forth, quickly scribbling Japanese language questions for you on the board. Her boundless energy reflects the lively pace of her teaching style. Her enthusiasm is infectious, transforming the learning environment into a dynamic and engaging space.*""")

question_schema = ResponseSchema(name="question",
                             description=f"The JLPT question with a difficulty of N5")
# answers_schema = ResponseSchema(name="multiple choices",
#                                       description="the mulitiple choice of 4 possible answers")

choice_a_schema = ResponseSchema(name="A", description="Choice A")
choice_b_schema = ResponseSchema(name="B", description="Choice B")
choice_c_schema = ResponseSchema(name="C", description="Choice C")
choice_d_schema = ResponseSchema(name="D", description="Choice D")

# response_schemas = [question_schema, 
#                     answers_schema]
response_schemas = [question_schema, 
                    choice_a_schema, 
                    choice_b_schema, 
                    choice_c_schema, 
                    choice_d_schema]


output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

n_level= st.sidebar.selectbox("Set Question difficulty level",('N5 - Basic',
                                                  'N4 - Elementary', 
                                                  'N3 - Intermediate',
                                                  'N3 - Advanced',
                                                  'N1 - Fluent'))
gpt_model= st.sidebar.selectbox("Set GPT model",('gpt-4','gpt-3.5-turbo'),index=1)

persona_and_guidelines="""persona_and_goal: Aiko is a warm and welcoming college student from Tokyo, enthusiastic about teaching Japanese. She is skilled in tutoring learners of various proficiency levels and knowledgeable in Japanese culture. The goal is to provide engaging and supportive responses, fostering the user's understanding and speaking of Japanese. Responses should be tailored to the user's JLPT level {n_level}, incorporating cultural insights and language corrections in a friendly manner.difficulty: JLPT {n_level}"""

question_schema = ResponseSchema(name="question",
                             description=f"A JLPT question with a difficulty of {n_level}")
answers_schema = ResponseSchema(name="multiple choices",
                                      description="a mulitiple choice of 4 possible answers in markdown. have each be its own line")
explanation_schema = ResponseSchema(name="explanation",
                                    description="An explanation and breakdown in {response_lang}, particularly focusing on any corrections to the user's Japanese grammar or language use. This part aims to provide clear and helpful insights to aid the user's learning, explaining language points or cultural references mentioned in the response.")



if 'question' not in st.session_state:
    st.session_state.question = None
if 'choices' not in st.session_state:
    st.session_state.choices = None
if 'answer_choice' not in st.session_state:
    st.session_state.answer_choice = 0
if 'correct_answers' not in st.session_state:
    st.session_state.correct_answers = 0

def main():
    st.write(f"Correct answers this session: {st.session_state.correct_answers}")
    generate_button = st.button("Generate question")

    if generate_button:
        with st.spinner('Aiko is feverently writing on the chalkboard...'):
            chat = ChatOpenAI(model=gpt_model,openai_api_key=openai_api_key)
            prompt = ChatPromptTemplate.from_template(template=response_template)
            messages = prompt.format_messages(n_level=n_level,
                                            format_instructions=format_instructions)
            response = chat(messages)
            output_dict = output_parser.parse(response.content)
            st.session_state.question = output_dict.get('question')
            choices = [
                f"A: {output_dict.get('A')}",
                f"B: {output_dict.get('B')}",
                f"C: {output_dict.get('C')}",
                f"D: {output_dict.get('D')}"
            ]
            st.session_state.choices = choices
    
    st.markdown(st.session_state.question)
    if 'choices' in st.session_state and st.session_state.choices:
        answer_choice = st.radio("Pick one", st.session_state.choices+["I'm not sure"], key='answer_choice')
    else:
        answer_choice = None

    check_button = st.button("Check")
    
    if check_button and answer_choice:
        with st.spinner('Aiko is reviewing your response...'):
            chat = ChatOpenAI(model=gpt_model,openai_api_key=openai_api_key)
            prompt = ChatPromptTemplate.from_template(template=answer_template)
            if answer_choice=="I'm not sure":   
                response_instructions="I'm not sure what is the answer to {question}. Can you walk me through each choice? {choices}?".format(question=st.session_state.question,
                                                                                                                                              choices=st.session_state.choices)
            else:
                response_instructions="I think the answer to {question} is {answer_choice} from {choices}".format(question=st.session_state.question,
                                                                                                              choices=st.session_state.choices,
                                                                                                              answer_choice=answer_choice)
            messages = prompt.format_messages(response_instructions=response_instructions)
            response = chat(messages)
            st.markdown(response.content)
            if response.content[0]=='Y':
                st.session_state.correct_answers += 1
    
if __name__ == "__main__":
    if not openai_api_key.startswith('sk-'):
        st.info('Please enter your OpenAI API key on the left sidebar. You can create one at https://openai.com/blog/openai-api', icon='⚠')
    else:
        main()