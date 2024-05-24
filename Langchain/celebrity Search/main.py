## Integrate our code OpenAI API
import os

from constants import openai_key
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain.memory import ConversationBufferMemory


import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key

# streamlit framework

st.title("Celebrity Search With OPENAI API")
input_text = st.text_input("Search the text you want")

## OPEN AI LLMS
llm=OpenAI(temperature=0.8)

# Memory

person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='dob_history')
events_memory = ConversationBufferMemory(input_key='dob', memory_key='events_history')


# Prompt Template
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about {name}"

)

chain=LLMChain(llm=llm,prompt=first_input_prompt,verbose=True, output_key="person",memory=person_memory)


second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="When is {person} born?"

)

chain2=LLMChain(llm=llm,prompt=second_input_prompt,verbose=True, output_key="dob",memory=dob_memory)

third_input_prompt = PromptTemplate(
    input_variables=['dob'],
    template="Mention 10 major events happened around {dob} in the World"

)

chain3=LLMChain(llm=llm,prompt=third_input_prompt,verbose=True, output_key="events",memory=events_memory)



parent_chain=SequentialChain(chains=[chain,chain2,chain3],input_variables=['name'],output_variables=['person','dob','events'], verbose=True)




if input_text:
    st.write(parent_chain({"name":input_text}))

    with st.expander('Person Name'):
        st.info(person_memory.buffer)

    with st.expander("Person's Date of Birth"):
        st.info(dob_memory.buffer)

    with st.expander("Major Events"):
        st.info(events_memory.buffer)
