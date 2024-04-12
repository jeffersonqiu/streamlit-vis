import streamlit as st
from utils import init_data, init_model
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

import importlib
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from utils_test import prompt_generator_chart_type, prompt_generator_feature_engineering

df, data_sample = init_data(None)
st.dataframe(df)
column_names = df.columns
final_df = None

if 'env' not in st.session_state:
    st.session_state.env = {'df': df, 'output': None, 'st': st}

# model
chat_llm = ChatOpenAI(model='gpt-4-turbo-preview')
user_prompt = st.text_area("Tell me variable(s) you want to explore")

st.markdown("## Visualizations")

st_callbacks = StreamlitCallbackHandler(st.container())

chart_type_chain = LLMChain(llm=chat_llm, 
                            prompt=prompt_generator_chart_type(), 
                            output_parser=JsonOutputParser(), 
                            output_key='vis_specs'
                            )

chart_types = chart_type_chain.run({
    "data":data_sample,
    "topic": user_prompt,
    "num_charts": 2
})


for i, chart in enumerate(chart_types['charts']):
    st.write("Chart Number: ", i+1)
    st.write(chart)
    params = chart['parameters']
    fe_chain = LLMChain(llm=chat_llm, prompt=prompt_generator_feature_engineering(), output_key='final_output')
    fe_code = fe_chain.run({
        "data": data_sample,
        "column_names": column_names,
        "vis_specs": chart
    })
    try:
        exec(fe_code)
        st.write('Successfully Executed Feature Engineering')
    except Exception as e:
        st.write(f"Error during execution: {e}")

    if final_df is not None:
        # st.write(df.head())
        # st.write(final_df.head())  # Using .head() to display just the first few rows
        params['data_frame'] = final_df

        chart_type = chart['chartType']
        px_module = importlib.import_module("plotly.express")
        chart_function = getattr(px_module, chart_type.split('.')[-1])  
        fig = chart_function(**params)

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.write("final_df was not defined.")

