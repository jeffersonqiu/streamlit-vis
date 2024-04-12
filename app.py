import streamlit as st
import utils
from dotenv import load_dotenv
load_dotenv()
import importlib
from langchain.chains.llm import LLMChain
import pandas as pd

# # API key
# OPENAI_API_KEY = st.text_input('OpenAI API Key')


# hyperparameters
temp = st.number_input("Temperature", min_value=0.0, max_value=0.9, value=0.0, step=0.1)
num_charts = st.slider("Number of Plots", min_value=1, max_value=10, value=1)


# data
dataset = st.file_uploader("Upload your dataset", accept_multiple_files=False, type='csv')
df, data_sample = utils.init_data(dataset)
st.dataframe(df)

if 'env' not in st.session_state:
    st.session_state.env = {'df': df, 'output': None, 'st': st}

# model
chat_llm = utils.init_model(temp)
user_prompt = st.text_area("Tell me variable(s) you want to explore")

from langchain_core.output_parsers import JsonOutputParser

if st.button("Generate") and user_prompt:
    if not chat_llm:
        st.error('Error: Please add a valid OpenAI API Key.', icon="🚨")
    else:
        st.markdown("## Visualizations")

        prompt = utils.generate_prompt(num_charts, df, user_prompt)

        # generate result
        # (output, total_token, total_cost) = utils.generate_result(chat_llm, prompt)
        output_ori = utils.generate_result(chat_llm, prompt)


        parser = JsonOutputParser()

        output = parser.invoke(output_ori)


        st.session_state.env['output'] = output


        for i in range(len(output['charts'])):
            chart_data = output['charts'][i]
            params = chart_data['parameters']

            # generate final_df
            vis_specs = [output['charts'][i]]
            prompt_2 = utils.generate_prompt_dt_processor(vis_specs, data_sample, df.columns)
            res = chat_llm(prompt_2).content

            # st.write(type(res))
            final_df = None
            try:
                exec(res)
                # exec(res_2,)
                st.write('successfully implemented exec')
                st.write(final_df)
            except Exception as e:
                st.write(f"Error during execution: {e}")


            # final_df = st.session_state.env.get('final_df', None)

            st.write(st.session_state.env)

            if final_df is not None:
                # st.write(df.head())
                # st.write(final_df.head())  # Using .head() to display just the first few rows
                params['data_frame'] = final_df

                chart_type = chart_data['chartType']
                px_module = importlib.import_module("plotly.express")
                chart_function = getattr(px_module, chart_type.split('.')[-1])  
                fig = chart_function(**params)

                st.plotly_chart(fig, use_container_width=True)

            else:
                st.write("final_df was not defined.")

            # params['data_frame'] = df

            # chart_type = chart_data['chartType']
            # px_module = importlib.import_module("plotly.express")
            # chart_function = getattr(px_module, chart_type.split('.')[-1])  
            # fig = chart_function(df, **params)

            # st.plotly_chart(fig, use_container_width=True)

            