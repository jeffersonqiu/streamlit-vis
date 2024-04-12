import plotly.express as px
import streamlit as st
import importlib

# # Data for the chart
# data = {
#     "x": [3, 3, 4, 1, 2, 4, 1, 2, 0, 3],
# }

# # Parameters for the plot
# params = {
#     "nbins": 5,
#     "title": "Distribution of Home Club Goals",
#     "labels": {"x": "Goals"}
# }

# # Creating the histogram
# fig = px.histogram(data, 
#                    x="x", 
#                    nbins=params["nbins"], 
#                    labels=params["labels"])
# fig.update_layout(title=params["title"])

# # Display the plot
# fig.show()
# st.plotly_chart(fig, use_container_width=True)


import json
from utils import generate_prompt_dt_processor, init_data, init_model
from langchain_community.chat_models import ChatOpenAI
from langchain_community.callbacks import StreamlitCallbackHandler

dataset = st.file_uploader("Upload your dataset", accept_multiple_files=False, type='csv')
df, data_sample = init_data(dataset)

st.write(df.head())

with open('data/data_3.json', 'r') as file:
    output = json.load(file)

i = 0
st.write(output['charts'][i])

vis_specs = [output['charts'][i]]

prompt = generate_prompt_dt_processor(vis_specs, data_sample)

# st.write(prompt)

chat_llm = ChatOpenAI(model_name='gpt-4-turbo-preview')

res = chat_llm(prompt).content

st.write(res)

# Environment dictionary
if 'env' not in st.session_state:
    st.session_state.env = {'df': df}

# envi = {'df': df}


# Execute with error handling
try:
    exec(res, st.session_state.env)
except Exception as e:
    st.write(f"Error during execution: {e}")

# st.write(st.session_state.env)

final_df = st.session_state.env.get('final_df', None)

# Check if final_df was successfully defined
if final_df is not None:
    # st.write(df.head())
    # st.write(final_df.head())  # Using .head() to display just the first few rows
    chart_data = output['charts'][i]
    params = chart_data['parameters']
    params['data_frame'] = final_df

    chart_type = chart_data['chartType']
    px_module = importlib.import_module("plotly.express")
    chart_function = getattr(px_module, chart_type.split('.')[-1])  
    fig = chart_function(**params)

    st.plotly_chart(fig, use_container_width=True)

else:
    st.write("final_df was not defined.")

# fig = px.histogram(df, x='home_club_goals', nbins=5, title='Distribution of Home Club Goals', labels={'x': 'Goals'})
# st.plotly_chart(fig, use_container_width=True)



# chart_data = output['charts'][i]
# params = chart_data['parameters']
# params['data_frame'] = df

# chart_type = chart_data['chartType']
# px_module = importlib.import_module("plotly.express")
# chart_function = getattr(px_module, chart_type.split('.')[-1])  
# fig = chart_function(**params)

# st.plotly_chart(fig, use_container_width=True)