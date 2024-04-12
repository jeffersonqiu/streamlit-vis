import pandas as pd
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain_community.callbacks import get_openai_callback

from dotenv import load_dotenv
load_dotenv()


def init_data(dataset):
    """
    Takes in .csv dataset as input and 
    returns pandas DataFrame and a 10-row sample of the data
    """
    if dataset:
        df = pd.read_csv(dataset)
    else: 
        df = pd.read_csv('data/games.csv')

    data_sample = df.head(10).to_csv()

    return df, data_sample


def init_model(temperature):
    """
    Takes OPENAI_API_KEY and temperature as input and 
    returns intialized model
    """
    try:
        chat_llm = ChatOpenAI(model='gpt-4-turbo-preview', temperature=temperature)
        return chat_llm
    except ValueError:
        return None
    

def generate_prompt(num_charts, data_sample, topic):
    """
    Takes in number of plots to generate and a sample of data as a csv string 
    and returns ChatModel prompt for data visualization.
    """

    system_template = """/
        The following is a conversation between a Human and an AI assistant expert on data visualization with perfect Python 3 syntax. The human will provide a sample dataset for the AI to use as the source. The real dataset that the human will use with the response of the AI is going to have several more rows. The AI assistant will only reply in the following JSON format: 

        {{ 
        "charts": [{{'title': string, 'chartType': string, 'parameters': {{...}}}}, ... ]
        }}

        Instructions:

        1. chartType must only contain methods of plotly.express from the Python library Plotly.
        2. The format for charType string: plotly.express.chartType.
        3. For each chartType, parameters must contain the value to be used for all parameters of that plotly.express method.
        4. There should 4 parameters for each chart.
        5. Do not include "data_frame" in the parameters.
        6. Only use parameters from the list of columns in the data_frame!! 
        7. There should be {num_charts} charts in total.
        """
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = """
        Human: 
        This is the dataset:

        {data}

        Create chart that analyze this specific topic: {topic}
        """
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    prompt = chat_prompt.format_prompt(num_charts=str(num_charts), data=data_sample, topic=topic).to_messages()
    
    return prompt


def generate_result(chat_llm, prompt):
    """
    Takes ChatModel and prompt as input and returns formatted output
    """

    with get_openai_callback() as cb:
            result = chat_llm(prompt)
    # total_token = cb.total_tokens
    # total_cost = cb.total_cost

    # format result to output
    # output = json.loads(result.content)

    # return (output, total_token, total_cost)
    return result.content

def generate_prompt_dt_processor(vis_specs, data_sample, column_names):
    """
    Takes in number of plots to generate and a sample of data as a csv string 
    and returns ChatModel prompt for data visualization.
    """

    system_template = """/
        Instructions:
        1. Read the visualization specs as given to you. Check on all variables in 'parameters'.
        2. If any of the variables in 'parameters' does not appear as a column in the original dataset, return pandas function which transforms the original dataset into a new dataset containing ALL variables in parameters.
        3. Return this pandas operations in string form. Only return the string to execute without any explanation! 
        4. If there are >1 line of code, split them with ';'
        5. Sometimes you need to rename the column to ensure ALL variables in 'parameters' are represented exactly in the final_df dataset. 
        6. Always end the answer with 'final_df = df'

        Assumptions:
        1. Assume that original dataframe is given as 'df'
        2. Assume that the columns in the original dataframe might not have the right dtypes. Adjust it first to accept the right dtypes.

        Do not do this:
        1. Do not use python``` code here ``` format. Directly return pandas function in text format.
        """
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = """
        Human: 
        This is the dataset:
        {data}

        This is the column names in the original dataset:
        {column_names}

        This is the visualization specs: 
        {vis_specs}
        """
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    prompt = chat_prompt.format_prompt(vis_specs=vis_specs, data=data_sample, column_names=column_names).to_messages()
    
    return prompt
    