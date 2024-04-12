from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts import (
    PromptTemplate
)


def prompt_generator_chart_type():  
    system_template = """
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
    system_message_prompt = PromptTemplate.from_template(system_template)

    human_template = """
        Human: 
        This is the dataset:

        {data}

        Create chart that analyze this specific topic: {topic}
        """
    human_message_prompt = PromptTemplate.from_template(human_template)

    full_template = """{system_prompt}

    {human_prompt}
    """
    full_prompt = PromptTemplate.from_template(full_template)

    input_prompts = [
        ("system_prompt", system_message_prompt),
        ("human_prompt", human_message_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_prompt, pipeline_prompts=input_prompts, input_variables=['num_charts','data', 'topic']
    )
    return pipeline_prompt


def prompt_generator_feature_engineering():
    system_template = """
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
    system_message_prompt = PromptTemplate.from_template(system_template)

    human_template = """
        Human: 
        This is the dataset:
        {data}

        This is the column names in the original dataset:
        {column_names}

        This is the visualization specs: 
        {vis_specs}
        """
    human_message_prompt = PromptTemplate.from_template(human_template)

    full_template = """{system_prompt}

    {human_prompt}
    """
    full_prompt = PromptTemplate.from_template(full_template)

    input_prompts = [
        ("system_prompt", system_message_prompt),
        ("human_prompt", human_message_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_prompt, pipeline_prompts=input_prompts, input_variables=['data', 'column_names', 'vis_specs']
    )
    return pipeline_prompt



