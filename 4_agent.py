from langchain_huggingface import HuggingFaceEndpointEmbeddings, ChatHuggingFace
from langchain_perplexity import ChatPerplexity
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain import hub
from dotenv import load_dotenv
import os

load_dotenv()

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'https://api.weatherstack.com/current?access_key=f07d9636974c4120025fadf60678771b&query={city}'

  response = requests.get(url)

  return response.json()

llm = ChatPerplexity(model= 'sonar', api_key= os.getenv("PERPLEXITY_API_KEY"))

# Step 1: Pull the raw prompt
raw_prompt = hub.pull("hwchase17/react")

# Step 2: Rebuild it into a PromptTemplate (this strips out the `stop_sequences` metadata
#     which is what Perplexity was choking on)
prompt = PromptTemplate.from_template(raw_prompt.template)

# Step 3: Use this prompt to create the ReAct agent
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)

# What is the release date of Dhadak 2?
# What is the current temp of gurgaon
# Identify the birthplace city of Kalpana Chawla (search) and give its current temperature.

# Step 5: Invoke
response = agent_executor.invoke({"input": "What is the release date of Dhadak 2?"})
print(response)

print(response['output'])