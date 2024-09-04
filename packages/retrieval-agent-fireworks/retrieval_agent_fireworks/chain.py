from typing import List

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_models.fireworks import ChatFireworks
from langchain_core.pydantic_v1 import BaseModel

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Pinecone as PC
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from pinecone import Pinecone

import os

MODEL_ID = "accounts/fireworks/models/mixtral-8x7b-instruct"

os.environ["FIREWORKS_API_KEY"] = "vLidwCkn02CPkKNPFz0fp353XFCAOIxFQXnlwk0VunynrnWX" # AI Model
os.environ['COHERE_API_KEY'] = 'fnG3bEeE72QgamxF2bXIupoHHwPBttKKHbhZvg09' # Embeddings Model
os.environ['PINECONE_API_KEY'] = 'a89c0db2-bf43-4d8b-ad20-6a78f6a460aa' # Vector Database
os.environ["TAVILY_API_KEY"] = "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"  # Search Engine

# Set up tool(s)
search = TavilySearchAPIWrapper()

description = """"A search engine optimized for comprehensive, accurate, \
and trusted results. Useful for when you need to answer questions \
about current events or about recent information. \
Input should be a search query. \
If the user is asking about something that you don't know about, \
you should probably use this tool to see if that can provide any information, \
Always say "Searching to Internet..." at the first of the answer and provide a new paragraph for the first answer"""

tavily_tool = TavilySearchResults(api_wrapper=search, description=description)

vectorstore_tools_description = (
    "A tool that looks up documentation about the Agent Memory"
    "Use this tool to took anything up about Agent, Prompt Engineering or Adversarial Attack LLM."
    "Given the following extracted parts of a long document and a question, create a final answer."
    'Always say "Thanks for asking!" at the Final Answer.'
    """if this tool does not provide relevant information just say "Sorry, I Don`t Now" at the output"""
    "User other tool to provide relevant information via the search engine"
)

vectordb = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
docsearch = PC.from_existing_index("workshop", embeddings)

retriever = docsearch.as_retriever()

vectorstore_tool = create_retriever_tool(retriever, "docstore", vectorstore_tools_description)

tools = [vectorstore_tool, tavily_tool]

# Set up LLM
llm = ChatFireworks(
    model=MODEL_ID,
    model_kwargs={
        "temperature": 0.5, # 0
        "max_tokens": 2048,
        "top_p": 1,
    },
    cache=True,
)

# setup ReAct style prompt
prompt = hub.pull("hwchase17/react-json")
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

# define the agent
model_with_stop = llm.bind(stop=["\nObservation"])

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
    }
    | prompt
    | model_with_stop
    | ReActJsonSingleInputOutputParser()
)


class InputType(BaseModel):
    input: str


# instantiate AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
).with_types(input_type=InputType)