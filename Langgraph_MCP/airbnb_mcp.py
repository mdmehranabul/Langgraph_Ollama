from typing_extensions import TypedDict,Annotated
from typing import List
import os
import re
import operator
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen3"

llm = ChatOllama(model=MODEL_NAME, base_url=BASE_URL)

class AgentState(TypedDict):
    messages : Annotated[list, operator.add]


from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio

async def get_tools():

    client = MultiServerMCPClient(
        {
            "airbnb": {
                "command": "npx",
                "args": ["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"],
                "transport": "stdio"
            }
        }
    )

    tools = await client.get_tools()

    # print(f"Loaded {len(tools)}")
    # print(f"Tools available: {tools}")

    return tools


async def agent_node(state: AgentState):
    tools = await get_tools()

    llm_with_tools = llm.bind_tools(tools)

    response = llm_with_tools.invoke(state['messages'])

    return {'messages': [response]}

async def create_agent():
    tools = await get_tools()

    builder = StateGraph(AgentState)
    builder.add_node('agent',agent_node)
    builder.add_node('tools',ToolNode(tools))

    builder.add_edge(START, 'agent')
    builder.add_edge('tools', 'agent')

    builder.add_conditional_edges('agent',tools_condition)

    graph = builder.compile()

    return graph


async def search(query):
    agent = await create_agent()
    result = await agent.ainvoke({'messages':[HumanMessage(query)]})

    response = result['messages'][-1].content

    print(response)

    return response




if __name__ =="__main__":
    query = "Show me premium hotels for party in Mumbai with check in on 31st January 2026 and checkout of 2nd feb 2026"
    asyncio.run(search(query))