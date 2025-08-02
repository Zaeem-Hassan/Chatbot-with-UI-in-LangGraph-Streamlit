from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage , SystemMessage
from langchain.chat_models import ChatOpenAI
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import os

load_dotenv()

# ✅ Use Together.ai endpoint like OpenAI
llm = ChatOpenAI(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    openai_api_key=os.getenv("TOGETHER_API_KEY"),
    openai_api_base="https://api.together.xyz/v1",
    temperature=0.3,
    max_tokens=100
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = [SystemMessage(content="Answer should be short.")] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}
# ✅ Build LangGraph
checkpointer = InMemorySaver()
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)
