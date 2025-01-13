###########################
# LangGraph Agent
###########################

from __future__ import annotations

import time
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import asyncio
from types import TracebackType
from typing import Literal, TypedDict, Annotated, AsyncIterator, Optional, Union, Any, List
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

from livekit.agents.llm.llm import LLM, LLMStream, ChatContext, ChatChunk, Choice, ChoiceDelta, LLMCapabilities, ToolChoice
from livekit.agents.llm import function_context
from livekit.agents.utils import aio
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env.local")

# Initialize memory and tools
memory = MemorySaver()

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize Qdrant client and vector store
qdrant_client = QdrantClient(
    os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Initialize the embedding model
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Create vector store instance
vector_store = Qdrant(
    client=qdrant_client,
    collection_name="cv_docs",  # Make sure this matches your collection name
    embeddings=embeddings,
)

# Create retriever from the Qdrant vector store
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_work_experience",
    "Search and return information about Lachlan's work experience.",
)

search_tool = TavilySearchResults(max_results=2)

# Create tools array and tools node
tools = [retriever_tool, search_tool]
tools_node = ToolNode(tools)

# Initialize LLM
llm = ChatAnthropic(model="claude-3-5-haiku-latest")
llm_with_tools = llm.bind_tools(tools)

system_prompt = """
You are a voice assistant for prospective employers of Lachlan Chavasse. He is a 25 year old in London, UK. Your interface with users will be voice. 
You should answer questions about him in the third person. You should use short and concise responses, and avoid usage of unpronounceable punctuation. 
You have access to his cv and some other application documents.
You should also search the web for more information. 
You should answer questions about his work experience, skills, and interests. If you are asked questions that do not concern his work experience, skills, or interests, you should state that this is not Lachlan's response and give a succinct answer before directing them to enquire about his work experience, skills, or interests. 
"""

async def agent(state: State):
    messages = state["messages"]
    
    # Add system message to the beginning of the messages list
    messages = [("system", system_prompt)] + messages
    
    response = await llm_with_tools.ainvoke(messages)

    return {
        "messages": [response]
    }

# Build the graph
builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_node("tools", tools_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")
graph = builder.compile(checkpointer=memory)

###########################
# Export Classes
###########################

class LangGraphLLM(LLM):
    def __init__(self, graph):
        super().__init__()
        self._graph = graph
        self._capabilities = LLMCapabilities(supports_choices_on_int=True)

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        fnc_ctx: Optional[function_context.FunctionContext] = None,
        temperature: Optional[float] = None,
        n: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = None,
        tool_choice: Union["ToolChoice", Literal["auto", "required", "none"]] = None,
    ) -> "LangGraphLLMStream":
        return LangGraphLLMStream(
            self,
            chat_ctx=chat_ctx,
            fnc_ctx=fnc_ctx,
            conn_options=conn_options,
            graph=self._graph
        )

    async def aclose(self) -> None:
        pass

class LangGraphLLMStream(LLMStream):
    def __init__(
        self,
        llm: "LangGraphLLM",
        *,
        chat_ctx: ChatContext,
        fnc_ctx: Optional[function_context.FunctionContext],
        conn_options: APIConnectOptions,
        graph,
    ):
        super().__init__(llm, chat_ctx=chat_ctx, fnc_ctx=fnc_ctx, conn_options=conn_options)
        self._graph = graph
        self._response_queue = asyncio.Queue()
        print("LangGraphLLMStream initialized")  # Debug

    async def _run(self) -> None:
        """Run the agent and put responses into the queue"""
        print("LangGraphLLMStream _run")  # Debug
        try:
            # Extract the latest user message from the chat context
            user_message = None
            for message in self._chat_ctx.messages:
                if message.role == "user":
                    user_message = message.content
                    print("User message: ", user_message)  # Debug
            if not user_message:
                raise ValueError("No user message found in chat context")

            # Track what we've sent to avoid duplicates
            sent_responses = set()

            # Stream responses from the LangGraph agent
            async for response in self._graph.astream(
                {"messages": [("user", user_message)]},
                {"configurable": {"thread_id": "1"}},
            ):
                print("LangGraph response: ", response)  # Debug
                
                if not response.get("agent"):
                    continue

                # Get the AIMessage from the response
                ai_message = response["agent"]["messages"][0]
                
                # Handle the content which could be a string or a list of message parts
                if isinstance(ai_message.content, str):
                    # This is likely a final response
                    text = ai_message.content
                    if text and text not in sent_responses:
                        chunk = ChatChunk(
                            request_id="langgraph",
                            choices=[Choice(delta=ChoiceDelta(role="assistant", content=text))]
                        )
                        await self._response_queue.put(chunk)
                        print("LangGraphLLMStream final chunk: ", chunk)  # Debug
                        sent_responses.add(text)

                elif isinstance(ai_message.content, list):
                    # For intermediate messages (like "let me search"), send them immediately
                    for part in ai_message.content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                # This is an intermediate message - send it right away
                                text = part.get("text")
                                if text and text not in sent_responses:
                                    chunk = ChatChunk(
                                        request_id="langgraph",
                                        choices=[Choice(delta=ChoiceDelta(role="assistant", content=text))]
                                    )
                                    # Send intermediate messages immediately
                                    await self._response_queue.put(chunk)
                                    print("LangGraphLLMStream intermediate chunk: ", chunk)  # Debug
                                    sent_responses.add(text)
                                    # Add a small delay to ensure intermediate messages are processed
                                    await asyncio.sleep(0.1)

        except Exception as e:
            print(f"Error in _run: {e}")  # Debug
            raise
        finally:
            # Signal that we're done
            await self._response_queue.put(None)

    async def __anext__(self) -> ChatChunk:
        """Get the next chunk from the queue"""
        chunk = await self._response_queue.get()
        if chunk is None:
            raise StopAsyncIteration
        return chunk

    def __aiter__(self) -> AsyncIterator[ChatChunk]:
        """Start the processing and return self as the iterator"""
        # self._task = asyncio.create_task(self._run())
        return self

    async def aclose(self) -> None:
        """Clean up resources"""
        if hasattr(self, '_task'):
            await aio.gracefully_cancel(self._task)
        await super().aclose()
