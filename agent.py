###########################
# LiveKit Pipeline Agent
###########################

import logging

from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)

from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, silero, openai

from langgraph_agent import LangGraphLLM, graph


logger = logging.getLogger("voice-agent")

# Define the prewarm function
def prewarm(proc: JobProcess):
    """
    Initialize resources before the worker starts processing jobs.
    """
    # Load the VAD model
    proc.userdata["vad"] = silero.VAD.load()

    # Initialize and store the LangGraph agent
    proc.userdata["graph"] = graph  # Assuming `graph` is defined elsewhere

async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are Lachlan's voice assistant for prospective employers. Your interface with users will be voice. "
            "You should answer questions about him in the third person. You should use short and concise responses, and avoid usage of unpronounceable punctuation. "
            "You have access to his cv and some other application documents. You can also search the web for more information. "
            "You should answer questions about his work experience, skills, and interests. If you are asked questions that do not concern his work experience, skills, or interests, you should state that this is not Lachlan's response and give a succinct answer before directing them to enquire about his work experience, skills, or interests. "
        ),
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    # Retrieve the prewarmed resources
    vad = ctx.proc.userdata["vad"]
    graph = ctx.proc.userdata["graph"]

    # Wrap your LangGraph agent in the LangGraphLLM class
    langgraph_llm = LangGraphLLM(graph)

    # Replace OpenAI LLM with your LangGraph agent
    assistant = VoicePipelineAgent(
        vad=vad,
        stt=deepgram.STT(),
        llm=langgraph_llm,  # Use your LangGraph agent here
        tts=openai.TTS(),  # You can keep using OpenAI TTS or replace it
        chat_ctx=initial_ctx,
    )

    assistant.start(ctx.room, participant)
    print("Assistant started")
    # Greet the user
    await assistant.say("Hey, I'm Lachlan's voice assistant. How can I help you today?", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )