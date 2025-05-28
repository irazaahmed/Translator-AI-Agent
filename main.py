from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, Runner
from dotenv import load_dotenv
import streamlit as st
import asyncio
import os

# Load environment variables
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    st.error("Please set the GEMINI_API_KEY environment variable.")
    st.stop()

# Initialize asynchronous OpenAI client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Configure the OpenAI Chat Completions model
model_version = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

# Set up the RunConfig
config = RunConfig(
    model=model_version,
    model_provider=external_client,
    tracing_disabled=True
)

# Define the Translator Agent
translator_agent = Agent(
    name="Translator Agent",
    instructions="You are a helpful translator. Translate the input text, sentences, words into English."
)

# Streamlit UI
st.title("Translation Agent")
st.write("This is the best AI Agent for translating any text into English")
st.subheader("Input Text")
user_input = st.text_area("Enter text to translate: ", height=100)

# Async translation function
async def translate_text(input_text):
    try:
        # Use the Runner to execute the async operation
        response = await Runner.run(
            translator_agent,
            input=input_text,
            run_config=config
        )
        return response.final_output
    except Exception as e:
        return f"Error: {e}"

# Button logic for translation
if st.button("Translate"):
    if user_input.strip():
        with st.spinner("Translating..."):
            # Run the async function using asyncio
            translation = asyncio.run(translate_text(user_input))
            if "Error" in translation:
                st.error(translation)
            else:
                st.subheader("Translation Result")
                st.write(translation)
    else:
        st.warning("Please enter some text to translate.")


# Footer
st.markdown("---")
st.write("Created by [Ahmed Raza](https://www.linkedin.com/in/iahmedraza4/)")