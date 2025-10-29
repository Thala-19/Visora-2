import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import warnings
warnings.filterwarnings('ignore')

import google.generativeai as genai

# api_key = os.getenv('GOOGLE_API_KEY')
api_key = "AIzaSyARJOiO901RqUrDU89whfXT06yuVAM4_gg"

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file or environment variables.")

chain_cache = None

def load_gemini(api_key):
    global chain_cache
    if chain_cache is None:
        genai.configure(api_key=api_key)
        chat_model = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-2.0-flash-lite")

        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        prompt = ChatPromptTemplate.from_messages([
            ("system", "your system prompt..."),
            ("user", "{input}")
        ])

        output_parser = StrOutputParser()

        chain_cache = prompt | chat_model | output_parser
    return chain_cache

def gemini_get_response():
    transcript_file_path = "transcript.txt"  # Assuming ttt.py is in the same folder as transcript.txt
    input_text_from_file = ""

    chain = load_gemini(api_key)
    try:
        with open(transcript_file_path, "r", encoding="utf-8") as f:
            input_text_from_file = f.read().strip()
        if not input_text_from_file:
            # print(f"Info: The file {transcript_file_path} was empty. The model might not have specific input.")
            input_text_from_file = "Hello, what can you tell me?"
    except FileNotFoundError:
        # print(f"Error: The file {transcript_file_path} was not found. The model will not have user input.")
        input_text_from_file = "File not found. What's on your mind?"

    # --- Invoke the Chain with the Input from the File ---
    if input_text_from_file: # Only invoke if there's some input
        # print(f"Sending to Gemini model: \"{input_text_from_file}\"")
        response = chain.invoke({"input": input_text_from_file})
        # print("\nGemini Model Response:")
        print(response)

        # --- Save the Response to output.txt (for tts.py) ---
        output_file_path = "output.txt"
        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(response)
            print(f"\nResponse saved to {output_file_path}")
        except Exception as e:
            print(f"Error saving response to {output_file_path}: {e}")
    else:
        print("No input text from transcript.txt to process.")

    return response