import os
import io
import re
import csv  # Handling CSV file operations for data storage
from datetime import datetime  # Handling timestamps for saving data
import streamlit as st  # Streamlit for building the web app UI
from dotenv import load_dotenv  # Load environment variables from a .env file
from langchain.llms import Ollama  # Import LLM from LangChain for question-answering
from langchain.vectorstores import Chroma  # Chroma vector store for document retrieval
from langchain.embeddings import OpenAIEmbeddings  # OpenAI Embeddings for document embedding
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Splits large texts into smaller chunks
from langchain.chains import create_retrieval_chain  # Creates a retrieval chain for question answering
from langchain.chains.combine_documents import create_stuff_documents_chain  # Combines document pieces for better QA
from langchain.prompts import ChatPromptTemplate  # Creates prompts for the LLM
from langchain.document_loaders import CSVLoader  # Loads documents from a CSV file
from deep_translator import GoogleTranslator  # GoogleTranslator for translating text
from pydub import AudioSegment  # Handling audio data
from pydub.playback import play  # Playing audio data
from gtts import gTTS  # Google Text-to-Speech for generating audio
import tempfile  # For creating temporary files
import wave  # Handling .wav files for audio
import assemblyai as aai  # AssemblyAI for transcribing audio to text
import sounddevice as sd  # Handling sound recording from the microphone
import numpy as np  # Handling arrays and numerical operations
from datasets import Dataset, Features, Sequence, Value  # For handling datasets
from ragas import evaluate  # RAGAS for evaluation metrics of QA system
from ragas.metrics import faithfulness, answer_relevancy, context_utilization, context_precision, context_recall  # Evaluation metrics
from langchain_core.output_parsers import StrOutputParser  # Parsing output from the LLM
import time  # For handling delays and time-related functions
import pygame  # For playing audio using a different library (in addition to pydub)
import matplotlib.pyplot as plt  # For plotting charts and visual data
from st_audiorec import st_audiorec  # Microphone recording in Streamlit
from langchain_huggingface.embeddings import HuggingFaceEmbeddings  # HuggingFace embeddings for text
import requests  # For making HTTP requests (used for Perspective API)

# Load environment variables from .env file to access API keys securely
load_dotenv()

# Function to append a row of data to a CSV file
def append_to_csv(file_path, data):
    file_exists = os.path.isfile(file_path)  # Check if file already exists
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            # If the file doesn't exist, write the header row
            writer.writerow(['Time', 'Question', 'Answer', 'Feedback', 'Comments'])
        # Write the new data to the CSV file
        writer.writerow(data)

# Set environment variables for LangChain, AssemblyAI, and Perspective API using values from the .env file
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["ASSEMBLYAI_API_KEY"] = os.getenv("ASSEMBLYAI_API_KEY")
os.environ["PERSPECTIVE_API_KEY"] = os.getenv("PERSPECTIVE_API_KEY")

# Add custom CSS styles for Streamlit UI
st.markdown("""
    <style>
        .main-title { color: #359B6F; }  /* Main title color */
        .sub-title { color: #000054; }  /* Sub-title color */
        .gray-background {
            background-color: #f0f0f0;  /* Background color for gray boxes */
            padding: 20px;  /* Padding for the gray boxes */
            border-radius: 10px;  /* Rounded corners */
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }
    </style>
    """, unsafe_allow_html=True)

# Display the logo and title on the Streamlit app
st.image("C:\\LLM Master\\logo.jpg", width=200)  # Display company logo
st.markdown('<h1 class="main-title">METCLOUD - Question Answering System with RAG Evaluation</h1>', unsafe_allow_html=True)

# Creating two columns: one for explanation and one for the question-answer system
col1, col2 = st.columns([0.4, 0.6])

# First column: Introduction and example questions
with col1:
    st.markdown('<div class="gray-background"> <h4 class="sub-title"> Introduction </h4> <p> This application allows you to ask questions and receive answers using advanced AI models. You can record your questions via microphone, and the system will transcribe, translate, and answer them. Feedback is also collected to improve the system. </p>', unsafe_allow_html=True)
    st.markdown('</div> <br/>  <div class="gray-background"> <h5 class="sub-title"> Example Questions </h5>', unsafe_allow_html=True)
    st.write("""
    1. Define the terms Virus, Malware, and Ransomware.
    2. What is Phishing? Provide an example.
    3. What is SSL encryption?
    4. Define the terms Encryption and Decryption.
    5. What is a VPN and why is it used?
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Second column: Handling CSV and LLM setup
with col2:
    # Load the CSV file containing data for question-answering
    loader = CSVLoader(file_path="C:\\LLM Master\\Alldata_1000.csv")
    docs = loader.load()  
    st.write(f"Document loaded with {len(docs)} rows.")  # Display the number of rows loaded

    # Load the LLM (Large Language Model) using LangChain
    llm = Ollama(model="llama3.1", temperature=0.5)

    # Split large documents into smaller chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)  
    st.write(f"Number of splits created: {len(splits)}")  

    # Create vector store and retriever based on the document chunks
    if splits:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Use HuggingFace embeddings
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)  # Create Chroma vector store
        retriever = vectorstore.as_retriever()  # Create retriever to retrieve relevant chunks

        # Define the system's prompt for answering questions based on context
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise.\n\n{context}"
        )

        # Create prompt template for the LLM to respond to
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
        output_parser = StrOutputParser()  # Parser to handle the LLM's output

        # Create a chain for question answering (RAG system)
        question_answer_chain = create_stuff_documents_chain(llm, prompt, output_parser=output_parser)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Function to transcribe audio using AssemblyAI
def transcribe_audio_with_assemblyai(audio_path):
    aai.settings.api_key = os.environ["ASSEMBLYAI_API_KEY"]  
    transcriber = aai.Transcriber()  
    transcript = transcriber.transcribe(audio_path)  # Transcribe the audio file
    if transcript.status == aai.TranscriptStatus.error:
        return transcript.error  # Return an error if transcription failed
    else:
        return transcript.text  

# Regular expression for detecting URLs in questions (used for phishing detection)
URL_REGEX = r'(https?://\S+|www\.\S+)'

# Function to check if a question contains potential phishing or spam threats
def is_potential_threat(question):
    # Define keywords related to phishing, hacking, and malicious content
    blacklist_keywords = ['phishing', 'hack', 'ransomware', 'scam', 'virus', 'malware', 'password', 'urgent']
    # Check for sensitive keywords in the question
    if any(word in question.lower() for word in blacklist_keywords):
        return True, "Potential threat due to sensitive keywords."
    # Check for URLs (possible phishing attempt)
    if re.search(URL_REGEX, question):
        return True, "Potential threat due to presence of URL."
    return False, "No threat detected."

# Function to convert text to speech (TTS)
def text_to_speech(text, lang='ar'):
    try:
        tts = gTTS(text=text, lang=lang)  
        temp_audio_file = "temp_audio.mp3"  # Temporary file to store the generated audio
        tts.save(temp_audio_file) 
        pygame.mixer.init()  # Initialize pygame for audio playback
        pygame.mixer.music.load(temp_audio_file)  # Load the audio file into pygame
        pygame.mixer.music.play()  # Play the audio
        while pygame.mixer.music.get_busy():  
            pygame.time.Clock().tick(10)
        pygame.mixer.music.stop()  # Stop the audio playback
        pygame.mixer.quit()  # Quit pygame
        time.sleep(3)  # Small delay to ensure smooth execution
        os.remove(temp_audio_file)  # Remove the temporary file
    except Exception as e:
        st.write(f"An error occurred during text-to-speech: {e}")  

# Record audio input using st_audiorec (Streamlit component for audio recording)
wav_audio_data = st_audiorec()

# Main logic to handle audio input and question answering
if wav_audio_data:
    user_question = transcribe_audio_with_assemblyai(wav_audio_data)  # Transcribe the audio
    if user_question:
        translator = GoogleTranslator(source='auto', target='en')  # Set up a translator to translate text to English
        translated_question = translator.translate(user_question)  # Translate the transcribed question to English
        st.write("Translated question to English:", translated_question)  # Display the translated question

        # Check if the question contains any potential threats
        threat_detected, threat_reason = is_potential_threat(translated_question)
        if threat_detected:
            st.write(f"This question may be a potential threat: {threat_reason}.")  
        else:
            response = rag_chain.invoke({"input": translated_question})  # Get the response from the RAG system
            answer_in_english = response["answer"]  # Extract the answer

            # Limit the length of the answer to 5000 characters
            if len(answer_in_english) > 5000:
                answer_in_english = answer_in_english[:4997] + '...'

            # Translate the answer back to Arabic
            translated_answer = translator.translate(answer_in_english)
            st.write("Answer:", translated_answer)  # Display the translated answer

            # Convert the translated answer to speech and play it
            audio_file_path = text_to_speech(translated_answer)
            if audio_file_path:
                st.audio(audio_file_path, format='audio/mp3')  # Play the audio response

            relevant_docs = retriever.get_relevant_documents(translated_question)  # Retrieve relevant documents
            st.write("Relevant documents:", relevant_docs[0])  # Display relevant documents

            # Collect user feedback on the answer
            from streamlit_feedback import streamlit_feedback
            feedback = streamlit_feedback(
                feedback_type="faces",  # Collect feedback using faces (positive/neutral/negative)
                optional_text_label="[Optional] Please provide an explanation",
            )

            # If feedback is provided, append it to the CSV file
            if feedback:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get the current timestamp
                st.write("Thank you for your feedback!")
                feedback_data = [current_time,  translated_question, translated_answer, feedback.get('score', ''), feedback.get('text', '')]  # Feedback data
                append_to_csv('feedback_data.csv', feedback_data)  # Append feedback to the CSV file
    else:
        st.write("Could not transcribe the audio.")  
else:
    st.write("Please record your question using the microphone.")  
