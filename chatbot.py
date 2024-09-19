
# This version is working well and try use mic
import os
import io
import re
import csv  # Import for handling CSV operations
from datetime import datetime  # Import for handling timestamps
import streamlit as st
from dotenv import load_dotenv
from langchain.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import CSVLoader
from deep_translator import GoogleTranslator
from pydub import AudioSegment
from pydub.playback import play
from gtts import gTTS
import tempfile
import wave
import assemblyai as aai
import sounddevice as sd
import numpy as np
from datasets import Dataset, Features, Sequence, Value
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_utilization, context_precision, context_recall
from langchain_core.output_parsers import StrOutputParser
import time
import pygame
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from st_audiorec import st_audiorec
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import requests  # For making API requests to Google's Perspective API



# Load environment variables
load_dotenv()

def append_to_csv(file_path, data):
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write header if file does not exist
            writer.writerow(['Time', 'Question', 'Answer', 'Feedback', 'Comments'])
        writer.writerow(data)


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["ASSEMBLYAI_API_KEY"] = os.getenv("ASSEMBLYAI_API_KEY")
os.environ["PERSPECTIVE_API_KEY"] = os.getenv("PERSPECTIVE_API_KEY")


st.markdown("""
    <style>
        .main-title {
            color: #359B6F;
        } 
        
        .sub-title {
            color: #000054;
        }
        .gray-background {
            background-color: #f0f0f0;
            padding: 20px;
            border-radius: 10px;
            width: 100%; 
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }
        .stColumn  {
           padding: 10px;
        
        }
    </style>
    """, unsafe_allow_html=True)

# Streamlit UI
st.image("C:\\LLM Master\\logo.jpg", width=200)  
st.markdown('<h1 class="main-title">METCLOUD - Question Answering System with RAG Evaluation</h1>', unsafe_allow_html=True)

col1, col2 = st.columns([0.4, 0.6])
# Custom CSS for background color of col1

with col1:
    st.markdown('<div class="gray-background"> <h4 class="sub-title"> Introduction </h4> <p> This application allows you to ask questions and receive answers using advanced AI models. You can record your questions via microphone, and the system will transcribe, translate, and answer them. Feedback is also collected to improve the system. </p>', unsafe_allow_html=True)
   
   
    st.markdown(' </div> <br/>  <div class="gray-background"> <h5 class="sub-title"> Example Questions </h5>', unsafe_allow_html=True)
    st.write("""
    1. Define the terms Virus, Malware, and Ransomware.?
    2. What is Phishing? Provide an example?
    3. What is SSL encryption?
    4. Define the terms Encryption and Decryption?
    5. What is a VPN and why is it used?
    """)
    st.markdown('</div>', unsafe_allow_html=True)
with col2:

# Load CSV document
    loader = CSVLoader(file_path="C:\\LLM Master\\Alldata_1000.csv")
    docs = loader.load()
    st.write(f"Document loaded with {len(docs)} rows.")

# Load LLM model.
    
    llm = Ollama(model="llama3.1", temperature =0.5)

# Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    st.write(f"Number of splits created: {len(splits)}")
   
# Create vector store and retriever if there are splits
    if splits:
    #vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
   # retriever = vectorstore.as_retriever()
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()


    # Define system prompt and prompt template
            system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n{context}"
        )

            prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
        )

            output_parser = StrOutputParser()

    # Create question-answer chain and RAG chain
            question_answer_chain = create_stuff_documents_chain(llm, prompt, output_parser=output_parser)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Function to record audio
    


    # Function to transcribe audio using AssemblyAI
    def transcribe_audio_with_assemblyai(audio_path):
            aai.settings.api_key = os.environ["ASSEMBLYAI_API_KEY"]
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(audio_path)
            if transcript.status == aai.TranscriptStatus.error:
                return transcript.error
            else:
                return transcript.text

    URL_REGEX = r'(https?://\S+|www\.\S+)'

# Google Perspective API Key (you need to set up an API key)
    PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API_KEY")

# Function to check if the question contains a URL (potential phishing attempt)
    def contains_url(question):
            return re.search(URL_REGEX, question) is not None

# Function to classify spam using Google's Perspective API
       
# Main function for threat detection, including URL and spam classification
    def is_potential_threat(question):
            blacklist_keywords = ['phishing', 'hack', 'ransomware', 'scam', 'virus', 'malware', 'password', 'urgent']

    # Check for phishing-related keywords
            if any(word in question.lower() for word in blacklist_keywords):
                return True, "Potential threat due to sensitive keywords."

    # Check if question contains URLs (possible phishing attempt)
            if contains_url(question):
                 return True, "Potential threat due to presence of URL."

    # Check for spam using Google's Perspective API
               

            return False, "No threat detected."



    # Function to convert text to speech
    def text_to_speech(text, lang='ar'):
            try:
                tts = gTTS(text=text, lang=lang)
                temp_audio_file = "temp_audio.mp3"
                tts.save(temp_audio_file)
                pygame.mixer.init()
                pygame.mixer.music.load(temp_audio_file)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)

                pygame.mixer.music.stop()
                pygame.mixer.quit()

                time.sleep(3)
                os.remove(temp_audio_file)
            ##with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
               # tts.save(fp.name)
                #audio = AudioSegment.from_mp3(fp.name)
                #play(audio)
            except Exception as e:
             st.write(f"An error occurred during text-to-speech: {e}")
    from deep_translator import GoogleTranslator
    import traceback

   
    # Function to evaluate RAG system
    



    wav_audio_data = st_audiorec()
    

# Display the audio
    col_playback, col_space = st.columns([0.58, 0.42])
    #with col_playback:
        #st.audio(wav_audio_data, format='audio/wav')
    
    if wav_audio_data:
            user_question = transcribe_audio_with_assemblyai(wav_audio_data)
            if user_question:

                
                translator = GoogleTranslator(source='auto', target='en')

                # Translate question to English
                translated_question = translator.translate(user_question)
                st.write("translated_question to English:", translated_question)

                        # Check for potential threats or phishing attempts
                threat_detected, threat_reason = is_potential_threat(translated_question)
                if threat_detected:
                        st.write(f"This question may be a potential threat: {threat_reason}. Please proceed with caution.")
            
                else:
                        response = rag_chain.invoke({"input": translated_question})
                        answer_in_english = response["answer"]

                        # Ensure the answer length does not exceed the max length
                        if len(answer_in_english) > 5000:
                            answer_in_english = answer_in_english[:4997] + '...'

                        # Translate answer back to Arabic
                        translator = GoogleTranslator(source='en', target='ar')
                        translated_answer = translator.translate(answer_in_english)
                        

                        st.write("Answer:", translated_answer)
                        #text_to_speech(translated_answer)

                        audio_file_path = text_to_speech(translated_answer)
                        if audio_file_path:
                            st.audio(audio_file_path, format='audio/mp3')

                        relevant_docs = retriever.get_relevant_documents(translated_question)
                        st.write("relevant_docs", relevant_docs[0])

                    #if st.session_state.answer_generated:
                        from streamlit_feedback import streamlit_feedback
                        feedback = streamlit_feedback(
                            feedback_type="faces",
                            optional_text_label="[Optional] Please provide an explanation",
                            )
                        #feedback
                        if feedback:
                        # Get current time
                                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                st.write("Thank you for your feedback!")
                        
                        # Feedback data
                                feedback_data = [
                                current_time, 
                                feedback.get('score', ''), 
                                feedback.get('text', '')
                                ]
                        
                        # Append to CSV
                                append_to_csv('feedback_data.csv', feedback_data)
                                # Reset state after feedback is submitted
                        
    
            else:
                    st.write("Could not transcribe the audio.")
    else:
                    st.write("Please enter a question.")

   

  