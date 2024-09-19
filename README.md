# Innovative IT Support & Cybersecurity through Multi-Modal Large Language Models: Integrating Arabic Audio Response

## Project Overview

This project leverages **multi-modal large language models (LLMs)** to create an advanced **IT support and cybersecurity chatbot** for **METCLOUD**, designed specifically to handle **Arabic audio and text interactions**. The system combines **Retrieval-Augmented Generation (RAG)** with text-to-speech (TTS) and speech-to-text (STT) functionalities, ensuring efficient and contextually relevant support in Arabic, English, and French.

### Key Features

- **Arabic Audio Support**: The chatbot can transcribe Arabic speech inputs and provide spoken responses in Modern Standard Arabic and various Arabic dialects.
- **Multi-Language Support**: In addition to Arabic, the chatbot supports English and French interactions.
- **Retrieval-Augmented Generation (RAG)**: The chatbot combines generative AI with external knowledge retrieval, ensuring that the information provided is accurate and contextually appropriate.
- **Evaluation System**: A separate evaluation tool using the **RAGAS score** to assess model performance in terms of answer relevance, faithfulness, and context utilization.
- **User Feedback Mechanism**: The system gathers feedback from users to improve accuracy and performance over time.

## Technologies Used

- **Large Language Models (LLMs)**: Llama3.1, Llama3, Mistral, WizardLM2
- **RAG (Retrieval-Augmented Generation)**: To enhance the chatbot’s accuracy and reduce hallucinations.
- **Text-to-Speech (TTS) and Speech-to-Text (STT)**: AssemblyAI for transcription and gTTS for audio output.
- **LangChain**: Used for building RAG chains and managing the integration of models with document retrieval systems.
- **Hugging Face Embeddings**: For vectorizing text and supporting efficient retrieval.
- **Streamlit**: Provides the user interface for interacting with the chatbot and evaluation system.
- **Ollama**: For local deployment of LLMs and ensuring data privacy.
- **Chroma Vector Store**: For storing and managing the knowledge base.

## Setup Instructions

### Prerequisites

- **Python 3.8+**
- **Git**
- **Streamlit** (`pip install streamlit`)
- **LangChain** (`pip install langchain`)
- **Hugging Face Embeddings** (`pip install sentence-transformers`)
- **AssemblyAI** for transcription (API key required)
- **gTTS** (`pip install gTTS`)

### Usage
Chatbot Interface
Once the app is running, you can interact with the chatbot by either typing text or providing an Arabic audio input. The chatbot will process your request using RAG and return a relevant response, either as text or audio.

### Evaluation Tool
The evaluation tool allows you to test different models (Llama3.1, Mistral, WizardLM2, etc.) and configure hyperparameters like chunk size and overlap. The tool generates RAGAS scores for each configuration, helping you identify the most effective model.

### Future Work
Expand Knowledge Base: Increase the number of Arabic IT support records and cybersecurity content.
Handle More Dialects: Improve support for various Arabic dialects by fine-tuning language models.
Advanced Evaluation Metrics: Introduce more sophisticated evaluation techniques for Arabic-specific NLP tasks.

