"""
Streamlit web interface for the AI Chatbot
This module provides a user-friendly web interface for interacting with the chatbot.
"""

import streamlit as st
from model import AIChat
import time
import sys

# Page configuration with improved styling
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background: #2e2e2e;
        border-left: 5px solid #4CAF50;
    }
    .bot-message {
        background: #1e1e1e;
        border-left: 5px solid #2196F3;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """
    Initialize session state variables for the Streamlit app.
    Creates a new chatbot instance and message history if they don't exist.
    """
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = AIChat()
    if "messages" not in st.session_state:
        st.session_state.messages = []

def main():
    """
    Main function to run the Streamlit app.
    Sets up the user interface, handles chat interactions, and manages the chat history.
    """
    initialize_session_state()

    # Set up the header and welcome message with improved styling
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://raw.githubusercontent.com/microsoft/DialoGPT/master/bot.png", width=100)
    with col2:
        st.title("🤖 AI Assistant")
        st.markdown("""
        <div style='background-color: #1e1e1e; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
            Welcome! I'm your AI assistant powered by DialoGPT. I can help you with:
            * General conversations and questions
            * Basic task assistance
            * Information and explanations
            * And much more!
        </div>
        """, unsafe_allow_html=True)
    
    # Add a settings sidebar
    with st.sidebar:
        st.header("⚙️ Chat Settings")
        temperature = st.slider("Response Creativity", 0.0, 1.0, 0.7, 
                              help="Higher values make responses more creative but less focused")
        max_length = st.slider("Maximum Response Length", 100, 1000, 500, 
                             help="Maximum number of tokens in the response")
        
        st.header("🎨 Theme")
        theme = st.selectbox("Select Theme", ["Dark", "Light"], 
                           help="Choose the chat interface theme")

    # Display chat history - shows all previous messages in the conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Handle new user input
    if prompt := st.chat_input("What would you like to talk about?"):
        # Store and display the user's message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate the AI response using our chatbot model
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):  # Show a loading spinner while generating
                response = st.session_state.chatbot.generate_response(prompt)
                st.write(response)  # Display the generated response
        # Store the assistant's response in chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Provide a button to reset/clear the conversation
    if st.button("Reset Chat"):
        st.session_state.chatbot.reset_chat()  # Clear the model's conversation memory
        st.session_state.messages = []  # Clear the displayed chat history
        st.experimental_rerun()  # Refresh the page to show the clean state

if __name__ == "__main__":
    main()
