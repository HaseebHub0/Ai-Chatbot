"""
AI Chatbot using Hugging Face Transformers
This module implements the core chatbot functionality using a pre-trained language model.
The chatbot uses DialoGPT for natural language understanding and generation.
"""

# Import required libraries for the transformer model and PyTorch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class AIChat:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        """
        Initialize the chatbot with a pre-trained model.
        This sets up the model, tokenizer, and prepares for GPU acceleration if available.

        Args:
            model_name (str): Name of the pre-trained model to use
                            Default: microsoft/DialoGPT-medium
        """
        # Initialize tokenizer and model from the pre-trained model with better settings
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Enable GPU acceleration if available, otherwise use CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        print(f"Model loaded and running on {self.device}")
        
        # Initialize conversation history and context management
        self.chat_history_ids = None
        self.max_history_tokens = 1024  # Limit context window to prevent overflow

    def generate_response(self, user_input, max_length=1000):
        """
        Generate an AI response to the user's input using the language model.
        The method handles tokenization, context management, and response generation.

        Args:
            user_input (str): The user's message to respond to
            max_length (int): Maximum length of the generated response in tokens
            
        Returns:
            str: The AI-generated response to the user's input
        """
        # Format input with role markers for better context
        formatted_input = f"User: {user_input}\nAssistant:"
        
        # Encode user input with special tokens
        inputs = self.tokenizer.encode(formatted_input, 
                                     return_tensors='pt',
                                     add_special_tokens=True).to(self.device)
        
        # If there's chat history, append it while respecting max length
        if self.chat_history_ids is not None:
            # Combine history with new input, but limit total length
            combined_length = self.chat_history_ids.shape[1] + inputs.shape[1]
            if combined_length > self.max_history_tokens:
                # Keep only the most recent history that fits
                history_to_keep = self.max_history_tokens - inputs.shape[1]
                self.chat_history_ids = self.chat_history_ids[:, -history_to_keep:]
            inputs = torch.cat([self.chat_history_ids, inputs], dim=-1)
        
        # Generate response with improved parameters
        with torch.no_grad():
            chat_history_ids = self.model.generate(
                inputs,
                max_length=max_length,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2,        # Reduced to allow some natural repetition
                do_sample=True,
                top_k=50,                      # Increased for more vocabulary variety
                top_p=0.92,                    # Increased for more diverse responses
                temperature=0.85,              # Balanced between creativity and coherence
                min_length=10,                 # Reduced to allow natural short responses
                repetition_penalty=1.2,        # Prevent repetitive responses
                length_penalty=1.0,            # Balanced response length
                early_stopping=True,           # Stop when natural ending is reached
                num_return_sequences=1         # Generate one response
            )
        
        # Store chat history
        self.chat_history_ids = chat_history_ids
        
        # Decode and return the response
        response = self.tokenizer.decode(chat_history_ids[:, inputs.shape[-1]:][0], 
                                       skip_special_tokens=True)
        return response

    def reset_chat(self):
        """
        Reset the chat history to start a new conversation.
        This clears the conversation context and starts fresh.
        """
        self.chat_history_ids = None
