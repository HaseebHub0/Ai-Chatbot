# AI Chatbot with Hugging Face Transformers

A sophisticated yet simple-to-use AI chatbot built with Python, leveraging the power of Hugging Face's Transformers library and Microsoft's DialoGPT model. This chatbot features a clean web interface built with Streamlit, making it easy to interact with the AI.

## Features

- 🤖 Natural language understanding using DialoGPT
- 💬 Interactive web-based chat interface
- 🔄 Conversation history management
- 🚀 GPU acceleration support
- 🎯 Configurable response parameters
- 🌐 Easy-to-use web interface

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ai-chatbot.git
   cd ai-chatbot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the chatbot:**
   ```bash
   cd chatbot
   streamlit run app.py
   ```

## Configuration

You can customize the chatbot's behavior by modifying parameters in `model.py`:

- `temperature`: Controls response randomness (0.7 default)
- `top_k`: Limits vocabulary for next word prediction (50 default)
- `top_p`: Nucleus sampling parameter (0.9 default)
- `max_length`: Maximum response length (1000 tokens default)

## Project Structure

```
.
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
└── chatbot/
    ├── model.py          # AI model implementation
    └── app.py            # Streamlit web interface
```

## Advanced Usage

### Model Customization

You can use different models by changing the `model_name` parameter when initializing the `AIChat` class:

```python
chatbot = AIChat(model_name="microsoft/DialoGPT-large")  # For a larger model
```

### Response Generation Settings

Adjust response generation parameters in `generate_response()`:

```python
response = chatbot.generate_response(
    user_input,
    max_length=2000,  # Longer responses
    temperature=0.9   # More creative responses
)
```

## Contributing

Feel free to open issues or submit pull requests with improvements. We appreciate your contributions!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Microsoft DialoGPT](https://huggingface.co/microsoft/DialoGPT-medium)
- [Streamlit](https://streamlit.io/)