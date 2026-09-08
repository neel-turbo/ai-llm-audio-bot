# ai-llm-audio-bot
# Intelligent Voice Customer Service Assistant

A Streamlit customer service assistant using **Qwen3.5 4B locally through Ollama** for chat and conversation analysis. It supports customer service and lead generation prompts, text input, and voice interaction.


## Create Local Python Env

```bash 
pip install python-dotenv
echo "Add All Env Keys" > .env
add API keys API_KEY=xyz123secretkey in .env File

import os
from dotenv import load_dotenv
load_dotenv(find_dotenv(usecwd=True)) 
api_key = os.getenv("API_KEY")
```


## Run with your local model

From the repository root, activate your Python 3.11 environment and install dependencies:

```bash
cd ai-llm-audio-bot
pip install -r requirements.txt
```

The default model is `qwen3.5:4b-q4_K_M`, served at `http://localhost:11434`. If Ollama is already running with this model installed, no additional model setup is needed. Otherwise, install [Ollama](https://ollama.com/download), start it, and download the model:

```bash
ollama serve
# In another terminal:
ollama pull qwen3.5:4b-q4_K_M
```

Optional configuration: copy `.env.example` to `.env` and adjust the values:

```dotenv
LLM_MODEL=qwen3.5:4b-q4_K_M
OLLAMA_BASE_URL=http://localhost:11434
SAMPLE_RATE=16000
GROQ_API=your_groq_api_key_here
```

Launch the app from this directory:

```bash
streamlit run app.py
```

Open **Text Interaction**, enter a question, and click **Send**. Chat and conversation analysis need no OpenAI key. Thinking is disabled in the Ollama request for quicker voice-assistant responses. The integration uses the [Ollama chat API](https://docs.ollama.com/api/chat) through the existing `requests` dependency.

## Use the processor directly

```python
from utils.llm_utils import LLMProcessor

llm = LLMProcessor()  # qwen3.5:4b-q4_K_M on localhost:11434
print(llm.generate_response("Hello! How can you help me?"))

history = [
    {"role": "user", "content": "I need help tracking my order."},
    {"role": "assistant", "content": "Could you share your order number?"},
]
print(llm.generate_response("It is order 1234.", conversation_history=history))
print(llm.analyze_conversation(history)["analysis"])
```

The processor reads `LLM_MODEL` and `OLLAMA_BASE_URL` from the environment; the Streamlit app also loads `.env`. You can pass `model_name`, `base_url`, `temperature`, and `timeout` directly to `LLMProcessor`.

## Voice features

Only the LLM runs locally. The existing voice features use:

- **Speech recognition:** [Groq transcription API](https://console.groq.com/docs/speech-to-text), using `whisper-large-v3-turbo` on Groq servers. Set `GROQ_API` in your environment or `.env` (`GROQ_API_KEY` is also accepted). No local speech model or OpenAI API key is needed. Internet access and available Groq quota are required. Restart Streamlit after adding or changing your key.
- **Speech output:** gTTS, requiring internet access. In the text tab, enable **Read response aloud** to use it.
- **Recording:** a microphone connected to the machine running Streamlit, using SoundDevice.

## Troubleshooting

- **Cannot connect to Ollama:** start Ollama with `ollama serve` and check `OLLAMA_BASE_URL`.
- **Model not found:** run `ollama list` and ensure `LLM_MODEL` matches the installed tag exactly, or run `ollama pull qwen3.5:4b-q4_K_M`.
- **Slow first response:** the model may need time to load. The processor waits up to 120 seconds per request; increase its `timeout` argument if needed.
- **Groq transcription fails:** the app displays the reason, including missing/invalid keys, rate limits, connection failures, or no detected speech. Check `GROQ_API` and your Groq account limits. Text chat remains local and does not use that key.
- **No speech output:** check internet access for gTTS; text chat can run without speech output.