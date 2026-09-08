from typing import List, Dict, Any, Optional
import os
import requests

class LLMProcessor:
    def __init__(self, model_name: Optional[str] = None, temperature: float = 0.7,
                 base_url: Optional[str] = None, timeout: float = 120):
        """Configure a local Ollama model; no OpenAI API key is required.

        Model and server default to LLM_MODEL and OLLAMA_BASE_URL environment
        variables, or Qwen3.5 4B on localhost. Timeout is in seconds.
        """
        self.model_name = model_name or os.getenv("LLM_MODEL", "qwen3.5:4b-q4_K_M")
        self.temperature = temperature
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

        # Default system prompt
        self.default_system_prompt = """
        You are a professional customer service representative, capable of accurately understanding user needs and providing assistance.
        Your responses should be concise, friendly, and helpful.
        If you are unsure about an answer, please honestly state that you don't know rather than providing potentially inaccurate information.
        """
    
    def _chat(self, messages: List[Dict[str, str]]) -> str:
        """Call the native Ollama API and return only the answer text."""
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "think": False,
                    "options": {"temperature": self.temperature},
                },
                timeout=self.timeout,
            )
            if response.status_code == 404:
                raise RuntimeError(
                    f"Ollama model '{self.model_name}' was not found. "
                    f"Run: ollama pull {self.model_name}"
                )
            response.raise_for_status()
        except requests.Timeout as exc:
            raise RuntimeError("Ollama timed out. Try again after the model loads or increase timeout.") from exc
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. Start it with: ollama serve"
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        try:
            content = response.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError("Ollama returned an invalid chat response.") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Ollama returned an empty chat response.")
        return content

    def generate_response(self, prompt: str, 
                         conversation_history: Optional[List[Dict[str, str]]] = None, 
                         system_prompt: Optional[str] = None) -> str:
        """
        Generate response
        
        Args:
            prompt: User input
            conversation_history: Conversation history
            system_prompt: System prompt
            
        Returns:
            Generated response
        """
        # Use default system prompt or custom prompt
        system_content = system_prompt if system_prompt else self.default_system_prompt
        
        # Build message list
        messages = [{"role": "system", "content": system_content}]
        
        # Add conversation history
        if conversation_history:
            for message in conversation_history:
                if message["role"] in ("user", "assistant"):
                    messages.append({"role": message["role"], "content": message["content"]})

        # Add current user input
        messages.append({"role": "user", "content": prompt})
        
        # Generate response
        return self._chat(messages)
    
    def customize_for_call_center(self) -> None:
        """
        Customize LLM for call center scenarios
        """
        self.default_system_prompt = """
        You are a professional call center customer service representative, capable of handling various customer inquiries and issues.
        
        Please follow these guidelines:
        1. Maintain a professional, friendly, and polite attitude
        2. Provide clear and concise answers, avoiding lengthy explanations
        3. Proactively offer relevant information, but don't oversell
        4. If you need more information to answer a question, politely ask for it
        5. If you cannot resolve the customer's issue, offer the option to escalate to a human agent
        
        Remember, your goal is to efficiently resolve customer issues while providing a good customer experience.
        """
    
    def customize_for_lead_generation(self) -> None:
        """
        Customize LLM for lead generation scenarios
        """
        self.default_system_prompt = """
        You are a professional sales representative, responsible for initial communication with potential customers and collecting information.
        
        Please follow these guidelines:
        1. Introduce yourself and your company in a friendly manner
        2. Inquire about potential customers' needs and pain points
        3. Briefly introduce how relevant products or services can solve their problems
        4. Collect key information (such as contact details, best time to contact, etc.)
        5. Suggest next steps (such as arranging a demonstration, sending materials, etc.)
        
        Remember, your goal is to establish an initial relationship and collect sufficient information for follow-up, not to complete the sale in the first conversation.
        """
    
    def analyze_conversation(self, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Analyze conversation content and extract key information
        
        Args:
            conversation_history: Conversation history
            
        Returns:
            Analysis results containing key information
        """
        # Build analysis prompt
        analysis_prompt = """
        Please analyze the following conversation and extract the following information:
        1. Customer's main issues or needs
        2. Customer's emotional state
        3. Key information points (such as product interest, budget considerations, etc.)
        4. Suggested follow-up actions
        
        Conversation content:
        """
        
        # Add conversation history to prompt
        for message in conversation_history:
            role = "Customer" if message["role"] == "user" else "Assistant"
            analysis_prompt += f"\n{role}: {message['content']}"
        
        # Build messages
        messages = [
            {"role": "system", "content": "You are a professional conversation analysis expert, capable of extracting key information from conversations."},
            {"role": "user", "content": analysis_prompt}
        ]
        
        # Generate analysis
        response = self._chat(messages)
        
        # More structured processing logic can be added here
        # Currently simply returns text analysis results
        return {"analysis": response}