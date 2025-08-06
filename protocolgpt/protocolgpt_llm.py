"""
ProtocolGPT LLM Setup and Management
Following ProtocolGPT's proven approach for LLM initialization and configuration
"""

import sys
import logging
from typing import Optional
from langchain_ollama.chat_models import ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from matter_clusters import PROTOCOLGPT_CONFIG

class StreamStdOut(StreamingStdOutCallbackHandler):
    """Custom streaming handler following ProtocolGPT pattern"""
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        sys.stdout.write(token)
        sys.stdout.flush()

    def on_llm_start(self, serialized, prompts, **kwargs):
        sys.stdout.write("🤖 ")

    def on_llm_end(self, response, **kwargs):
        sys.stdout.write("\n")
        sys.stdout.flush()

class ProtocolGPTLLM:
    """
    LLM manager following ProtocolGPT's approach
    Handles model initialization, configuration, and callbacks
    """
    
    def __init__(self, model_name: str = "llama3.1", config: Optional[dict] = None):
        self.model_name = model_name
        self.config = config or PROTOCOLGPT_CONFIG.copy()
        self.llm = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM
        self._setup_llm()
    
    def _setup_llm(self):
        """Setup language model following ProtocolGPT pattern"""
        print(f"🤖 Initializing {self.model_name}...")
        
        try:
            callbacks = CallbackManager([StreamStdOut()])
            self.llm = ChatOllama(
                model=self.model_name,
                temperature=self.config["temperature"],
                num_predict=self.config["max_tokens"],
                callback_manager=callbacks
            )
            print("✅ Language model initialized")
            
        except Exception as e:
            print(f"❌ Error initializing LLM: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure Ollama is running: ollama serve")
            print(f"2. Make sure the model is available: ollama pull {self.model_name}")
            print("3. Check if the model name is correct")
            raise
    
    def get_llm(self) -> ChatOllama:
        """Get the initialized LLM instance"""
        if self.llm is None:
            raise RuntimeError("LLM not initialized. Call _setup_llm() first.")
        return self.llm
    
    def update_config(self, new_config: dict):
        """Update configuration and reinitialize LLM if needed"""
        config_changed = False
        
        # Check if relevant config changed
        for key in ["temperature", "max_tokens"]:
            if key in new_config and new_config[key] != self.config.get(key):
                config_changed = True
                break
        
        self.config.update(new_config)
        
        # Reinitialize if config changed
        if config_changed:
            print("🔄 Configuration changed, reinitializing LLM...")
            self._setup_llm()
    
    def test_connection(self) -> bool:
        """Test if the LLM is working properly"""
        try:
            print("🧪 Testing LLM connection...")
            response = self.llm.invoke("Hello, respond with 'OK' if you can hear me.")
            print(f"✅ LLM test successful")
            return True
        except Exception as e:
            print(f"❌ LLM test failed: {e}")
            return False
    
    def get_model_info(self) -> dict:
        """Get information about the current model"""
        return {
            "model_name": self.model_name,
            "temperature": self.config["temperature"],
            "max_tokens": self.config["max_tokens"],
            "status": "initialized" if self.llm else "not_initialized"
        }

def create_llm(model_name: str = "llama3.1", config: Optional[dict] = None) -> ProtocolGPTLLM:
    """
    Factory function to create and initialize LLM following ProtocolGPT pattern
    
    Args:
        model_name: Name of the Ollama model to use
        config: Configuration dictionary (optional)
    
    Returns:
        Initialized ProtocolGPTLLM instance
    """
    return ProtocolGPTLLM(model_name, config)
