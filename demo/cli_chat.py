#!/usr/bin/env python3
"""
Simple Ollama Chat following LangChain documentation examples
"""

from langchain_ollama.llms import OllamaLLM
from langchain_ollama.chat_models import ChatOllama
from langchain.schema import HumanMessage, AIMessage, SystemMessage

def simple_llm_example():
    """Simple OllamaLLM example"""
    print("=== OllamaLLM Example ===")
    
    # Initialize OllamaLLM
    llm = OllamaLLM(model="deepseek-r1")
    
    # Simple invoke
    response = llm.invoke("What is machine learning?")
    print(f"Response: {response}")
    
    return llm

def simple_chat_example():
    """Simple ChatOllama example"""
    print("\n=== ChatOllama Example ===")
    
    # Initialize ChatOllama
    chat = ChatOllama(model="deepseek-r1")
    
    # Create messages
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of France?")
    ]
    
    # Get response
    response = chat.invoke(messages)
    print(f"Response: {response.content}")
    
    return chat

def interactive_chat():
    """Interactive chat using ChatOllama"""
    print("\n=== Interactive Chat ===")
    
    # Initialize ChatOllama
    chat = ChatOllama(
        model="deepseek-r1",
        temperature=0.7,
        num_predict=512
    )
    
    # Message history
    messages = [
        SystemMessage(content="You are a helpful AI assistant.")
    ]
    
    print("Chat started! Type 'quit' to exit.")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Add user message
            messages.append(HumanMessage(content=user_input))
            
            # Get response
            response = chat.invoke(messages)
            
            # Add AI response to history
            messages.append(AIMessage(content=response.content))
            
            print(f"Assistant: {response.content}")
            
        except KeyboardInterrupt:
            print("\nChat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def streaming_example():
    """Example with streaming responses"""
    print("\n=== Streaming Example ===")
    
    llm = OllamaLLM(model="deepseek-r1")
    
    print("Streaming response:")
    for chunk in llm.stream("Tell me a short story about AI"):
        print(chunk, end="", flush=True)
    print()  # New line after streaming

def main():
    """Main function to demonstrate different usage patterns"""
    print("🚀 Ollama LangChain Examples")
    print("=" * 40)
    
    try:
        # # Example 1: Simple LLM
        # simple_llm_example()
        
        # # Example 2: Simple Chat
        # simple_chat_example()
        
        # # Example 3: Streaming
        # streaming_example()
        
        # Example 4: Interactive Chat
        interactive_chat()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure Ollama is running and the model is available.")
        print("Run: ollama serve")
        print("Run: ollama pull deepseek-r1")

if __name__ == "__main__":
    main()