#!/usr/bin/env python3
"""
FSM Generator RAG System for Matter Specification
Based on ProtocolGPT implementation but adapted for FSM generation from Matter specification documents
"""

import os
import sys
import logging
from typing import List, Dict, Any
from pathlib import Path

from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.document_loaders import BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Import our custom modules
from fsm_config import DEFAULT_MODEL_CONFIG, FSM_PATTERNS
from fsm_utils import enhance_query_with_context, FSMParser, FSMValidator, FSMGenerator
from advanced_fsm_generator import AdvancedFSMGenerator

class FSMGeneratorRAG:
    """
    FSM Generator using RAG approach for Matter Specification
    """
    
    def __init__(self, matter_spec_path: str, model_name: str = "deepseek-r1"):
        self.matter_spec_path = matter_spec_path
        self.model_name = model_name
        self.setup_logging()
        
        # Initialize components with config
        config = DEFAULT_MODEL_CONFIG
        self.llm = ChatOllama(
            model=model_name, 
            temperature=config["temperature"], 
            num_predict=config["max_tokens"]
        )
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.vector_store = None
        self.memory = MemorySaver()
        self.agent_executor = None
        
        # Load and process documents
        self._load_documents()
        self._create_retrieval_tool()
        self._create_agent()
        
        # Initialize advanced FSM generator
        self.advanced_generator = AdvancedFSMGenerator(self)
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('fsm_generator.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_documents(self):
        """Load and chunk Matter specification documents"""
        self.logger.info(f"Loading Matter specification from: {self.matter_spec_path}")
        
        # Load HTML document
        loader = BSHTMLLoader(self.matter_spec_path)
        docs = loader.load()
        
        # Split documents into chunks with config values
        config = DEFAULT_MODEL_CONFIG
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        all_splits = text_splitter.split_documents(docs)
        
        # Create vector store
        self.vector_store = InMemoryVectorStore(self.embeddings)
        self.vector_store.add_documents(documents=all_splits)
        
        self.logger.info(f"Loaded {len(all_splits)} document chunks")
        
    def _create_retrieval_tool(self):
        """Create retrieval tool for Matter specification"""
        @tool(response_format="content_and_artifact")
        def retrieve_matter_context(query: str):
            """
            Retrieve relevant context from Matter specification for FSM generation.
            Use this to find protocol states, transitions, events, and state machine definitions.
            Focus on device lifecycles, connection management, security procedures, and command processing.
            """
            config = DEFAULT_MODEL_CONFIG
            retrieved_docs = self.vector_store.similarity_search(query, k=config["k_retrieval"])
            serialized = "\n\n".join(
                (f"Source: {doc.metadata.get('source', 'Matter Specification')}\n"
                 f"Content: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs
            
        self.retrieve_tool = retrieve_matter_context
        
    def _create_agent(self):
        """Create the FSM generation agent"""
        # Create agent with memory
        self.agent_executor = create_react_agent(
            self.llm, 
            [self.retrieve_tool], 
            checkpointer=self.memory
        )
        
    def get_fsm_generation_prompt(self) -> str:
        """
        Get the strong prompt for FSM generation from Matter specification
        """
        return """You are an expert protocol analysis assistant specialized in generating Finite State Machines (FSMs) from the Matter specification. Your task is to analyze Matter protocol documentation and extract state machine definitions.

## Your Capabilities:
1. **Protocol Analysis**: Deep understanding of Matter protocol states, transitions, and events
2. **FSM Extraction**: Identify and formalize state machines from specification text
3. **State Identification**: Recognize protocol states like commissioning states, operational states, error states
4. **Transition Analysis**: Extract state transitions triggered by events, commands, or conditions
5. **Event Recognition**: Identify protocol events that cause state changes

## FSM Generation Guidelines:

### 1. State Identification:
- Look for explicit state definitions in the Matter specification
- Identify implicit states from protocol behavior descriptions
- Common Matter states include: Initial, Commissioning, Operational, Error, Shutdown, Idle
- Device-specific states: Pairing, Authenticated, Configured, etc.

### 2. Transition Analysis:
- Find triggers: commands, events, timeouts, errors
- Identify conditions: security checks, device capabilities, network status
- Map source state → trigger/condition → destination state

### 3. Event Recognition:
- Protocol commands (e.g., commission, configure, reset)
- Network events (connection, disconnection, timeout)
- Security events (authentication success/failure)
- Device events (ready, error, capability announcement)

### 4. FSM Output Format:
Generate FSMs in this structured format:

```
FSM_NAME: [Descriptive name]
DESCRIPTION: [Brief description of what this FSM represents]

STATES:
- STATE_NAME: [Description]
- ...

INITIAL_STATE: [Name of initial state]

TRANSITIONS:
- FROM: [Source State] | EVENT: [Trigger] | CONDITION: [If any] | TO: [Destination State]
- ...

EVENTS:
- EVENT_NAME: [Description of the event/trigger]
- ...
```

### 5. Analysis Strategy:
When analyzing Matter specification:
1. First retrieve context about the specific protocol area you're analyzing
2. Look for state diagrams, state tables, or state descriptions
3. Identify the lifecycle of devices, connections, or processes
4. Extract error handling and recovery states
5. Map out normal operation flow vs. exception handling

### 6. Common Matter FSM Patterns:
- **Device Commissioning**: Initial → Discovery → Pairing → Provisioning → Operational
- **Connection Management**: Disconnected → Connecting → Connected → Authenticated
- **Command Processing**: Idle → Processing → Response → Idle
- **Error Handling**: Normal → Error → Recovery → Normal

## Instructions:
- Use the retrieve_matter_context tool to search for relevant information
- Be thorough in your analysis - look for both explicit and implicit state machines
- Provide clear, implementable FSM definitions
- Include error states and recovery mechanisms
- Focus on practical, implementable state machines

When asked to generate an FSM, start by retrieving relevant context from the Matter specification, then provide a detailed analysis and formal FSM definition."""

    def generate_fsm(self, query: str, thread_id: str = "fsm_generation") -> str:
        """
        Generate FSM based on user query using advanced generator
        """
        try:
            result = self.advanced_generator.generate_fsm(query, thread_id)
            
            if result["success"]:
                response = result["response"]
                validation = result["validation"]
                
                if validation["is_valid"]:
                    self.logger.info(f"Successfully generated valid FSM for query: {query}")
                    return response
                else:
                    self.logger.warning(f"Generated FSM has validation errors: {validation['errors']}")
                    return f"{response}\n\n⚠️ Validation Warnings:\n" + "\n".join(validation["errors"])
            else:
                self.logger.error(f"Failed to generate FSM: {result['response']}")
                return result["response"]
                
        except Exception as e:
            self.logger.error(f"Error in FSM generation: {e}")
            return f"Error generating FSM: {e}"
    
    def export_fsm(self, query: str, export_format: str = "json") -> str:
        """Export generated FSM in specified format"""
        result = self.advanced_generator.generate_fsm(query)
        return self.advanced_generator.export_fsm(result, export_format)
    
    def interactive_fsm_generation(self):
        """
        Interactive FSM generation interface with advanced features
        """
        print("🔄 Advanced FSM Generator for Matter Specification")
        print("=" * 60)
        print("Generate comprehensive FSMs from the Matter IoT specification!")
        print("\n📋 Available FSM Patterns:")
        for pattern_name, pattern_info in FSM_PATTERNS.items():
            print(f"  • {pattern_name}: {pattern_info['description']}")
        
        print("\n💡 Example requests:")
        print("  • 'Generate FSM for device commissioning process'")
        print("  • 'Create state machine for Matter device connection lifecycle'")
        print("  • 'Extract FSM for security handshake in Matter devices'")
        print("  • 'Generate command processing state machine'")
        
        print("\n🛠️  Commands:")
        print("  • Type 'export json' after generating an FSM to export as JSON")
        print("  • Type 'export python' to export as Python enum")
        print("  • Type 'export mermaid' to export as Mermaid diagram")
        print("  • Type 'patterns' to see available FSM patterns")
        print("  • Type 'quit' to exit")
        print("=" * 60)
        
        thread_id = "interactive_session"
        last_query = None
        
        while True:
            try:
                user_input = input("\n🤖 Your FSM request: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                    
                if user_input.lower() == 'patterns':
                    self._show_patterns()
                    continue
                    
                if user_input.lower().startswith('export ') and last_query:
                    export_format = user_input.split()[1]
                    try:
                        exported = self.export_fsm(last_query, export_format)
                        print(f"\n📄 Exported FSM ({export_format}):\n{exported}")
                    except Exception as e:
                        print(f"❌ Export error: {e}")
                    continue
                    
                if not user_input:
                    continue
                    
                print("\n🔍 Analyzing Matter specification...")
                response = self.generate_fsm(user_input, thread_id)
                print(f"\n📋 Generated FSM:\n{response}")
                last_query = user_input
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                self.logger.error(f"Interactive session error: {e}")
    
    def _show_patterns(self):
        """Show available FSM patterns"""
        print("\n📋 Available FSM Patterns:")
        for pattern_name, pattern_info in FSM_PATTERNS.items():
            print(f"\n• {pattern_name.upper()}:")
            print(f"  Description: {pattern_info['description']}")
            print(f"  Typical States: {', '.join(pattern_info['typical_states'])}")
            print(f"  Key Events: {', '.join(pattern_info['key_events'])}")


def main():
    """Main function"""
    matter_spec_path = "d:/Thesis_Matter_2025/Codes/demo/Matter_Specification.html"
    
    if not os.path.exists(matter_spec_path):
        print(f"❌ Matter specification not found at: {matter_spec_path}")
        return
        
    try:
        print("🚀 Initializing FSM Generator RAG System...")
        fsm_generator = FSMGeneratorRAG(matter_spec_path)
        
        print("✅ System initialized successfully!")
        fsm_generator.interactive_fsm_generation()
        
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        logging.error(f"System initialization error: {e}")


if __name__ == "__main__":
    main()
