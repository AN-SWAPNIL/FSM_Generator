#!/usr/bin/env python3
"""
Matter Cluster FSM Extractor
Following ProtocolGPT's FSM inference approach combined with cluster-specific analysis
Uses LangChain ConversationalRetrievalChain with Llama3.1 for practical FSM extraction
"""

import os
import json
import logging
import re
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Core LangChain imports (following ProtocolGPT pattern)
from langchain_ollama.chat_models import ChatOllama
from langchain_community.document_loaders import UnstructuredHTMLLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

@dataclass
class MatterClusterInfo:
    """Information about a Matter cluster"""
    cluster_id: str
    cluster_name: str
    description: str
    category: str
    attributes: List[str]
    commands: List[str]
    events: List[str]

class StreamStdOut(StreamingStdOutCallbackHandler):
    """Custom streaming handler following ProtocolGPT pattern"""
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        import sys
        sys.stdout.write(token)
        sys.stdout.flush()

    def on_llm_start(self, serialized, prompts, **kwargs):
        import sys
        sys.stdout.write("🤖 ")

    def on_llm_end(self, response, **kwargs):
        import sys
        sys.stdout.write("\n")
        sys.stdout.flush()

class MatterClusterProtocolGPTExtractor:
    """
    Matter Cluster FSM extractor following ProtocolGPT's proven approach
    Combines code filtering, RAG retrieval, and conversational chains for FSM extraction
    """
    
    def __init__(self, spec_path: str, model_name: str = "llama3.1"):
        self.spec_path = spec_path
        self.model_name = model_name
        self.vector_store = None
        self.retriever = None
        self.llm = None
        
        # Configuration following ProtocolGPT defaults
        self.config = {
            "max_tokens": 4096,
            "chunk_size": 2056,
            "chunk_overlap": 256,
            "k": 4,
            "temperature": 0.2  # Low for factual FSM responses
        }
        
        # Matter clusters definition (cluster-based approach)
        self.matter_clusters = self._define_matter_clusters()
        
        # Matter-specific patterns for code filtering (following ProtocolGPT approach)
        self.matter_patterns = {
            "cluster": r'cluster|Cluster|CLUSTER',
            "state": r'state|State|STATE|status|Status',
            "command": r'command|Command|COMMAND|cmd|Cmd',
            "attribute": r'attribute|Attribute|ATTRIBUTE|attr|Attr',
            "event": r'event|Event|EVENT|callback|Callback'
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components (following ProtocolGPT pattern)
        self._setup_llm()
        self._setup_embeddings()
        self._load_and_process_spec()
        self._create_retriever()
    
    def _define_matter_clusters(self) -> Dict[str, MatterClusterInfo]:
        """Define Matter clusters based on specification"""
        return {
            "on_off": MatterClusterInfo(
                cluster_id="0x0006",
                cluster_name="On/Off",
                description="Basic on/off functionality for devices",
                category="lighting",
                attributes=["OnOff", "GlobalSceneControl", "OnTime", "OffWaitTime"],
                commands=["Off", "On", "Toggle"],
                events=["StateChanged"]
            ),
            "level_control": MatterClusterInfo(
                cluster_id="0x0008", 
                cluster_name="Level Control",
                description="Controls the level/brightness of devices",
                category="lighting",
                attributes=["CurrentLevel", "RemainingTime", "MinLevel", "MaxLevel"],
                commands=["MoveToLevel", "Move", "Step", "Stop", "MoveToLevelWithOnOff"],
                events=["LevelChanged"]
            ),
            "door_lock": MatterClusterInfo(
                cluster_id="0x0101",
                cluster_name="Door Lock",
                description="Controls door lock functionality",
                category="access_control", 
                attributes=["LockState", "LockType", "ActuatorEnabled", "AutoRelockTime"],
                commands=["LockDoor", "UnlockDoor", "SetCredential", "ClearCredential"],
                events=["DoorLockAlarm", "LockOperation", "LockOperationError"]
            ),
            "thermostat": MatterClusterInfo(
                cluster_id="0x0201",
                cluster_name="Thermostat",
                description="Controls HVAC thermostat functionality", 
                category="hvac",
                attributes=["LocalTemperature", "OccupiedCoolingSetpoint", "OccupiedHeatingSetpoint", "SystemMode"],
                commands=["SetpointRaiseLower", "SetWeeklySchedule", "GetWeeklySchedule"],
                events=["TemperatureChanged", "SetpointChanged"]
            ),
            "window_covering": MatterClusterInfo(
                cluster_id="0x0102",
                cluster_name="Window Covering",
                description="Controls window covering devices like blinds and shades",
                category="covering",
                attributes=["Type", "PhysicalClosedLimitLift", "PhysicalClosedLimitTilt", "CurrentPositionLift"],
                commands=["UpOrOpen", "DownOrClose", "StopMotion", "GoToLiftValue", "GoToTiltValue"],
                events=["PositionChanged", "TargetChanged"]
            ),
            "color_control": MatterClusterInfo(
                cluster_id="0x0300",
                cluster_name="Color Control",
                description="Controls color properties of color-capable lights",
                category="lighting",
                attributes=["CurrentHue", "CurrentSaturation", "CurrentX", "CurrentY", "ColorTemperatureMireds"],
                commands=["MoveToHue", "MoveHue", "StepHue", "MoveToSaturation", "MoveToColor"],
                events=["ColorChanged", "HueChanged", "SaturationChanged"]
            )
        }
    
    def _setup_llm(self):
        """Setup language model following ProtocolGPT pattern"""
        print(f"🤖 Initializing {self.model_name}...")
        callbacks = CallbackManager([StreamStdOut()])
        self.llm = ChatOllama(
            model=self.model_name,
            temperature=self.config["temperature"],
            num_predict=self.config["max_tokens"],
            callback_manager=callbacks
        )
        print("✅ Language model initialized")
    
    def _setup_embeddings(self):
        """Setup embeddings model following ProtocolGPT pattern"""
        print("🔤 Initializing embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("✅ Embeddings model initialized")
    
    def _get_matter_fsm_files(self, root_dir: str) -> List[str]:
        """
        Find Matter cluster-related files using pattern matching
        Following ProtocolGPT's code filtering approach
        """
        def get_all_files(folder):
            file_paths = []
            for root, directories, files in os.walk(folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
            return file_paths
        
        # File extensions to analyze (following ProtocolGPT)
        allow_files = ['.txt', '.js', '.mjs', '.ts', '.tsx', '.css', '.scss', '.less', '.html', '.htm', 
                       '.json', '.py', '.java', '.c', '.cpp', '.cs', '.go', '.php', '.rb', '.rs', 
                       '.swift', '.kt', '.scala', '.m', '.h', '.sh', '.pl', '.pm', '.lua', '.sql']
        
        files = get_all_files(root_dir)
        fsm_files = []
        
        # Combine all Matter patterns
        combined_pattern = '|'.join(self.matter_patterns.values())
        
        for filename in files:
            # Check file extension
            if any(filename.endswith(ext) for ext in allow_files):
                try:
                    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
                        text = file.read()
                    # Look for Matter cluster patterns
                    if re.search(combined_pattern, text, re.IGNORECASE):
                        fsm_files.append(filename)
                        print(f"📄 Found Matter-related file: {filename}")
                except Exception as e:
                    # Skip files that can't be read
                    continue
        
        return fsm_files
    
    def _load_and_process_spec(self):
        """Load and process Matter specification following ProtocolGPT pattern"""
        print(f"📄 Loading document: {self.spec_path}")
        
        if not Path(self.spec_path).exists():
            raise FileNotFoundError(f"Document not found: {self.spec_path}")
        
        # Load HTML specification
        loader = UnstructuredHTMLLoader(self.spec_path)
        docs = loader.load()
        
        if not docs:
            raise ValueError("No content loaded from document")
        
        print(f"📖 Loaded {len(docs)} document(s)")
        
        # Use language-aware splitting for better code analysis (following ProtocolGPT)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"],
            length_function=len,
        )
        
        all_splits = text_splitter.split_documents(docs)
        print(f"✂️  Split into {len(all_splits)} chunks")
        
        # Create vector store following ProtocolGPT incremental approach
        print("🗃️  Creating vector store...")
        if all_splits:
            self.vector_store = FAISS.from_documents([all_splits[0]], self.embeddings)
            for i, text in enumerate(all_splits[1:]):
                print(f'\r🗃️  Creating vector store ({i + 1}/{len(all_splits)})', end='')
                self.vector_store.add_documents([text])
                time.sleep(0.01)  # Small delay to prevent overwhelming
            print("\n✅ Vector store created and indexed")
        else:
            raise ValueError("No documents to create vector store")
    
    def _create_retriever(self):
        """Create retriever following ProtocolGPT pattern"""
        print("🔗 Setting up retriever...")
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.config["k"]}
        )
        print("✅ Retriever setup complete")
    
    def _get_cluster_fsm_prompts(self, cluster: MatterClusterInfo) -> Dict[str, str]:
        """
        Generate ProtocolGPT-style prompts for Matter cluster FSM extraction
        Following the paper's methodology for practical FSM inference
        """
        
        base_context = f"""
CLUSTER ANALYSIS: {cluster.cluster_name} ({cluster.cluster_id})
Category: {cluster.category}
Description: {cluster.description}
Attributes: {', '.join(cluster.attributes)}
Commands: {', '.join(cluster.commands) if cluster.commands else 'None (server-only)'}
Events: {', '.join(cluster.events)}
"""

        return {
            "file_paths": f"""{base_context}

Which files in the Matter specification define the {cluster.cluster_name} cluster's message types, protocol states, and state transitions? 

Look for files that contain:
- Message type definitions for {cluster.cluster_name}
- State definitions and state machines
- State transition logic
- Command handling for: {', '.join(cluster.commands)}
- Event generation for: {', '.join(cluster.events)}

Desired format:
{{
    "Related files": ["/path/to/messages", "/path/to/states", "/path/to/transitions"]
}}""",

            "states": f"""{base_context}

Can you provide a list of all the {cluster.cluster_name} cluster states defined in the Matter specification?

Focus on states related to:
- Device operational states (on/off, active/inactive, etc.)
- Command processing states (idle, processing, completed, error)
- Attribute change states
- Event handling states

Desired format:
{{
    "States": ["state1", "state2", "state3", ...]
}}""",

            "messages": f"""{base_context}

What are the message types defined for the {cluster.cluster_name} cluster in the Matter specification?

Include:
- Command messages: {', '.join(cluster.commands)}
- Response messages
- Event messages: {', '.join(cluster.events)}
- Error messages
- Status messages

Desired format:
{{
    "Messages": ["message1", "message2", "message3", ...]
}}""",

            "transitions": f"""{base_context}

The {cluster.cluster_name} cluster has various states and message types. When the device receives different messages, the device's state transitions to different states.

What are the possible state transitions for the {cluster.cluster_name} cluster? Analyze each state and determine:
- What messages can trigger transitions from that state
- What the destination state would be
- Any conditions or prerequisites

For each current state, identify all possible transitions.

Desired format:
{{
    "state1": [
        {{"receive_message": "message1", "next_state": "state2"}},
        {{"receive_message": "message2", "next_state": "state3"}}
    ],
    "state2": [
        {{"receive_message": "message3", "next_state": "state1"}}
    ]
}}""",

            "complete_fsm": f"""{base_context}

Generate a complete Finite State Machine (FSM) for the {cluster.cluster_name} cluster following this structure:

1. Extract all states for the cluster
2. Extract all message types for the cluster  
3. Define all state transitions with triggering messages

The FSM should be a comprehensive model showing:
- Initial states
- All possible states the device can be in
- All messages that can be received/sent
- All state transitions triggered by messages
- Final/terminal states if any

Desired format:
{{
    "cluster_name": "{cluster.cluster_name}",
    "cluster_id": "{cluster.cluster_id}",
    "states": ["state1", "state2", ...],
    "messages": ["message1", "message2", ...],
    "initial_states": ["initial_state"],
    "final_states": ["final_state"],
    "transitions": {{
        "state1": [
            {{"receive_message": "message1", "next_state": "state2"}},
            {{"receive_message": "message2", "next_state": "state3"}}
        ],
        "state2": [
            {{"receive_message": "message3", "next_state": "state1"}}
        ]
    }}
}}"""
        }
    
    def extract_cluster_fsm(self, cluster_name: str, query_type: str = "complete_fsm") -> Dict[str, Any]:
        """
        Extract FSM for specific cluster using ProtocolGPT's conversational approach
        """
        if cluster_name not in self.matter_clusters:
            raise ValueError(f"Unknown cluster: {cluster_name}. Available: {list(self.matter_clusters.keys())}")
        
        cluster = self.matter_clusters[cluster_name]
        prompts = self._get_cluster_fsm_prompts(cluster)
        
        if query_type not in prompts:
            raise ValueError(f"Unknown query type: {query_type}. Available: {list(prompts.keys())}")
        
        query = prompts[query_type]
        
        print(f"\n🔍 Extracting {query_type} for {cluster.cluster_name} cluster")
        print("=" * 60)
        
        # Create conversational retrieval chain following ProtocolGPT
        memory = ConversationSummaryMemory(
            llm=self.llm,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=self.retriever,
            memory=memory,
            return_source_documents=True,
            get_chat_history=lambda h: h,
        )
        
        print("🤔 Processing with conversational retrieval chain...")
        
        # Execute query
        result = qa_chain({"question": query})
        
        # Extract response
        answer = result.get("answer", "")
        source_docs = result.get("source_documents", [])
        
        print(f"\n🤖 FSM Analysis Generated")
        print("=" * 40)
        print(answer[:800] + "..." if len(answer) > 800 else answer)
        
        # Show source files
        if source_docs:
            file_paths = [doc.metadata.get("source", "Unknown") for doc in source_docs]
            print(f"\n📄 Source documents:")
            for file_path in set(file_paths):
                print(f"  - {file_path}")
        
        return {
            "cluster_name": cluster_name,
            "cluster_id": cluster.cluster_id,
            "query_type": query_type,
            "answer": answer,
            "raw_response": result,
            "source_documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in source_docs
            ],
            "modeling_framework": "ProtocolGPT"
        }
    
    def extract_all_cluster_fsms(self, cluster_name: str) -> Dict[str, Any]:
        """Extract all FSM components for a specific cluster"""
        if cluster_name not in self.matter_clusters:
            raise ValueError(f"Unknown cluster: {cluster_name}")
        
        cluster = self.matter_clusters[cluster_name]
        results = {}
        
        query_types = ["file_paths", "states", "messages", "transitions", "complete_fsm"]
        
        print(f"\n🚀 Extracting all FSM components for {cluster.cluster_name} cluster...")
        
        for query_type in query_types:
            try:
                print(f"\n--- Processing {query_type} ---")
                result = self.extract_cluster_fsm(cluster_name, query_type)
                results[query_type] = result
                time.sleep(1)  # Brief pause between queries
            except Exception as e:
                print(f"❌ Error extracting {query_type}: {e}")
                results[query_type] = {"error": str(e)}
        
        return {
            "cluster_name": cluster_name,
            "cluster_id": cluster.cluster_id,
            "extraction_results": results,
            "summary": {
                "total_queries": len(query_types),
                "successful_extractions": len([r for r in results.values() if "error" not in r]),
                "failed_extractions": len([r for r in results.values() if "error" in r])
            }
        }
    
    def chat_with_cluster(self, cluster_name: str):
        """
        Interactive chat session for cluster analysis
        Following ProtocolGPT's chat_loop pattern
        """
        if cluster_name not in self.matter_clusters:
            raise ValueError(f"Unknown cluster: {cluster_name}")
        
        cluster = self.matter_clusters[cluster_name]
        
        print(f"\n🤖 Interactive Matter Cluster Analysis")
        print(f"📋 Cluster: {cluster.cluster_name} ({cluster.cluster_id})")
        print(f"📝 Description: {cluster.description}")
        print(f"🏷️  Category: {cluster.category}")
        print("=" * 60)
        print("Ask questions about this cluster's FSM, states, transitions, etc.")
        print("Type 'exit', 'quit', or 'q' to end the session.")
        print("=" * 60)
        
        # Create conversational chain
        memory = ConversationSummaryMemory(
            llm=self.llm,
            memory_key="chat_history", 
            return_messages=True,
            output_key="answer"
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=self.retriever,
            memory=memory,
            return_source_documents=True,
            get_chat_history=lambda h: h,
        )
        
        while True:
            try:
                query = input(f"\n👉 [{cluster.cluster_name}] ").strip()
                
                if not query:
                    print("🤖 Please enter a question about the cluster")
                    continue
                
                if query.lower() in ('exit', 'quit', 'q'):
                    break
                
                # Add cluster context to the query
                contextualized_query = f"""
Analyzing the {cluster.cluster_name} cluster ({cluster.cluster_id}) in Matter specification.

Cluster Information:
- Description: {cluster.description}
- Category: {cluster.category}
- Attributes: {', '.join(cluster.attributes)}
- Commands: {', '.join(cluster.commands)}
- Events: {', '.join(cluster.events)}

Question: {query}
"""
                
                print("🤔 Analyzing...")
                result = qa_chain({"question": contextualized_query})
                
                # Show source documents
                source_docs = result.get("source_documents", [])
                if source_docs:
                    file_paths = [doc.metadata.get("source", "Unknown") for doc in source_docs]
                    print(f"\n📄 Source references:")
                    for file_path in set(file_paths):
                        print(f"  - {Path(file_path).name}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n👋 Chat session ended!")

def main():
    """Main function following ProtocolGPT pattern"""
    print("🚀 Initializing Matter Cluster ProtocolGPT FSM Extractor")
    print("="*60)
    
    spec_path = os.path.join(os.path.dirname(__file__), "Matter_Specification.html")
    
    if not os.path.exists(spec_path):
        print(f"❌ Matter specification not found at: {spec_path}")
        print("Please copy Matter_Specification.html to this directory")
        return
    
    try:
        extractor = MatterClusterProtocolGPTExtractor(
            spec_path=spec_path,
            model_name="llama3.1"
        )
        
        print("\n✅ Initialization complete!")
        print(f"\n📋 Available clusters ({len(extractor.matter_clusters)}):")
        for i, (cluster_key, cluster) in enumerate(extractor.matter_clusters.items(), 1):
            print(f"  {i}. {cluster.cluster_name} ({cluster.cluster_id}) - {cluster.category}")
        
        query_types = ["file_paths", "states", "messages", "transitions", "complete_fsm"]
        print(f"\n🔧 Available query types:")
        for i, query_type in enumerate(query_types, 1):
            print(f"  {i}. {query_type}")
        
        print("\n" + "="*60)
        print("🚀 Matter Cluster ProtocolGPT FSM Extraction Assistant")
        print("="*60)
        print("Extract FSMs using ProtocolGPT's proven methodology!")
        print("Commands:")
        print("  - 'cluster_name query_type' (e.g., 'on_off complete_fsm')")
        print("  - 'all cluster_name' to extract all components for a cluster")
        print("  - 'chat cluster_name' for interactive analysis")
        print("  - 'batch' to extract complete FSMs for all clusters")
        print("  - 'quit', 'exit' to exit")
        print("="*60)
        
        results = []
        
        while True:
            try:
                user_input = input("\n🤖 Enter command: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if user_input.lower() == 'batch':
                    print("\n🚀 Batch processing: extracting complete FSMs for all clusters...")
                    for cluster_name in extractor.matter_clusters.keys():
                        try:
                            print(f"\n{'='*40}")
                            print(f"Processing {cluster_name}")
                            print(f"{'='*40}")
                            
                            result = extractor.extract_cluster_fsm(cluster_name, "complete_fsm")
                            results.append(result)
                            
                            # Save individual result
                            os.makedirs("protocolgpt_fsm_results", exist_ok=True)
                            filename = f"protocolgpt_fsm_results/{cluster_name}_complete_fsm.json"
                            with open(filename, 'w', encoding='utf-8') as f:
                                json.dump(result, f, indent=2, ensure_ascii=False)
                            print(f"💾 Saved: {filename}")
                            
                        except Exception as e:
                            print(f"❌ Error processing {cluster_name}: {e}")
                    continue
                
                parts = user_input.split()
                if len(parts) == 2:
                    if parts[0] == 'all':
                        cluster_name = parts[1]
                        if cluster_name in extractor.matter_clusters:
                            result = extractor.extract_all_cluster_fsms(cluster_name)
                            results.append(result)
                            
                            # Save comprehensive result
                            os.makedirs("protocolgpt_fsm_results", exist_ok=True)
                            filename = f"protocolgpt_fsm_results/{cluster_name}_all_components.json"
                            with open(filename, 'w', encoding='utf-8') as f:
                                json.dump(result, f, indent=2, ensure_ascii=False)
                            print(f"💾 Saved: {filename}")
                        else:
                            print(f"❌ Unknown cluster: {cluster_name}")
                    elif parts[0] == 'chat':
                        cluster_name = parts[1]
                        if cluster_name in extractor.matter_clusters:
                            extractor.chat_with_cluster(cluster_name)
                        else:
                            print(f"❌ Unknown cluster: {cluster_name}")
                    else:
                        cluster_name, query_type = parts
                        if cluster_name in extractor.matter_clusters and query_type in query_types:
                            result = extractor.extract_cluster_fsm(cluster_name, query_type)
                            results.append(result)
                            
                            # Save individual result
                            os.makedirs("protocolgpt_fsm_results", exist_ok=True)
                            filename = f"protocolgpt_fsm_results/{cluster_name}_{query_type}.json"
                            with open(filename, 'w', encoding='utf-8') as f:
                                json.dump(result, f, indent=2, ensure_ascii=False)
                            print(f"💾 Saved: {filename}")
                        else:
                            print(f"❌ Invalid cluster or query type")
                            print("Available clusters:", list(extractor.matter_clusters.keys()))
                            print("Available query types:", query_types)
                else:
                    print("❌ Invalid command format")
                    print("Use: 'cluster_name query_type', 'all cluster_name', 'chat cluster_name', or 'batch'")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Save all results summary
        if results:
            os.makedirs("protocolgpt_fsm_results", exist_ok=True)
            summary_file = "protocolgpt_fsm_results/extraction_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "total_extractions": len(results),
                    "clusters_analyzed": list(set(r.get("cluster_name", "") for r in results)),
                    "results": results
                }, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Saved extraction summary: {summary_file}")
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Please make sure 'Matter_Specification.html' is in the same directory.")
    except Exception as e:
        print(f"❌ Error initializing extractor: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Make sure the model is available: ollama pull llama3.1")
        print("3. Check if all dependencies are installed")
        print("4. Ensure Matter_Specification.html exists in the current directory")
        logging.exception("Initialization error")

if __name__ == "__main__":
    main()
