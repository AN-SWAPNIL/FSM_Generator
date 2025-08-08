#!/usr/bin/env python3
"""
Matter Cluster-Specific FSM Generator
Following Les Modeling Framework from USENIX Security Papers
Creates formal logic models using State-Transitional Logic Rules and Event Generation Rules
Based on rag_implementation.py structure with LangGraph workflow
"""

import os
import json
import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Core LangChain/LangGraph imports (following rag_implementation.py)
from langchain_ollama.chat_models import ChatOllama
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

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

class MatterClusterLesFSMExtractor:
    """
    Cluster-specific FSM extractor following Les modeling framework from USENIX Security papers
    Generates formal logic models with State-Transitional Logic Rules and Event Generation Rules
    """
    
    def __init__(self, spec_path: str, model_name: str = "llama3.1"):
        self.spec_path = spec_path
        self.model_name = model_name
        self.vector_store = None
        self.graph = None
        self.memory = MemorySaver()
        
        # Matter clusters definition
        self.matter_clusters = self._define_matter_clusters()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components (following rag_implementation.py pattern)
        self._setup_llm()
        self._setup_embeddings()
        self._load_and_process_spec()
        self._setup_graph()
    
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
            )
        }
    
    def _setup_llm(self):
        """Setup language model (following rag_implementation.py)"""
        print(f"🤖 Initializing {self.model_name}...")
        self.llm = ChatOllama(
            model=self.model_name,
            temperature=0,  # Low for consistent formal logic
            num_predict=4096
        )
        print("✅ Language model initialized")
    
    def _setup_embeddings(self):
        """Setup embeddings model (following rag_implementation.py)"""
        print("🔤 Initializing embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("✅ Embeddings model initialized")
    
    def _load_and_process_spec(self):
        """Load and process Matter specification document (following rag_implementation.py)"""
        print(f"📄 Loading document: {self.spec_path}")
        
        if not Path(self.spec_path).exists():
            raise FileNotFoundError(f"Document not found: {self.spec_path}")
        
        loader = UnstructuredHTMLLoader(self.spec_path)
        docs = loader.load()
        
        if not docs:
            raise ValueError("No content loaded from document")
        
        print(f"📖 Loaded {len(docs)} document(s)")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            length_function=len,
        )
        
        all_splits = text_splitter.split_documents(docs)
        print(f"✂️  Split into {len(all_splits)} chunks")
        
        print("🗃️  Creating vector store...")
        self.vector_store = FAISS.from_documents(all_splits, self.embeddings)
        print("✅ Vector store created and indexed")
    
    def _setup_graph(self):
        """Setup the LangGraph Les FSM extraction workflow"""
        print("🔗 Setting up Les FSM extraction workflow...")
        
        @tool(response_format="content_and_artifact")
        def retrieve_cluster_info(query: str):
            """Retrieve cluster-specific information from the Matter specification."""
            retrieved_docs = self.vector_store.similarity_search(query, k=6)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata.get('source', 'Matter Specification')}\n"
                 f"Content: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs
        
        self.retrieve_tool = retrieve_cluster_info
        
        graph_builder = StateGraph(MessagesState)
        
        def query_or_respond(state: MessagesState):
            """Generate tool call for retrieval or respond directly."""
            llm_with_tools = self.llm.bind_tools([retrieve_cluster_info])
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}
        
        tools = ToolNode([retrieve_cluster_info])
        
        def generate_les_model(state: MessagesState):
            """Generate Les formal model using retrieved context."""
            recent_tool_messages = []
            for message in reversed(state["messages"]):
                if message.type == "tool":
                    recent_tool_messages.append(message)
                else:
                    break
            tool_messages = recent_tool_messages[::-1]
            
            docs_content = "\n\n".join(doc.content for doc in tool_messages)
            system_message_content = (
                "You are a formal logic verification specialist implementing the Les modeling framework "
                "from USENIX Security papers. Generate formal logic models using State-Transitional Logic Rules "
                "and Event Generation Rules. Follow the exact syntax and semantics from the Les framework.\n\n"
                "CRITICAL: Output ONLY the formal Les model code without any explanations, "
                "markdown formatting, or additional text. Start directly with the Les syntax.\n\n"
                f"Retrieved Context:\n{docs_content}"
            )
            
            conversation_messages = [
                message
                for message in state["messages"]
                if message.type in ("human", "system")
                or (message.type == "ai" and not message.tool_calls)
            ]
            
            prompt = [SystemMessage(system_message_content)] + conversation_messages
            response = self.llm.invoke(prompt)
            return {"messages": [response]}
        
        graph_builder.add_node("query_or_respond", query_or_respond)
        graph_builder.add_node("tools", tools)
        graph_builder.add_node("generate_les_model", generate_les_model)
        
        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "generate_les_model")
        graph_builder.add_edge("generate_les_model", END)
        
        self.graph = graph_builder.compile(checkpointer=self.memory)
        print("✅ Les FSM extraction workflow setup complete")

def get_les_cluster_queries(cluster: MatterClusterInfo) -> Dict[str, str]:
    """
    Generate Les modeling framework queries for a specific cluster
    Following USENIX Security papers approach with State-Transitional Logic Rules
    """
    
    base_context = f"""
CLUSTER ANALYSIS: {cluster.cluster_name} ({cluster.cluster_id})
Category: {cluster.category}
Attributes: {', '.join(cluster.attributes)}
Commands: {', '.join(cluster.commands) if cluster.commands else 'None (server-only)'}
Events: {', '.join(cluster.events)}

FORMAL MODELING TASK: Generate Les framework formal logic model following USENIX Security papers.
"""

    return {
        "state_transition_rules": f"""{base_context}

Extract State-Transitional Logic Rules for {cluster.cluster_name} cluster.

Output ONLY Les syntax rules (no explanations):

*** Variable declarations
vars DeviceX : Device .
vars UserY : User .
vars StateA StateB : State .

*** State-transition rules for {cluster.cluster_name}
eq < DeviceX |'attribute': StateA , ... >
   $ UserY'command'DeviceX
-> < DeviceX |'attribute': StateB , ... > .

[Continue with ALL state-transition rules for this cluster]

Extract from Matter specification and model ALL state changes for {cluster.cluster_name} cluster.
""",

        "event_generation_rules": f"""{base_context}

Extract Event Generation Rules for {cluster.cluster_name} cluster.

Output ONLY Les syntax rules (no explanations):

*** Event templates for {cluster.cluster_name}
ev1(UserX, DeviceY) = $ UserX'command'DeviceY .
ev2(DeviceX, UserY) = $ DeviceX'event'UserY .

*** Event generation rules
rl < DeviceX |'attribute': StateA , ... >
   < UserY |'proximity': DeviceX , ... >
--> ev1(UserY, DeviceX) .

[Continue with ALL event generation rules for this cluster]

Extract from Matter specification and model ALL event generation for {cluster.cluster_name} cluster.
""",

        "security_violations": f"""{base_context}

Extract Security Violations for {cluster.cluster_name} cluster following sv1-sv4 pattern.

Output ONLY Les syntax (no explanations):

*** Security violations for {cluster.cluster_name}

*** sv1: Unauthorized access violation
sv1 = ♢((¬userAuthorized ∧ ⃝¬userAuthorized) ∧ stateChanged)

*** sv2: Privilege escalation violation  
sv2 = ♢((userRemote ∧ ¬userBound) ∧ ⃝(userRemote ∧ userBound))

*** Atomic propositions
E@S<DeviceX|'attribute':StateA,...>⊢stateChanged=true

[Continue with ALL security violations for this cluster]

Extract security properties and violations for {cluster.cluster_name} cluster.
""",

        "complete_les_model": f"""{base_context}

Generate COMPLETE Les formal model for {cluster.cluster_name} cluster.

Output ONLY complete Les syntax (no explanations):

*** Complete Les model for {cluster.cluster_name} cluster ({cluster.cluster_id})

*** Variable declarations  
vars DeviceX : Device .
vars UserY : User .
vars StateA StateB : State .

*** Initial states
< DeviceX |'attribute': 'initial' , ... >
< UserY |'proximity': 'remote' , ... >

*** State-transition rules
[ALL state-transition rules]

*** Event generation rules  
[ALL event generation rules]

*** Security violations
[ALL security violations]

*** Atomic propositions
[ALL atomic propositions]

Generate complete formal model for {cluster.cluster_name} cluster following Les framework.
"""
    }

def extract_les_cluster_model(extractor: MatterClusterLesFSMExtractor, cluster_name: str, 
                             query_type: str, thread_id: str = "default") -> Dict[str, Any]:
    """Extract Les formal model for specific cluster using LangGraph workflow"""
    if cluster_name not in extractor.matter_clusters:
        raise ValueError(f"Unknown cluster: {cluster_name}")
    
    cluster = extractor.matter_clusters[cluster_name]
    queries = get_les_cluster_queries(cluster)
    
    if query_type not in queries:
        raise ValueError(f"Unknown query type: {query_type}")
    
    query = queries[query_type]
    
    print(f"\n🔍 Extracting {query_type} for {cluster.cluster_name} cluster")
    print("=" * 60)
    
    config = {"configurable": {"thread_id": thread_id}}
    
    print("🤔 Processing with Les FSM extraction workflow...")
    final_response = None
    
    for step in extractor.graph.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
        config=config,
    ):
        final_message = step["messages"][-1]
        if hasattr(final_message, 'content') and final_message.content:
            if final_message.type == "ai" and not getattr(final_message, 'tool_calls', None):
                final_response = final_message.content
    
    if not final_response:
        raise ValueError("No response generated from Les FSM extraction workflow")
    
    # Clean up the response to extract only Les model code
    cleaned_response = clean_les_model_output(final_response)
    
    print(f"\n🤖 Les Model Generated")
    print("=" * 40)
    print(cleaned_response[:800] + "..." if len(cleaned_response) > 800 else cleaned_response)
    
    return {
        "cluster_name": cluster_name,
        "cluster_id": cluster.cluster_id,
        "query_type": query_type,
        "les_model": cleaned_response,
        "raw_response": final_response,
        "thread_id": thread_id,
        "modeling_framework": "Les"
    }

def clean_les_model_output(response: str) -> str:
    """Clean up LLM response to extract only Les model code"""
    # Remove markdown code blocks
    response = re.sub(r'```[a-zA-Z]*\n?', '', response)
    response = re.sub(r'```', '', response)
    
    # Remove common explanatory text
    lines = response.split('\n')
    cleaned_lines = []
    
    skip_patterns = [
        'based on', 'following', 'analysis', 'here is', 'this model',
        'explanation', 'note:', 'important:', 'output:', 'format:'
    ]
    
    for line in lines:
        line = line.strip()
        if line and not any(pattern in line.lower() for pattern in skip_patterns):
            if (line.startswith('***') or 
                line.startswith('vars ') or 
                line.startswith('eq ') or 
                line.startswith('rl ') or 
                line.startswith('ceq ') or
                line.startswith('crl ') or
                line.startswith('sv') or
                line.startswith('ev') or
                line.startswith('E@') or
                line.startswith('<') or
                line.startswith('->') or
                line.startswith('-->')):
                cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def main():
    """Main function for Les cluster FSM extraction (following rag_implementation.py pattern)"""
    print("🚀 Initializing Matter Cluster Les FSM Extractor")
    print("="*50)
    
    spec_path = os.path.join(os.path.dirname(__file__), "Matter_Specification.html")
    
    if not os.path.exists(spec_path):
        print(f"❌ Matter specification not found at: {spec_path}")
        print("Please copy Matter_Specification.html to this directory")
        return
    
    try:
        extractor = MatterClusterLesFSMExtractor(
            spec_path=spec_path,
            model_name="llama3.1"
        )
        
        print("\n✅ Initialization complete!")
        print(f"\n📋 Available clusters ({len(extractor.matter_clusters)}):")
        for i, (cluster_key, cluster) in enumerate(extractor.matter_clusters.items(), 1):
            print(f"  {i}. {cluster.cluster_name} ({cluster.cluster_id}) - {cluster.category}")
        
        query_types = ["state_transition_rules", "event_generation_rules", "security_violations", "complete_les_model"]
        print(f"\n🔧 Available Les model types:")
        for i, query_type in enumerate(query_types, 1):
            print(f"  {i}. {query_type}")
        
        print("\n" + "="*60)
        print("🚀 Matter Cluster Les FSM Extraction Assistant")
        print("="*60)
        print("Extract formal Les models following USENIX Security papers methodology!")
        print("Commands:")
        print("  - 'cluster_name query_type' (e.g., 'on_off state_transition_rules')")
        print("  - 'all cluster_name' to extract all Les models for a cluster")
        print("  - 'complete' to extract complete Les models for all clusters")
        print("  - 'quit', 'exit' to exit")
        print("="*60)
        
        results = []
        
        while True:
            try:
                user_input = input("\n🤖 Enter command: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if user_input.lower() == 'complete':
                    print("\n🚀 Extracting complete Les models for all clusters...")
                    for cluster_name in extractor.matter_clusters.keys():
                        try:
                            thread_id = f"complete_{cluster_name}"
                            result = extract_les_cluster_model(extractor, cluster_name, "complete_les_model", thread_id)
                            results.append(result)
                            
                            os.makedirs("les_fsm_results", exist_ok=True)
                            filename = f"les_fsm_results/{cluster_name}_complete_les_model.json"
                            with open(filename, 'w', encoding='utf-8') as f:
                                json.dump(result, f, indent=2, ensure_ascii=False)
                            print(f"💾 Saved: {filename}")
                            
                            # Also save as .les file
                            les_filename = f"les_fsm_results/{cluster_name}_complete_model.les"
                            with open(les_filename, 'w', encoding='utf-8') as f:
                                f.write(result["les_model"])
                            print(f"💾 Saved Les model: {les_filename}")
                            
                        except Exception as e:
                            print(f"❌ Error extracting {cluster_name}: {e}")
                    continue
                
                parts = user_input.split()
                if len(parts) == 2:
                    if parts[0] == 'all':
                        cluster_name = parts[1]
                        if cluster_name in extractor.matter_clusters:
                            print(f"\n🚀 Extracting all Les models for {cluster_name} cluster...")
                            for query_type in query_types:
                                try:
                                    thread_id = f"all_{cluster_name}_{query_type}"
                                    result = extract_les_cluster_model(extractor, cluster_name, query_type, thread_id)
                                    results.append(result)
                                    
                                    os.makedirs("les_fsm_results", exist_ok=True)
                                    filename = f"les_fsm_results/{cluster_name}_{query_type}.json"
                                    with open(filename, 'w', encoding='utf-8') as f:
                                        json.dump(result, f, indent=2, ensure_ascii=False)
                                    print(f"💾 Saved: {filename}")
                                    
                                    # Also save as .les file
                                    les_filename = f"les_fsm_results/{cluster_name}_{query_type}.les"
                                    with open(les_filename, 'w', encoding='utf-8') as f:
                                        f.write(result["les_model"])
                                    print(f"💾 Saved Les model: {les_filename}")
                                    
                                except Exception as e:
                                    print(f"❌ Error in {query_type}: {e}")
                        else:
                            print(f"❌ Unknown cluster: {cluster_name}")
                    else:
                        cluster_name, query_type = parts
                        if cluster_name in extractor.matter_clusters and query_type in query_types:
                            thread_id = f"{cluster_name}_{query_type}"
                            result = extract_les_cluster_model(extractor, cluster_name, query_type, thread_id)
                            results.append(result)
                            
                            os.makedirs("les_fsm_results", exist_ok=True)
                            filename = f"les_fsm_results/{cluster_name}_{query_type}.json"
                            with open(filename, 'w', encoding='utf-8') as f:
                                json.dump(result, f, indent=2, ensure_ascii=False)
                            print(f"💾 Saved: {filename}")
                            
                            # Also save as .les file
                            les_filename = f"les_fsm_results/{cluster_name}_{query_type}.les"
                            with open(les_filename, 'w', encoding='utf-8') as f:
                                f.write(result["les_model"])
                            print(f"💾 Saved Les model: {les_filename}")
                        else:
                            print(f"❌ Invalid cluster or query type")
                            print("Available clusters:", list(extractor.matter_clusters.keys()))
                            print("Available query types:", query_types)
                else:
                    print("❌ Invalid command format")
                    print("Use: 'cluster_name query_type' or 'all cluster_name' or 'complete'")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Save all results
        if results:
            os.makedirs("les_fsm_results", exist_ok=True)
            all_results_file = "les_fsm_results/all_les_fsm_results.json"
            with open(all_results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Saved all {len(results)} results to: {all_results_file}")
        
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
