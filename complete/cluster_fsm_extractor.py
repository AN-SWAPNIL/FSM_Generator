#!/usr/bin/env python3
"""
Matter Cluster-Specific FSM Generator
Based on USENIX Security papers approach for formal logic modeling
Creates separate FSMs for each Matter cluster following formal verification methodology
Following rag_implementation.py structure with LangGraph workflow
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

def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """
    Extract JSON from LLM response that may contain markdown and explanatory text
    """
    # Look for JSON in code blocks first
    json_pattern = r'```json\s*(\{.*?\})\s*```'
    matches = re.findall(json_pattern, response_text, re.DOTALL)
    
    if matches:
        try:
            # Try the first JSON code block
            return json.loads(matches[0])
        except json.JSONDecodeError:
            pass
    
    # Look for JSON without code blocks
    json_pattern = r'(\{[^{}]*"cluster_id"[^{}]*\{.*?\}[^{}]*\})'
    matches = re.findall(json_pattern, response_text, re.DOTALL)
    
    if matches:
        try:
            return json.loads(matches[0])
        except json.JSONDecodeError:
            pass
    
    # Try to find any valid JSON object
    try:
        # Look for opening brace and try to parse from there
        start = response_text.find('{')
        if start != -1:
            # Find the matching closing brace
            brace_count = 0
            end = start
            for i, char in enumerate(response_text[start:], start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            
            if end > start:
                json_str = response_text[start:end]
                return json.loads(json_str)
    except:
        pass
    
    # Return raw response if no JSON found
    return {"raw_response": response_text, "error": "Could not extract valid JSON"}

@dataclass
class MatterClusterInfo:
    """Information about a Matter cluster"""
    cluster_id: str
    cluster_name: str
    description: str
    category: str  # lighting, hvac, access_control, etc.
    attributes: List[str]
    commands: List[str]
    events: List[str]

class MatterClusterFSMExtractor:
    """
    Cluster-specific FSM extractor following formal logic modeling approach
    Based on rag_implementation.py structure with LangGraph workflow
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
            "color_control": MatterClusterInfo(
                cluster_id="0x0300",
                cluster_name="Color Control", 
                description="Controls color properties of lighting devices",
                category="lighting",
                attributes=["CurrentHue", "CurrentSaturation", "CurrentX", "CurrentY", "ColorTemperature"],
                commands=["MoveToHue", "MoveHue", "StepHue", "MoveToSaturation", "MoveSaturation"],
                events=["ColorChanged"]
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
                description="Controls window covering devices",
                category="shades_blinds",
                attributes=["Type", "CurrentPositionLift", "CurrentPositionTilt", "TargetPositionLift"],
                commands=["UpOrOpen", "DownOrClose", "StopMotion", "GoToLiftValue", "GoToTiltValue"],
                events=["PositionChanged", "OperationCompleted"]
            ),
            "occupancy_sensing": MatterClusterInfo(
                cluster_id="0x0406", 
                cluster_name="Occupancy Sensing",
                description="Detects occupancy in an area",
                category="sensors",
                attributes=["Occupancy", "OccupancySensorType", "PIROccupiedToUnoccupiedDelay"],
                commands=[],  # Server-only cluster
                events=["OccupancyChanged"]
            ),
            "temperature_measurement": MatterClusterInfo(
                cluster_id="0x0402",
                cluster_name="Temperature Measurement", 
                description="Measures ambient temperature",
                category="sensors",
                attributes=["MeasuredValue", "MinMeasuredValue", "MaxMeasuredValue", "Tolerance"],
                commands=[],  # Server-only cluster
                events=["TemperatureMeasured"]
            ),
            "illuminance_measurement": MatterClusterInfo(
                cluster_id="0x0400",
                cluster_name="Illuminance Measurement",
                description="Measures ambient light level",
                category="sensors", 
                attributes=["MeasuredValue", "MinMeasuredValue", "MaxMeasuredValue", "Tolerance"],
                commands=[],  # Server-only cluster
                events=["IlluminanceChanged"]
            ),
            "fan_control": MatterClusterInfo(
                cluster_id="0x0202",
                cluster_name="Fan Control",
                description="Controls fan speed and direction",
                category="hvac",
                attributes=["FanMode", "FanModeSequence", "PercentSetting", "PercentCurrent"],
                commands=["Step", "SpeedSetting"],
                events=["FanModeChanged", "PercentSettingChanged"]
            )
        }
    
    def _setup_llm(self):
        """Setup language model (following rag_implementation.py)"""
        print(f"🤖 Initializing {self.model_name}...")
        self.llm = ChatOllama(
            model=self.model_name,
            temperature=0.05,  # Very low for consistent formal logic
            num_predict=4096
        )
        print("✅ Language model initialized")
    
    def _setup_embeddings(self):
        """Setup embeddings model (following rag_implementation.py)"""
        print("🔤 Initializing embeddings model...")
        # Using HuggingFace embeddings like rag_implementation.py
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("✅ Embeddings model initialized")
    
    def _load_and_process_spec(self):
        """Load and process Matter specification document (following rag_implementation.py)"""
        print(f"📄 Loading document: {self.spec_path}")
        
        # Check if document exists
        if not Path(self.spec_path).exists():
            raise FileNotFoundError(f"Document not found: {self.spec_path}")
        
        # Load HTML specification
        loader = UnstructuredHTMLLoader(self.spec_path)
        docs = loader.load()
        
        if not docs:
            raise ValueError("No content loaded from document")
        
        print(f"📖 Loaded {len(docs)} document(s)")
        
        # Split into chunks optimized for cluster analysis
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Larger chunks for formal specification analysis
            chunk_overlap=300,
            length_function=len,
        )
        
        all_splits = text_splitter.split_documents(docs)
        print(f"✂️  Split into {len(all_splits)} chunks")
        
        # Create vector store
        print("🗃️  Creating vector store...")
        self.vector_store = FAISS.from_documents(all_splits, self.embeddings)
        print("✅ Vector store created and indexed")
    
    def _setup_graph(self):
        """Setup the LangGraph FSM extraction workflow (following rag_implementation.py)"""
        print("🔗 Setting up FSM extraction workflow...")
        
        # Create the retrieval tool for cluster-specific queries
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
        
        # Create graph builder
        graph_builder = StateGraph(MessagesState)
        
        # Node 1: Query or respond directly
        def query_or_respond(state: MessagesState):
            """Generate tool call for retrieval or respond directly."""
            llm_with_tools = self.llm.bind_tools([retrieve_cluster_info])
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}
        
        # Node 2: Tool execution
        tools = ToolNode([retrieve_cluster_info])
        
        # Node 3: Generate FSM response
        def generate_fsm(state: MessagesState):
            """Generate FSM analysis using retrieved context."""
            # Get generated ToolMessages
            recent_tool_messages = []
            for message in reversed(state["messages"]):
                if message.type == "tool":
                    recent_tool_messages.append(message)
                else:
                    break
            tool_messages = recent_tool_messages[::-1]
            
            # Format into prompt
            docs_content = "\n\n".join(doc.content for doc in tool_messages)
            system_message_content = (
                "You are a formal logic verification specialist for Matter protocol analysis. "
                "Extract formal finite state machines (FSMs) following rigorous state-transitional logic rules. "
                "Use the retrieved context from the Matter specification to generate precise FSM models "
                "with states, transitions, events, and formal properties. "
                "\n\n"
                "CRITICAL: You must respond with ONLY valid JSON. Do not include explanatory text, "
                "markdown formatting, code blocks, or any other content. Return pure JSON only. "
                "Follow the exact JSON format specified in the user query."
                "\n\n"
                f"Retrieved Context:\n{docs_content}"
            )
            
            # Get conversation messages (excluding tool calls and tool messages)
            conversation_messages = [
                message
                for message in state["messages"]
                if message.type in ("human", "system")
                or (message.type == "ai" and not message.tool_calls)
            ]
            
            prompt = [SystemMessage(system_message_content)] + conversation_messages
            
            # Generate response
            response = self.llm.invoke(prompt)
            return {"messages": [response]}
        
        # Add nodes to graph
        graph_builder.add_node("query_or_respond", query_or_respond)
        graph_builder.add_node("tools", tools)
        graph_builder.add_node("generate_fsm", generate_fsm)
        
        # Set entry point and edges
        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "generate_fsm")
        graph_builder.add_edge("generate_fsm", END)
        
        # Compile with memory
        self.graph = graph_builder.compile(checkpointer=self.memory)
        print("✅ FSM extraction workflow setup complete")

# Formal Logic FSM Query Templates following LL-Verifier approach
def get_cluster_fsm_queries(cluster: MatterClusterInfo) -> Dict[str, str]:
    """
    Generate formal logic FSM queries for a specific cluster
    Following LL-Verifier methodology with State-Transitional Logic Rules
    """
    
    base_prompt = f"""
You are a formal logic verification specialist analyzing the Matter {cluster.cluster_name} cluster (ID: {cluster.cluster_id}).
Your task is to extract a formal finite state machine (FSM) following rigorous state-transitional logic rules.

CLUSTER CONTEXT:
- Cluster: {cluster.cluster_name} ({cluster.cluster_id})
- Category: {cluster.category}
- Description: {cluster.description}
- Attributes: {', '.join(cluster.attributes)}
- Commands: {', '.join(cluster.commands) if cluster.commands else 'None (server-only cluster)'}
- Events: {', '.join(cluster.events)}

FORMAL MODELING REQUIREMENTS:
1. Use State-Transitional Logic Rules (following LL-Verifier methodology)
2. Model deterministic and non-deterministic transitions
3. Include preconditions and postconditions for each transition
4. Define atomic propositions for model checking
5. Consider threat model constraints and security properties
"""

    return {
        "cluster_states": f"""{base_prompt}

TASK: Extract all possible states for the {cluster.cluster_name} cluster.

FORMAL SPECIFICATION REQUIREMENTS:
- Identify ALL possible device states for this cluster
- Include operational states, error states, transitional states
- Consider attribute value combinations that define distinct states
- Model both stable states and transient states
- Include initialization and termination states

For each state, provide:
1. State name (formal identifier)
2. State description (semantic meaning)
3. State type (operational/error/transient/initial/final)
4. Preconditions (what must be true to enter this state)
5. Invariants (what must remain true while in this state)
6. Atomic propositions (for model checking verification)

OUTPUT FORMAT (JSON):
{{
    "cluster_id": "{cluster.cluster_id}",
    "cluster_name": "{cluster.cluster_name}",
    "fsm_type": "cluster_states",
    "states": [
        {{
            "state_id": "formal_state_identifier",
            "state_name": "Human readable name",
            "state_type": "operational|error|transient|initial|final",
            "description": "Detailed state description",
            "preconditions": ["condition1", "condition2"],
            "invariants": ["invariant1", "invariant2"],
            "atomic_propositions": ["prop1", "prop2"],
            "attribute_values": {{"attr1": "value", "attr2": "range"}}
        }}
    ]
}}

Analyze the Matter specification and extract the complete state space for {cluster.cluster_name} cluster.

IMPORTANT: Respond with ONLY the JSON object. Do not include any explanatory text, markdown formatting, or code blocks. Return pure JSON only.
""",

        "cluster_transitions": f"""{base_prompt}

TASK: Extract state transitions with formal logic rules for the {cluster.cluster_name} cluster.

FORMAL SPECIFICATION REQUIREMENTS:
- Model ALL possible state transitions triggered by commands, events, or conditions
- Include both deterministic and non-deterministic transitions
- Define guards (preconditions) and actions (postconditions) for each transition
- Consider error conditions and exception handling
- Model timeout-based transitions and asynchronous events
- Include security-relevant transitions and access control

For each transition, provide:
1. Source state and target state
2. Trigger (command/event/condition/timeout)
3. Guard conditions (when transition is allowed)
4. Actions (what happens during transition)
5. Side effects (attribute changes, events generated)
6. Error conditions and exception handling

OUTPUT FORMAT (JSON):
{{
    "cluster_id": "{cluster.cluster_id}",
    "cluster_name": "{cluster.cluster_name}",
    "fsm_type": "cluster_transitions", 
    "transitions": [
        {{
            "transition_id": "formal_transition_identifier",
            "from_state": "source_state_id",
            "to_state": "target_state_id",
            "trigger": {{
                "type": "command|event|condition|timeout",
                "name": "trigger_name",
                "parameters": ["param1", "param2"]
            }},
            "guard_conditions": ["condition1", "condition2"],
            "actions": ["action1", "action2"],
            "postconditions": ["postcond1", "postcond2"],
            "side_effects": ["effect1", "effect2"],
            "error_handling": ["error_case1", "error_case2"],
            "deterministic": true
        }}
    ]
}}

Analyze the Matter specification and extract ALL state transitions for {cluster.cluster_name} cluster.

IMPORTANT: Respond with ONLY the JSON object. Do not include any explanatory text, markdown formatting, or code blocks. Return pure JSON only.
""",

        "cluster_events": f"""{base_prompt}

TASK: Extract event generation rules for the {cluster.cluster_name} cluster.

FORMAL SPECIFICATION REQUIREMENTS:
- Model ALL events that can be generated by this cluster
- Include triggering conditions for each event
- Define event parameters and data structures
- Model event propagation and handling rules
- Consider timing constraints and event ordering
- Include error events and exception notifications

For each event, provide:
1. Event name and identifier
2. Triggering conditions (when event is generated)
3. Event parameters and data
4. Delivery guarantees and timing
5. Event handling requirements
6. Related state changes

OUTPUT FORMAT (JSON):
{{
    "cluster_id": "{cluster.cluster_id}",
    "cluster_name": "{cluster.cluster_name}",
    "fsm_type": "cluster_events",
    "events": [
        {{
            "event_id": "formal_event_identifier", 
            "event_name": "Human readable name",
            "description": "Event description",
            "triggering_conditions": ["condition1", "condition2"],
            "parameters": [
                {{
                    "name": "param_name",
                    "type": "data_type", 
                    "description": "parameter description"
                }}
            ],
            "delivery_guarantee": "best_effort|reliable|ordered",
            "timing_constraints": ["constraint1", "constraint2"],
            "state_effects": ["effect1", "effect2"],
            "error_conditions": ["error1", "error2"]
        }}
    ]
}}

Analyze the Matter specification and extract event generation rules for {cluster.cluster_name} cluster.

IMPORTANT: Respond with ONLY the JSON object. Do not include any explanatory text, markdown formatting, or code blocks. Return pure JSON only.
""",

        "cluster_properties": f"""{base_prompt}

TASK: Extract security properties and verification constraints for the {cluster.cluster_name} cluster.

FORMAL SPECIFICATION REQUIREMENTS:
- Define security properties using Linear Temporal Logic (LTL)
- Model safety properties (bad things never happen)
- Model liveness properties (good things eventually happen)
- Define access control and authorization properties
- Model data integrity and consistency properties
- Include threat model constraints

For each property, provide:
1. Property name and type
2. LTL formula or formal specification
3. Informal description
4. Verification approach
5. Threat model assumptions
6. Expected behavior

OUTPUT FORMAT (JSON):
{{
    "cluster_id": "{cluster.cluster_id}",
    "cluster_name": "{cluster.cluster_name}", 
    "fsm_type": "cluster_properties",
    "properties": [
        {{
            "property_id": "formal_property_identifier",
            "property_name": "Human readable name",
            "property_type": "safety|liveness|security|consistency",
            "ltl_formula": "LTL formula if applicable",
            "informal_description": "Natural language description", 
            "verification_approach": "model_checking|theorem_proving|testing",
            "threat_assumptions": ["assumption1", "assumption2"],
            "expected_behavior": "Description of correct behavior",
            "violation_consequences": ["consequence1", "consequence2"]
        }}
    ]
}}

Analyze the Matter specification and extract security properties for {cluster.cluster_name} cluster.

IMPORTANT: Respond with ONLY the JSON object. Do not include any explanatory text, markdown formatting, or code blocks. Return pure JSON only.
""",

        "complete_cluster_fsm": f"""{base_prompt}

TASK: Generate complete formal FSM model for the {cluster.cluster_name} cluster.

FORMAL SPECIFICATION REQUIREMENTS:
- Integrate states, transitions, events, and properties into unified model
- Ensure FSM completeness and consistency
- Validate state reachability and transition coverage
- Model both normal operation and error handling
- Include formal verification properties
- Generate model in format suitable for model checking tools

OUTPUT FORMAT (JSON):
{{
    "cluster_id": "{cluster.cluster_id}",
    "cluster_name": "{cluster.cluster_name}",
    "fsm_type": "complete_cluster_fsm",
    "metadata": {{
        "specification_version": "Matter 1.0",
        "modeling_approach": "state_transitional_logic",
        "verification_method": "model_checking",
        "threat_model": "standard_matter_security"
    }},
    "initial_state": "initial_state_id",
    "final_states": ["final_state1", "final_state2"],
    "states": [...],
    "transitions": [...], 
    "events": [...],
    "properties": [...],
    "verification_results": {{
        "state_count": "number",
        "transition_count": "number", 
        "reachability": "all_states_reachable",
        "deadlock_freedom": "verified",
        "property_satisfaction": ["satisfied_prop1", "satisfied_prop2"]
    }}
}}

Generate the complete formal FSM model for {cluster.cluster_name} cluster suitable for automated verification.

IMPORTANT: Respond with ONLY the JSON object. Do not include any explanatory text, markdown formatting, or code blocks. Return pure JSON only.
"""
    }

def extract_cluster_fsm(extractor: MatterClusterFSMExtractor, cluster_name: str, query_type: str, thread_id: str = "default") -> Dict[str, Any]:
    """Extract FSM for specific cluster and query type using LangGraph workflow"""
    if cluster_name not in extractor.matter_clusters:
        raise ValueError(f"Unknown cluster: {cluster_name}")
    
    cluster = extractor.matter_clusters[cluster_name]
    queries = get_cluster_fsm_queries(cluster)
    
    if query_type not in queries:
        raise ValueError(f"Unknown query type: {query_type}")
    
    query = queries[query_type]
    
    print(f"\n🔍 Extracting {query_type} for {cluster.cluster_name} cluster")
    print("=" * 60)
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # Use the LangGraph workflow to get response
    print("🤔 Processing with FSM extraction workflow...")
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
        raise ValueError("No response generated from FSM extraction workflow")
    
    # Extract JSON from the response
    print(f"\n🔧 Parsing FSM JSON...")
    extracted_json = extract_json_from_response(final_response)
    
    # Check if we got valid JSON
    if "error" in extracted_json:
        print("⚠️  Warning: Could not extract clean JSON, saving raw response")
        fsm_data = extracted_json
    else:
        print("✅ Successfully extracted JSON FSM data")
        fsm_data = extracted_json
    
    print(f"\n🤖 FSM Analysis Complete")
    print("=" * 40)
    
    # Show structured preview
    if "cluster_id" in fsm_data:
        print(f"Cluster: {fsm_data.get('cluster_name', 'N/A')} ({fsm_data.get('cluster_id', 'N/A')})")
        if "states" in fsm_data:
            print(f"States extracted: {len(fsm_data['states'])}")
        if "transitions" in fsm_data:
            print(f"Transitions extracted: {len(fsm_data['transitions'])}")
        if "events" in fsm_data:
            print(f"Events extracted: {len(fsm_data['events'])}")
    else:
        print("Raw response preview:")
        print(str(fsm_data)[:300] + "..." if len(str(fsm_data)) > 300 else str(fsm_data))
    
    return {
        "cluster_name": cluster_name,
        "cluster_id": cluster.cluster_id,
        "query_type": query_type,
        "fsm_data": fsm_data,  # Structured FSM data
        "raw_response": final_response,  # Keep original for debugging
        "thread_id": thread_id
    }

def main():
    """Main function for cluster-specific FSM extraction (following rag_implementation.py pattern)"""
    print("🚀 Initializing Matter Cluster FSM Extractor")
    print("="*50)
    
    # Path to Matter specification
    spec_path = os.path.join(os.path.dirname(__file__), "Matter_Specification.html")
    
    if not os.path.exists(spec_path):
        print(f"❌ Matter specification not found at: {spec_path}")
        print("Please copy Matter_Specification.html to this directory")
        return
    
    try:
        # Initialize the FSM extractor (following rag_implementation.py pattern)
        extractor = MatterClusterFSMExtractor(
            spec_path=spec_path,
            model_name="llama3.1"
        )
        
        print("\n✅ Initialization complete!")
        print(f"\n📋 Available clusters ({len(extractor.matter_clusters)}):")
        for i, (cluster_key, cluster) in enumerate(extractor.matter_clusters.items(), 1):
            print(f"  {i}. {cluster.cluster_name} ({cluster.cluster_id}) - {cluster.category}")
        
        query_types = ["cluster_states", "cluster_transitions", "cluster_events", "cluster_properties", "complete_cluster_fsm"]
        print(f"\n🔧 Available FSM query types:")
        for i, query_type in enumerate(query_types, 1):
            print(f"  {i}. {query_type}")
        
        print("\n" + "="*60)
        print("� Matter Cluster FSM Extraction Assistant")
        print("="*60)
        print("Extract formal FSMs for Matter clusters following USENIX paper methodology!")
        print("Commands:")
        print("  - 'cluster_name query_type' (e.g., 'on_off cluster_states')")
        print("  - 'all cluster_name' to extract all FSM types for a cluster")
        print("  - 'complete' to extract complete FSMs for all clusters")
        print("  - 'interactive' for step-by-step extraction")
        print("  - 'quit', 'exit' to exit")
        print("="*60)
        
        results = []
        
        while True:
            try:
                user_input = input("\n🤖 Enter command: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if user_input.lower() == 'interactive':
                    # Interactive mode following rag_implementation.py pattern
                    print("\n🔄 Entering interactive FSM extraction mode...")
                    print("Ask questions about Matter cluster FSMs!")
                    
                    thread_id = "interactive_fsm"
                    config = {"configurable": {"thread_id": thread_id}}
                    
                    while True:
                        try:
                            question = input("\n🙋 FSM Question: ").strip()
                            
                            if question.lower() in ['quit', 'exit', 'back']:
                                break
                                
                            if not question:
                                continue
                            
                            print("\n🤔 Analyzing...")
                            
                            # Stream the response using LangGraph
                            messages_seen = 0
                            for step in extractor.graph.stream(
                                {"messages": [{"role": "user", "content": question}]},
                                stream_mode="values",
                                config=config,
                            ):
                                current_messages = len(step["messages"])
                                if current_messages > messages_seen:
                                    latest_message = step["messages"][-1]
                                    if hasattr(latest_message, 'content') and latest_message.content:
                                        if latest_message.type == "ai" and not getattr(latest_message, 'tool_calls', None):
                                            print(f"\n🤖 FSM Analysis: {latest_message.content}")
                                    messages_seen = current_messages
                                    
                        except KeyboardInterrupt:
                            print("\n\nExiting interactive mode...")
                            break
                        except Exception as e:
                            print(f"❌ Error in interactive mode: {e}")
                    continue
                
                if user_input.lower() == 'complete':
                    print("\n🚀 Extracting complete FSMs for all clusters...")
                    for cluster_name in extractor.matter_clusters.keys():
                        try:
                            thread_id = f"complete_{cluster_name}"
                            result = extract_cluster_fsm(extractor, cluster_name, "complete_cluster_fsm", thread_id)
                            results.append(result)
                            
                            # Save individual result
                            os.makedirs("cluster_fsm_results", exist_ok=True)
                            filename = f"cluster_fsm_results/{cluster_name}_complete_fsm.json"
                            with open(filename, 'w', encoding='utf-8') as f:
                                json.dump(result, f, indent=2, ensure_ascii=False)
                            print(f"💾 Saved: {filename}")
                            
                        except Exception as e:
                            print(f"❌ Error extracting {cluster_name}: {e}")
                    continue
                
                parts = user_input.split()
                if len(parts) == 2:
                    if parts[0] == 'all':
                        cluster_name = parts[1]
                        if cluster_name in extractor.matter_clusters:
                            print(f"\n🚀 Extracting all FSM types for {cluster_name} cluster...")
                            for query_type in query_types:
                                try:
                                    thread_id = f"all_{cluster_name}_{query_type}"
                                    result = extract_cluster_fsm(extractor, cluster_name, query_type, thread_id)
                                    results.append(result)
                                    
                                    # Save individual result
                                    os.makedirs("cluster_fsm_results", exist_ok=True)
                                    filename = f"cluster_fsm_results/{cluster_name}_{query_type}.json"
                                    with open(filename, 'w', encoding='utf-8') as f:
                                        json.dump(result, f, indent=2, ensure_ascii=False)
                                    print(f"💾 Saved: {filename}")
                                except Exception as e:
                                    print(f"❌ Error in {query_type}: {e}")
                        else:
                            print(f"❌ Unknown cluster: {cluster_name}")
                    else:
                        cluster_name, query_type = parts
                        if cluster_name in extractor.matter_clusters and query_type in query_types:
                            thread_id = f"{cluster_name}_{query_type}"
                            result = extract_cluster_fsm(extractor, cluster_name, query_type, thread_id)
                            results.append(result)
                            
                            # Save result
                            os.makedirs("cluster_fsm_results", exist_ok=True)
                            filename = f"cluster_fsm_results/{cluster_name}_{query_type}.json"
                            with open(filename, 'w', encoding='utf-8') as f:
                                json.dump(result, f, indent=2, ensure_ascii=False)
                            print(f"💾 Saved: {filename}")
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
            os.makedirs("cluster_fsm_results", exist_ok=True)
            all_results_file = "cluster_fsm_results/all_cluster_fsm_results.json"
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
