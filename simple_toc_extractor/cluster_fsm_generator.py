#!/usr/bin/env python3
"""
Matter Cluster FSM Generator
Generates Finite State Machine models for each Matter cluster from detailed JSON
Following Les Modeling Framework from USENIX Security Papers
Based on cluster_detail_extractor.py structure with Gemini AI
"""

import json
import logging
import os
import sys
import signal
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

# LangChain and Gemini imports (following existing pattern)
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Configuration
from config import (
    GOOGLE_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE,
    CLUSTER_CATEGORIES
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FSMState:
    """Represents a state in the FSM"""
    name: str
    description: str
    is_initial: bool = False
    is_final: bool = False
    attributes_monitored: List[str] = None
    
    def __post_init__(self):
        if self.attributes_monitored is None:
            self.attributes_monitored = []

@dataclass
class FSMTransition:
    """Represents a transition in the FSM"""
    from_state: str
    to_state: str
    trigger: str
    guard_condition: str = ""
    actions: List[str] = None
    
    def __post_init__(self):
        if self.actions is None:
            self.actions = []

@dataclass
class FSMModel:
    """Complete FSM model for a cluster"""
    cluster_name: str
    cluster_id: str
    category: str
    states: List[FSMState]
    transitions: List[FSMTransition]
    initial_state: str
    attributes_used: List[str]
    commands_handled: List[str]
    events_generated: List[str]
    invariants: List[str]
    
    def __post_init__(self):
        if self.attributes_used is None:
            self.attributes_used = []
        if self.commands_handled is None:
            self.commands_handled = []
        if self.events_generated is None:
            self.events_generated = []
        if self.invariants is None:
            self.invariants = []

class ClusterFSMGenerator:
    """Generates FSM models for Matter clusters using Gemini AI"""
    
    def __init__(self, detailed_json_path: str, output_dir: str = "fsm_results"):
        """
        Initialize the FSM generator
        
        Args:
            detailed_json_path: Path to the matter_clusters_detailed.json file
            output_dir: Directory to save FSM results
        """
        self.detailed_json_path = detailed_json_path
        self.output_dir = output_dir
        self.clusters_data = None
        self.llm = None
        self.embeddings = None
        self.interruption_handler_set = False
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self._load_clusters_data()
        self._init_llm()
        self._setup_interruption_handler()
    
    def _setup_interruption_handler(self):
        """Setup signal handlers for graceful interruption"""
        if not self.interruption_handler_set:
            def signal_handler(signum, frame):
                logger.warning(f"Received signal {signum}. Exiting gracefully...")
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
            signal.signal(signal.SIGTERM, signal_handler)  # Termination
            self.interruption_handler_set = True
            logger.info("Interruption handlers set up")
    
    def _init_llm(self):
        """Initialize the language model and embeddings"""
        try:
            self.llm = GoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=GOOGLE_API_KEY,
                temperature=GEMINI_TEMPERATURE,
                max_output_tokens=8192  # Increase token limit for complete responses
            )
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY
            )
            logger.info("Initialized Gemini LLM and embeddings")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise
    
    def _load_clusters_data(self):
        """Load clusters data from detailed JSON file"""
        try:
            with open(self.detailed_json_path, 'r', encoding='utf-8') as f:
                self.clusters_data = json.load(f)
            cluster_count = len(self.clusters_data.get('clusters', []))
            logger.info(f"Loaded {cluster_count} detailed clusters")
        except Exception as e:
            logger.error(f"Error loading clusters data: {e}")
            raise
    
    def get_fsm_generation_prompt(self, cluster_info: Dict[str, Any]) -> str:
        """
        Generate FSM extraction prompt for a specific cluster
        
        Args:
            cluster_info: Detailed cluster information
            
        Returns:
            Formatted prompt string
        """
        cluster_data = cluster_info.get('cluster_info', {})
        cluster_name = cluster_data.get('cluster_name', 'Unknown')
        cluster_id = cluster_data.get('cluster_id', 'Unknown')
        category = cluster_info.get('metadata', {}).get('category', 'Unknown')
        
        # Extract key information
        attributes = cluster_data.get('attributes', [])
        commands = cluster_data.get('commands', [])
        events = cluster_data.get('events', [])
        features = cluster_data.get('features', [])
        
        # Format attributes info
        attr_info = "\n".join([
            f"- {attr.get('name', 'Unknown')} (ID: {attr.get('id', 'N/A')}, Type: {attr.get('type', 'N/A')}, Access: {attr.get('access', 'N/A')})"
            for attr in attributes
        ]) if attributes else "None"
        
        # Format commands info
        cmd_info = "\n".join([
            f"- {cmd.get('name', 'Unknown')} (ID: {cmd.get('id', 'N/A')}, Direction: {cmd.get('direction', 'N/A')})"
            for cmd in commands
        ]) if commands else "None"
        
        # Format events info
        event_info = "\n".join([
            f"- {event.get('name', 'Unknown')} (ID: {event.get('id', 'N/A')}, Priority: {event.get('priority', 'N/A')})"
            for event in events
        ]) if events else "None"
        
        return f"""
You are a formal methods expert creating a Finite State Machine (FSM) for a Matter IoT cluster.

CLUSTER: {cluster_name} (ID: {cluster_id}, Category: {category})

ATTRIBUTES: {attr_info}

COMMANDS: {cmd_info}

EVENTS: {event_info}

TASK: Create a focused FSM model with essential states and transitions.

Focus on these key states:
1. Uninitialized - Before cluster setup
2. Ready - Normal operation state  
3. Processing - When handling commands
4. Error - Error/fault state

Include realistic transitions based on commands and attribute changes.

REQUIRED JSON OUTPUT (must be complete and valid):
{{
  "fsm_model": {{
    "cluster_name": "{cluster_name}",
    "cluster_id": "{cluster_id}",
    "category": "{category}",
    "states": [
      {{
        "name": "state_name",
        "description": "brief description",
        "is_initial": true/false,
        "is_final": false,
        "attributes_monitored": ["relevant_attributes"],
        "invariants": ["key_conditions"]
      }}
    ],
    "transitions": [
      {{
        "from_state": "source_state",
        "to_state": "target_state", 
        "trigger": "command_or_event_name",
        "guard_condition": "when this transition is valid",
        "actions": ["what_happens"],
        "response_command": "response_if_any"
      }}
    ],
    "initial_state": "initial_state_name",
    "final_states": [],
    "attributes_used": {[attr.get('name', '') for attr in attributes[:5]]},
    "commands_handled": {[cmd.get('name', '') for cmd in commands[:5]]},
    "events_generated": {[event.get('name', '') for event in events[:3]]},
    "formal_properties": {{
      "safety_properties": ["no_invalid_transitions", "state_consistency"],
      "liveness_properties": ["eventually_ready", "command_completion"],
      "invariants": ["state_valid", "attribute_consistency"]
    }}
  }}
}}

Generate ONLY complete valid JSON. Keep descriptions concise. Ensure all JSON brackets are properly closed.
"""
    
    def generate_cluster_fsm(self, cluster_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate FSM model for a single cluster
        
        Args:
            cluster_info: Detailed cluster information
            
        Returns:
            Generated FSM model as dictionary
        """
        cluster_data = cluster_info.get('cluster_info', {})
        cluster_name = cluster_data.get('cluster_name', 'Unknown')
        
        logger.info(f"Generating FSM for cluster: {cluster_name}")
        
        try:
            # Generate prompt
            prompt = self.get_fsm_generation_prompt(cluster_info)
            
            # Get LLM response
            response = self.llm.invoke(prompt)
            
            # Parse JSON response
            try:
                # Clean response by removing code block markers if present
                clean_response = response.strip()
                
                # Handle multiple code block formats
                if clean_response.startswith('```json'):
                    clean_response = clean_response[7:]  # Remove ```json
                elif clean_response.startswith('```'):
                    clean_response = clean_response[3:]  # Remove ```
                
                if clean_response.endswith('```'):
                    clean_response = clean_response[:-3]  # Remove ```
                
                clean_response = clean_response.strip()
                
                # Log response length for debugging
                logger.debug(f"Response length for {cluster_name}: {len(clean_response)} characters")
                
                # Try to find complete JSON if response was truncated
                if not clean_response.endswith('}'):
                    # Look for the last complete JSON object
                    brace_count = 0
                    last_complete_pos = -1
                    
                    for i, char in enumerate(clean_response):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                last_complete_pos = i
                    
                    if last_complete_pos > 0:
                        clean_response = clean_response[:last_complete_pos + 1]
                        logger.warning(f"Response was truncated, using partial JSON for {cluster_name}")
                    else:
                        logger.error(f"Could not find complete JSON for {cluster_name}")
                        return self._create_fallback_fsm(cluster_info)
                
                # Additional validation - check if JSON starts correctly
                if not clean_response.startswith('{'):
                    logger.error(f"Response does not start with '{{' for {cluster_name}")
                    logger.error(f"First 200 chars: {clean_response[:200]}")
                    return self._create_fallback_fsm(cluster_info)
                
                fsm_model = json.loads(clean_response)
                
                # Add generation metadata
                if 'fsm_model' in fsm_model:
                    fsm_model['fsm_model']['generation_timestamp'] = datetime.now().isoformat()
                    fsm_model['fsm_model']['source_metadata'] = {
                        'extraction_method': cluster_info.get('metadata', {}).get('extraction_method', 'Unknown'),
                        'source_pages': cluster_info.get('metadata', {}).get('source_pages', 'Unknown'),
                        'section_number': cluster_info.get('metadata', {}).get('section_number', 'Unknown')
                    }
                
                logger.info(f"Successfully generated FSM for {cluster_name}")
                return fsm_model
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error for {cluster_name}: {e}")
                logger.error(f"LLM response: {response[:500]}...")
                return self._create_fallback_fsm(cluster_info)
                
        except Exception as e:
            logger.error(f"Error generating FSM for {cluster_name}: {e}")
            return self._create_fallback_fsm(cluster_info)
    
    def _create_fallback_fsm(self, cluster_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a basic fallback FSM when generation fails
        
        Args:
            cluster_info: Cluster information
            
        Returns:
            Basic FSM structure
        """
        cluster_data = cluster_info.get('cluster_info', {})
        cluster_name = cluster_data.get('cluster_name', 'Unknown')
        cluster_id = cluster_data.get('cluster_id', 'Unknown')
        category = cluster_info.get('metadata', {}).get('category', 'Unknown')
        
        return {
            "fsm_model": {
                "cluster_name": cluster_name,
                "cluster_id": cluster_id,
                "category": category,
                "generation_timestamp": datetime.now().isoformat(),
                "states": [
                    {
                        "name": "Uninitialized",
                        "description": "Initial state before cluster initialization",
                        "is_initial": True,
                        "is_final": False,
                        "attributes_monitored": [],
                        "invariants": []
                    },
                    {
                        "name": "Ready",
                        "description": "Cluster is ready for operation",
                        "is_initial": False,
                        "is_final": False,
                        "attributes_monitored": [],
                        "invariants": []
                    },
                    {
                        "name": "Error",
                        "description": "Error state - FSM generation failed",
                        "is_initial": False,
                        "is_final": True,
                        "attributes_monitored": [],
                        "invariants": []
                    }
                ],
                "transitions": [
                    {
                        "from_state": "Uninitialized",
                        "to_state": "Ready",
                        "trigger": "initialization",
                        "guard_condition": "true",
                        "actions": ["initialize_cluster"],
                        "response_command": None
                    }
                ],
                "initial_state": "Uninitialized",
                "final_states": ["Error"],
                "attributes_used": [],
                "commands_handled": [],
                "events_generated": [],
                "formal_properties": {
                    "safety_properties": ["No invalid state transitions"],
                    "liveness_properties": ["Eventually reaches Ready state"],
                    "invariants": ["State consistency"]
                },
                "verification_notes": {
                    "model_assumptions": ["Fallback model due to generation failure"],
                    "coverage_analysis": "Minimal coverage - generation failed",
                    "limitations": ["This is a fallback model", "Actual cluster behavior not modeled"]
                },
                "generation_error": "FSM generation failed - using fallback model"
            }
        }
    
    def save_fsm_model(self, fsm_model: Dict[str, Any], cluster_name: str, section_number: str = None):
        """
        Save FSM model to JSON file and generate formal specification files (.tla, .pml)
        
        Args:
            fsm_model: Generated FSM model
            cluster_name: Name of the cluster
            section_number: Section number for filename
        """
        try:
            # Clean cluster name for filename
            safe_name = "".join(c for c in cluster_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_name = safe_name.replace(' ', '_')
            
            # Create filename base
            if section_number:
                filename_base = f"{section_number}_{safe_name}"
            else:
                filename_base = f"{safe_name}"
            
            # Save JSON file
            json_filepath = os.path.join(self.output_dir, f"{filename_base}_fsm.json")
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(fsm_model, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved FSM JSON to: {json_filepath}")
            
            # Generate and save TLA+ specification
            tla_content = self._generate_tla_specification(fsm_model, cluster_name)
            tla_filepath = os.path.join(self.output_dir, f"{filename_base}_spec.tla")
            with open(tla_filepath, 'w', encoding='utf-8') as f:
                f.write(tla_content)
            logger.info(f"Saved TLA+ spec to: {tla_filepath}")
            
            # Generate and save Promela model
            promela_content = self._generate_promela_model(fsm_model, cluster_name)
            promela_filepath = os.path.join(self.output_dir, f"{filename_base}_model.pml")
            with open(promela_filepath, 'w', encoding='utf-8') as f:
                f.write(promela_content)
            logger.info(f"Saved Promela model to: {promela_filepath}")
            
        except Exception as e:
            logger.error(f"Error saving FSM model for {cluster_name}: {e}")
    
    def _generate_tla_specification(self, fsm_model: Dict[str, Any], cluster_name: str) -> str:
        """
        Generate TLA+ formal specification from FSM model
        
        Args:
            fsm_model: FSM model dictionary
            cluster_name: Name of the cluster
            
        Returns:
            TLA+ specification as string
        """
        model = fsm_model.get("fsm_model", {})
        cluster_id = model.get("cluster_id", "unknown")
        states = model.get("states", [])
        transitions = model.get("transitions", [])
        attributes = model.get("attributes_used", [])
        commands = model.get("commands_handled", [])
        
        # Create safe module name
        module_name = "".join(c for c in cluster_name if c.isalnum())
        
        # Generate state names
        state_names = [state.get("name", f"State{i}") for i, state in enumerate(states)]
        initial_state = model.get("initial_state", state_names[0] if state_names else "Init")
        
        tla_content = f'''---- MODULE {module_name}Cluster ----
EXTENDS Naturals, Sequences, FiniteSets, TLC

{chr(92)}* Matter {cluster_name} Cluster Specification
{chr(92)}* Cluster ID: {cluster_id}
{chr(92)}* Generated: {model.get("generation_timestamp", "unknown")}

{chr(92)}* Constants
CONSTANTS
    MAX_USERS,           {chr(92)}* Maximum concurrent users
    MAX_COMMANDS,        {chr(92)}* Maximum pending commands
    TIMEOUT_DURATION     {chr(92)}* Command timeout duration

{chr(92)}* Variables
VARIABLES
    cluster_state,       {chr(92)}* Current cluster state: {{{", ".join(f'"{s}"' for s in state_names)}}}
    attribute_values,    {chr(92)}* Attribute value mappings
    pending_commands,    {chr(92)}* Queue of pending commands
    user_sessions,       {chr(92)}* Active user sessions
    event_history,       {chr(92)}* Generated events history
    error_conditions     {chr(92)}* Current error conditions

{chr(92)}* Type definitions
ClusterState == {{{", ".join(f'"{s}"' for s in state_names)}}}
AttributeType == {{{", ".join(f'"{attr}"' for attr in attributes[:10]) if attributes else '"none"'}}}
CommandType == {{{", ".join(f'"{cmd}"' for cmd in commands[:10]) if commands else '"nop"'}}}

{chr(92)}* Initial state predicate
Init ==
    /\\ cluster_state = "{initial_state}"
    /\\ attribute_values = [attr \\in AttributeType |-> "default"]
    /\\ pending_commands = <<>>
    /\\ user_sessions = {{}}
    /\\ event_history = <<>>
    /\\ error_conditions = {{}}

'''

        # Add transition predicates
        backslash = chr(92)
        for i, transition in enumerate(transitions[:10]):  # Limit to first 10 transitions
            from_state = transition.get("from_state", "Unknown")
            to_state = transition.get("to_state", "Unknown")
            trigger = transition.get("trigger", f"action{i}")
            guard = transition.get("guard_condition", "TRUE")
            actions = transition.get("actions", [])
            
            # Handle trigger as list or string
            if isinstance(trigger, list):
                trigger_name = "_or_".join([str(t).replace(" ", "") for t in trigger[:3]])  # Limit to first 3
            else:
                trigger_name = str(trigger).replace(" ", "")
            
            tla_content += f'''{backslash}* Transition: {from_state} -> {to_state}
{trigger_name}Action ==
    /\\ cluster_state = "{from_state}"
    /\\ {str(guard).replace("true", "TRUE").replace("false", "FALSE") if guard and str(guard) != "true" else "TRUE"}
    /\\ cluster_state' = "{to_state}"
    /\\ UNCHANGED <<attribute_values, pending_commands, user_sessions, event_history, error_conditions>>

'''

        # Add safety properties
        tla_content += f'''{backslash}* Safety properties
TypeInvariant ==
    /\\ cluster_state \\in ClusterState
    /\\ {backslash}A attr \\in DOMAIN attribute_values : attr \\in AttributeType
    /\\ Len(pending_commands) <= MAX_COMMANDS

SafetyInvariant ==
    /\\ cluster_state \\in ClusterState
    /\\ (cluster_state = "Error") => (Cardinality(error_conditions) > 0)

{backslash}* Liveness properties
EventualProgress ==
    /\\ (cluster_state = "Processing") ~> (cluster_state \\in {{"Ready", "Error"}})

{backslash}* Next state relation
Next ==
    {backslash}/ {f" {backslash}/ ".join([
        f'{str(t.get("trigger", f"action{i}")).replace(" ", "") if isinstance(t.get("trigger"), str) else "_or_".join([str(x).replace(" ", "") for x in t.get("trigger", [f"action{i}"])[:3]])}Action'
        for i, t in enumerate(transitions[:10])
    ])}

{backslash}* Specification
Spec == Init /\\ [][Next]_<<cluster_state, attribute_values, pending_commands, user_sessions, event_history, error_conditions>>

{backslash}* Fairness conditions
Fairness == WF_<<cluster_state, attribute_values, pending_commands, user_sessions, event_history, error_conditions>>(Next)

====
'''
        return tla_content
    
    def _generate_promela_model(self, fsm_model: Dict[str, Any], cluster_name: str) -> str:
        """
        Generate Promela model for SPIN model checker
        
        Args:
            fsm_model: FSM model dictionary
            cluster_name: Name of the cluster
            
        Returns:
            Promela model as string
        """
        model = fsm_model.get("fsm_model", {})
        cluster_id = model.get("cluster_id", "unknown")
        states = model.get("states", [])
        transitions = model.get("transitions", [])
        commands = model.get("commands_handled", [])
        
        # Flatten commands if they are nested lists
        flat_commands = []
        for cmd in commands[:8]:  # Limit to first 8
            if isinstance(cmd, list):
                flat_commands.extend([str(c) for c in cmd[:3]])  # Limit nested items
            else:
                flat_commands.append(str(cmd))
        
        # Generate state enumeration
        state_names = [state.get("name", f"state{i}").lower().replace(" ", "_") for i, state in enumerate(states)]
        initial_state = model.get("initial_state", state_names[0] if state_names else "init").lower().replace(" ", "_")
        
        promela_content = f'''/*
 * Promela Model for Matter {cluster_name} Cluster
 * Cluster ID: {cluster_id}
 * Generated: {model.get("generation_timestamp", "unknown")}
 * For verification with SPIN model checker
 */

#define MAX_USERS 3
#define MAX_COMMANDS 5

/* Message types */
mtype = {{ {", ".join(f'{cmd.lower().replace(" ", "_")}' for cmd in flat_commands[:8]) if flat_commands else "nop"} }};

/* State enumeration */
mtype = {{ {", ".join(state_names)} }};

/* Global variables */
mtype cluster_state = {initial_state};
byte active_users = 0;
bool error_condition = false;
chan user_commands = [MAX_COMMANDS] of {{ mtype, byte }};
chan cluster_events = [MAX_COMMANDS] of {{ mtype, byte }};
chan security_alerts = [MAX_COMMANDS] of {{ mtype, byte }};

/* User session structure */
typedef UserSession {{
    bool authenticated;
    byte role;
    byte session_id;
}};

UserSession users[MAX_USERS];

/* Security context */
typedef SecurityContext {{
    byte threat_level;
    bool access_granted;
}};

SecurityContext security;

/* Initialize cluster */
inline init_cluster() {{
    cluster_state = {initial_state};
    error_condition = false;
    security.threat_level = 0;
    security.access_granted = true;
}}

/* Process command with security checks */
inline process_command(cmd, user_id) {{
    if
    :: (users[user_id].authenticated && security.access_granted) ->
        atomic {{
            if
'''

        # Add state transitions
        for transition in transitions[:8]:  # Limit to first 8 transitions
            from_state = transition.get("from_state", "unknown").lower().replace(" ", "_")
            to_state = transition.get("to_state", "unknown").lower().replace(" ", "_")
            trigger = transition.get("trigger", "unknown")
            
            # Handle trigger as list or string
            if isinstance(trigger, list):
                trigger_name = "_or_".join([str(t).lower().replace(" ", "_") for t in trigger[:3]])
            else:
                trigger_name = str(trigger).lower().replace(" ", "_")
            
            promela_content += f'''            :: (cluster_state == {from_state}) ->
                cluster_state = {to_state};
                cluster_events ! {trigger_name}, user_id;
'''

        promela_content += f'''            :: else -> 
                security_alerts ! error, user_id;
            fi
        }}
        cluster_state = {state_names[1] if len(state_names) > 1 else initial_state};
        cluster_events ! cmd, user_id;
    :: else -> 
        security_alerts ! error, user_id;
    fi
}}

/* User authentication process */
proctype UserAuth(byte uid) {{
    users[uid].session_id = uid;
    
    do
    :: atomic {{
        if
        :: (active_users < MAX_USERS) ->
            users[uid].authenticated = true;
            users[uid].role = 1; /* default user role */
            active_users++;
            break;
        :: else ->
            printf("Max users reached{chr(92)}n");
            break;
        fi
    }}
    od;
    
    /* User activity loop */
    do
    :: user_commands ? cmd, _ ->
        process_command(cmd, uid);
    :: timeout ->
        users[uid].authenticated = false;
        active_users--;
        break;
    od
}}

/* Cluster state machine */
proctype ClusterStateMachine() {{
    mtype cmd;
    byte user_id;
    
    init_cluster();
    
    do
    :: user_commands ? cmd, user_id ->
        process_command(cmd, user_id);
    :: timeout ->
        if
        :: (cluster_state == processing) ->
            cluster_state = error;
            security_alerts ! timeout, 0;
        :: else -> skip;
        fi
    :: (cluster_state == error) ->
        cluster_state = {initial_state};
        printf("Cluster recovered from error{chr(92)}n");
    od
}}

/* Security monitor */
proctype SecurityMonitor() {{
    mtype alert_type;
    byte context;
    
    do
    :: security_alerts ? alert_type, context ->
        security.threat_level++;
        if
        :: (security.threat_level > 3) ->
            printf("CRITICAL SECURITY THREAT DETECTED{chr(92)}n");
            security.access_granted = false;
        :: else -> skip;
        fi
    :: timeout ->
        if
        :: (security.threat_level > 0) ->
            security.threat_level--;
        :: else -> skip;
        fi
    od
}}

/* LTL Properties for verification */
ltl safety1 {{ [](cluster_state == processing -> <>(cluster_state == {initial_state} || cluster_state == error)) }}
ltl safety2 {{ []((active_users > 0) -> <>(active_users == 0)) }}
ltl security1 {{ [](security.threat_level == 4 -> <>security.threat_level < 4) }}
ltl liveness1 {{ []<>(cluster_state == {initial_state}) }}

/* Main process */
init {{
    atomic {{
        run ClusterStateMachine();
        run SecurityMonitor();
        run UserAuth(0);
        run UserAuth(1);
    }}
}}
'''
        return promela_content
    
    def generate_all_fsms(self, limit: Optional[int] = None, skip_existing: bool = True) -> Dict[str, Any]:
        """
        Generate FSM models for all clusters
        
        Args:
            limit: Optional limit on number of clusters to process
            skip_existing: Whether to skip clusters that already have FSM files
            
        Returns:
            Summary of generation results
        """
        if not self.clusters_data or 'clusters' not in self.clusters_data:
            logger.error("No clusters data found")
            return {"error": "No clusters data found"}
        
        clusters = self.clusters_data['clusters']
        total_clusters = len(clusters)
        
        if limit:
            clusters = clusters[:limit]
            logger.info(f"Processing {limit} out of {total_clusters} clusters")
        else:
            logger.info(f"Processing all {total_clusters} clusters")
        
        results = {
            "total_clusters": total_clusters,
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "generation_summary": [],
            "start_time": datetime.now().isoformat()
        }
        
        for i, cluster_info in enumerate(clusters, 1):
            cluster_data = cluster_info.get('cluster_info', {})
            cluster_name = cluster_data.get('cluster_name', f'Cluster_{i}')
            section_number = cluster_info.get('metadata', {}).get('section_number', 'Unknown')
            
            logger.info(f"[{i}/{len(clusters)}] Processing: {cluster_name} (Section {section_number})")
            
            # Check if FSM already exists
            if skip_existing:
                safe_name = "".join(c for c in cluster_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_name = safe_name.replace(' ', '_')
                expected_file = os.path.join(self.output_dir, f"{section_number}_{safe_name}_fsm.json")
                
                if os.path.exists(expected_file):
                    logger.info(f"FSM already exists for {cluster_name}, skipping...")
                    results["skipped"] += 1
                    results["generation_summary"].append({
                        "cluster_name": cluster_name,
                        "section_number": section_number,
                        "status": "skipped",
                        "reason": "FSM file already exists"
                    })
                    continue
            
            try:
                # Generate FSM
                fsm_model = self.generate_cluster_fsm(cluster_info)
                
                # Save FSM
                self.save_fsm_model(fsm_model, cluster_name, section_number)
                
                results["successful"] += 1
                results["generation_summary"].append({
                    "cluster_name": cluster_name,
                    "section_number": section_number,
                    "status": "success",
                    "has_generation_error": "generation_error" in fsm_model.get("fsm_model", {})
                })
                
            except Exception as e:
                logger.error(f"Failed to process {cluster_name}: {e}")
                results["failed"] += 1
                results["generation_summary"].append({
                    "cluster_name": cluster_name,
                    "section_number": section_number,
                    "status": "failed",
                    "error": str(e)
                })
            
            results["processed"] += 1
        
        results["end_time"] = datetime.now().isoformat()
        
        # Save generation summary
        summary_file = os.path.join(self.output_dir, "generation_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generation complete: {results['successful']} successful, {results['failed']} failed, {results['skipped']} skipped")
        logger.info(f"Summary saved to: {summary_file}")
        
        return results
    
    def get_existing_fsm_files(self) -> List[str]:
        """Get list of existing FSM files in output directory"""
        try:
            return [f for f in os.listdir(self.output_dir) if f.endswith('_fsm.json')]
        except Exception as e:
            logger.error(f"Error listing FSM files: {e}")
            return []
    
    def validate_fsm_model(self, fsm_model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the structure and consistency of an FSM model
        
        Args:
            fsm_model: FSM model to validate
            
        Returns:
            Validation results
        """
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            model = fsm_model.get("fsm_model", {})
            
            # Check required fields
            required_fields = ["cluster_name", "states", "transitions", "initial_state"]
            for field in required_fields:
                if field not in model:
                    validation_results["errors"].append(f"Missing required field: {field}")
                    validation_results["is_valid"] = False
            
            # Validate states
            states = model.get("states", [])
            if not states:
                validation_results["errors"].append("No states defined")
                validation_results["is_valid"] = False
            else:
                state_names = {state.get("name") for state in states}
                
                # Check initial state exists
                initial_state = model.get("initial_state")
                if initial_state and initial_state not in state_names:
                    validation_results["errors"].append(f"Initial state '{initial_state}' not found in states")
                    validation_results["is_valid"] = False
                
                # Check transitions reference valid states
                for transition in model.get("transitions", []):
                    from_state = transition.get("from_state")
                    to_state = transition.get("to_state")
                    
                    if from_state and from_state not in state_names:
                        validation_results["errors"].append(f"Transition references unknown from_state: {from_state}")
                        validation_results["is_valid"] = False
                    
                    if to_state and to_state not in state_names:
                        validation_results["errors"].append(f"Transition references unknown to_state: {to_state}")
                        validation_results["is_valid"] = False
            
            # Check for unreachable states (warning)
            if states and "transitions" in model:
                reachable_states = {model.get("initial_state")}
                for transition in model["transitions"]:
                    reachable_states.add(transition.get("to_state"))
                
                for state in states:
                    if state.get("name") not in reachable_states:
                        validation_results["warnings"].append(f"State '{state.get('name')}' may be unreachable")
        
        except Exception as e:
            validation_results["errors"].append(f"Validation error: {e}")
            validation_results["is_valid"] = False
        
        return validation_results


def main():
    """Main function"""
    # File paths
    detailed_json_path = "matter_clusters_detailed.json"
    output_dir = "fsm_results"
    
    # Check if input file exists
    if not os.path.exists(detailed_json_path):
        logger.error(f"Input file not found: {detailed_json_path}")
        logger.error("Please run cluster_detail_extractor.py first to generate the detailed clusters JSON")
        return
    
    try:
        # Initialize FSM generator
        logger.info("Initializing FSM generator...")
        generator = ClusterFSMGenerator(detailed_json_path, output_dir)
        
        # Show existing FSM files
        existing_files = generator.get_existing_fsm_files()
        if existing_files:
            logger.info(f"Found {len(existing_files)} existing FSM files")
        
        # Generate FSMs for all clusters
        logger.info("Starting FSM generation for all clusters...")
        results = generator.generate_all_fsms(
            limit=3,  # Process all clusters - change to number for testing specific amount
            skip_existing=False  # Skip clusters that already have FSM files
        )
        
        # Print summary
        print("\n" + "="*60)
        print("FSM GENERATION SUMMARY")
        print("="*60)
        print(f"Total clusters: {results['total_clusters']}")
        print(f"Processed: {results['processed']}")
        print(f"Successful: {results['successful']}")
        print(f"Failed: {results['failed']}")
        print(f"Skipped: {results['skipped']}")
        print(f"Output directory: {output_dir}")
        print("="*60)
        
        if results['failed'] > 0:
            print("\nFailed clusters:")
            for item in results['generation_summary']:
                if item['status'] == 'failed':
                    print(f"  - {item['cluster_name']} (Section {item['section_number']}): {item.get('error', 'Unknown error')}")
        
        print(f"\nFSM models saved to: {os.path.abspath(output_dir)}")
        
    except KeyboardInterrupt:
        logger.warning("Generation interrupted by user")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
