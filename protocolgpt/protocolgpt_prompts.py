"""
ProtocolGPT Prompt Templates for Matter Cluster FSM Extraction
Following ProtocolGPT's proven methodology for practical FSM inference
"""

from typing import Dict
from matter_clusters import MatterClusterInfo

class ProtocolGPTPrompts:
    """
    Prompt templates following ProtocolGPT's approach for FSM extraction
    """
    
    @staticmethod
    def get_cluster_fsm_prompts(cluster: MatterClusterInfo) -> Dict[str, str]:
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
    
    @staticmethod
    def get_interactive_prompt(cluster: MatterClusterInfo, user_question: str) -> str:
        """
        Generate interactive chat prompt with cluster context
        """
        return f"""
Analyzing the {cluster.cluster_name} cluster ({cluster.cluster_id}) in Matter specification.

Cluster Information:
- Description: {cluster.description}
- Category: {cluster.category}
- Attributes: {', '.join(cluster.attributes)}
- Commands: {', '.join(cluster.commands)}
- Events: {', '.join(cluster.events)}

Question: {user_question}
"""
    
    @staticmethod
    def get_batch_processing_prompt(cluster_names: list) -> str:
        """
        Generate prompt for batch processing multiple clusters
        """
        return f"""
Perform FSM analysis for multiple Matter clusters: {', '.join(cluster_names)}

For each cluster, extract:
1. All possible states
2. All message types
3. State transitions with triggering messages

Provide a comprehensive FSM model for each cluster following the same JSON structure as individual extractions.
"""
    
    @staticmethod
    def get_comparative_analysis_prompt(cluster1: MatterClusterInfo, cluster2: MatterClusterInfo) -> str:
        """
        Generate prompt for comparing two clusters
        """
        return f"""
Compare and analyze the FSMs of two Matter clusters:

Cluster 1: {cluster1.cluster_name} ({cluster1.cluster_id})
- Category: {cluster1.category}
- Commands: {', '.join(cluster1.commands)}
- Events: {', '.join(cluster1.events)}

Cluster 2: {cluster2.cluster_name} ({cluster2.cluster_id})
- Category: {cluster2.category}
- Commands: {', '.join(cluster2.commands)}
- Events: {', '.join(cluster2.events)}

Analyze:
1. Similarities in state structures
2. Differences in state transitions
3. Common patterns or behaviors
4. Interaction possibilities between clusters

Provide insights on how these clusters might interact in a Matter device implementation.
"""
    
    @staticmethod
    def get_validation_prompt(cluster: MatterClusterInfo, extracted_fsm: dict) -> str:
        """
        Generate prompt for validating extracted FSM
        """
        return f"""
Validate the following extracted FSM for {cluster.cluster_name} cluster:

Extracted FSM:
{extracted_fsm}

Verification checklist:
1. Are all states logically consistent?
2. Do transitions make sense for the cluster's purpose?
3. Are all commands from the specification covered?
4. Are all events properly represented?
5. Is the initial state appropriate?
6. Are there any missing or incorrect transitions?

Provide feedback and suggest corrections if needed.
"""

    @staticmethod
    def get_system_prompts() -> Dict[str, str]:
        """
        Get system prompts for different types of analysis
        """
        return {
            "fsm_extraction": """You are an expert in protocol analysis and finite state machine extraction. 
Your task is to analyze Matter specification documents and extract accurate FSM models. 
Focus on identifying states, transitions, and the messages that trigger them. 
Provide responses in the requested JSON format without additional explanations.""",
            
            "interactive_analysis": """You are a Matter protocol expert assistant. 
You help users understand Matter clusters, their behaviors, and state machines. 
Provide clear, accurate, and helpful responses based on the Matter specification. 
Reference specific sections of the specification when possible.""",
            
            "validation": """You are a protocol validation expert. 
Your task is to review extracted FSMs for accuracy and completeness. 
Check for logical consistency, missing transitions, and conformance to the Matter specification. 
Provide constructive feedback and suggestions for improvement.""",
            
            "comparison": """You are a protocol architecture analyst. 
Your task is to compare and analyze relationships between different Matter clusters. 
Identify patterns, dependencies, and potential interactions between clusters. 
Provide insights that help understand the overall Matter ecosystem."""
        }

def get_query_types() -> list:
    """Get list of available query types"""
    return ["file_paths", "states", "messages", "transitions", "complete_fsm"]

def validate_query_type(query_type: str) -> bool:
    """Validate if query type is supported"""
    return query_type in get_query_types()

def get_prompt_for_cluster_and_type(cluster: MatterClusterInfo, query_type: str) -> str:
    """
    Get specific prompt for cluster and query type
    
    Args:
        cluster: MatterClusterInfo instance
        query_type: Type of query (states, messages, transitions, etc.)
    
    Returns:
        Formatted prompt string
    
    Raises:
        ValueError: If query_type is not supported
    """
    if not validate_query_type(query_type):
        raise ValueError(f"Unsupported query type: {query_type}. Available: {get_query_types()}")
    
    prompts = ProtocolGPTPrompts.get_cluster_fsm_prompts(cluster)
    return prompts[query_type]
