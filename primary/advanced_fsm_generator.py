"""
Advanced FSM extraction and generation system with specialized prompts
"""

from typing import List, Dict, Any, Optional
import logging
from langchain_core.prompts import PromptTemplate
from fsm_config import FSM_PATTERNS, MATTER_CONTEXT_KEYWORDS
from fsm_utils import enhance_query_with_context, FSMParser, FSMValidator, FSMGenerator

class FSMPromptManager:
    """Manages specialized prompts for different types of FSM generation"""
    
    def __init__(self):
        self.base_system_prompt = self._get_base_system_prompt()
        self.specialized_prompts = self._get_specialized_prompts()
        
    def _get_base_system_prompt(self) -> str:
        return """You are an expert protocol analysis assistant specialized in generating Finite State Machines (FSMs) from the Matter IoT specification.

## Core Expertise:
- Matter protocol architecture and state management
- IoT device lifecycle modeling
- Network protocol state analysis
- Security state machine design
- Error handling and recovery patterns

## Analysis Methodology:
1. **Context Retrieval**: Search for relevant Matter specification sections
2. **Pattern Recognition**: Identify common IoT protocol patterns
3. **State Extraction**: Find explicit and implicit states
4. **Transition Mapping**: Identify triggers, conditions, and state changes
5. **Validation**: Ensure FSM completeness and correctness

## Output Requirements:
- Clear, implementable FSM definitions
- Proper state identification with descriptions
- Complete transition mappings
- Event definitions with triggers
- Error handling and recovery mechanisms
- Industry-standard formatting"""

    def _get_specialized_prompts(self) -> Dict[str, str]:
        return {
            "commissioning": """
## Device Commissioning FSM Generation

Focus on Matter device onboarding and commissioning process:

### Key Areas to Analyze:
- Device discovery mechanisms
- Pairing and authentication procedures
- Network provisioning steps
- Credential establishment
- Operational state transitions

### Expected States:
- Unprovisioned/Factory Reset
- Discoverable/Advertising
- Pairing/Authentication
- Network Provisioning
- Credential Setup
- Operational
- Error/Recovery states

### Critical Events:
- Commission start/trigger
- Discovery completion
- Pairing success/failure
- Authentication events
- Provisioning completion
- Error conditions

Generate a comprehensive commissioning FSM with proper error handling.
""",
            
            "connection": """
## Connection Management FSM Generation

Focus on Matter device network connection lifecycle:

### Key Areas to Analyze:
- Connection establishment procedures
- Session management
- Keep-alive mechanisms
- Reconnection strategies
- Timeout handling

### Expected States:
- Disconnected
- Connecting/Establishing
- Connected/Active
- Authenticated
- Idle/Standby
- Timeout/Retry
- Error states

### Critical Events:
- Connection requests
- Network availability
- Authentication completion
- Timeout events
- Disconnection triggers
- Error conditions

Generate a robust connection management FSM.
""",
            
            "security": """
## Security Handshake FSM Generation

Focus on Matter security establishment and maintenance:

### Key Areas to Analyze:
- Security handshake procedures
- Key exchange mechanisms
- Certificate validation
- Trust establishment
- Security maintenance

### Expected States:
- Insecure
- Handshake Initiation
- Key Exchange
- Certificate Verification
- Trust Establishment
- Secured
- Security Error states

### Critical Events:
- Handshake initiation
- Key generation/exchange
- Certificate presentation
- Verification success/failure
- Security establishment
- Security violations

Generate a comprehensive security FSM with proper error handling.
""",
            
            "command": """
## Command Processing FSM Generation

Focus on Matter command and response processing:

### Key Areas to Analyze:
- Command reception and parsing
- Processing workflows
- Response generation
- Error handling
- Timeout management

### Expected States:
- Idle/Ready
- Receiving Command
- Parsing/Validation
- Processing
- Response Generation
- Sending Response
- Error/Timeout states

### Critical Events:
- Command reception
- Parsing completion
- Validation success/failure
- Processing completion
- Response ready
- Send completion
- Error/timeout events

Generate a detailed command processing FSM.
""",
            
            "general": """
## General Protocol FSM Generation

Analyze the Matter specification for any protocol state machine patterns:

### Analysis Approach:
1. Search for state descriptions and lifecycle information
2. Identify process flows and operational sequences
3. Extract error conditions and recovery mechanisms
4. Map events and triggers
5. Define clear state boundaries

### Output Focus:
- Well-defined states with clear purposes
- Complete transition coverage
- Comprehensive event handling
- Error states and recovery paths
- Practical implementation considerations

Generate a thorough FSM based on the specific request.
"""
        }
    
    def get_prompt_for_query(self, query: str) -> str:
        """Get specialized prompt based on query content"""
        query_lower = query.lower()
        
        # Determine FSM type based on query keywords
        if any(kw in query_lower for kw in ['commission', 'onboard', 'pair', 'provision']):
            prompt_type = "commissioning"
        elif any(kw in query_lower for kw in ['connect', 'session', 'network', 'link']):
            prompt_type = "connection"
        elif any(kw in query_lower for kw in ['security', 'auth', 'encrypt', 'handshake']):
            prompt_type = "security"
        elif any(kw in query_lower for kw in ['command', 'request', 'response', 'process']):
            prompt_type = "command"
        else:
            prompt_type = "general"
            
        # Combine base prompt with specialized prompt
        specialized = self.specialized_prompts[prompt_type]
        
        return f"{self.base_system_prompt}\n\n{specialized}"

class AdvancedFSMGenerator:
    """Advanced FSM generation with query analysis and specialized prompts"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.prompt_manager = FSMPromptManager()
        self.logger = logging.getLogger(__name__)
        
    def generate_fsm(self, query: str, thread_id: str = "advanced_fsm") -> Dict[str, Any]:
        """Generate FSM with advanced analysis and validation"""
        
        # Enhance query with context
        enhanced_query = enhance_query_with_context(query)
        
        # Get specialized prompt
        specialized_prompt = self.prompt_manager.get_prompt_for_query(query)
        
        # Create full generation prompt
        full_prompt = f"""
{specialized_prompt}

## Task Instructions:
1. Use the retrieve_matter_context tool to search for relevant information
2. Analyze the retrieved context thoroughly
3. Generate a complete FSM definition in the specified format
4. Include comprehensive error handling
5. Validate the FSM for completeness

## User Request:
{enhanced_query}

## Required FSM Format:
```
FSM_NAME: [Descriptive name]
DESCRIPTION: [Brief description of what this FSM represents]

STATES:
- STATE_NAME: [Description]
- [Continue for all states...]

INITIAL_STATE: [Name of initial state]

TRANSITIONS:
- FROM: [Source State] | EVENT: [Trigger] | CONDITION: [If any] | TO: [Destination State]
- [Continue for all transitions...]

EVENTS:
- EVENT_NAME: [Description of the event/trigger]
- [Continue for all events...]

NOTES:
- [Any additional implementation notes or considerations]
```

Begin analysis now by retrieving relevant Matter specification context.
"""
        
        try:
            config = {"configurable": {"thread_id": thread_id}}
            messages = [{"role": "user", "content": full_prompt}]
            
            # Generate FSM using the RAG system
            response_parts = []
            for event in self.rag_system.agent_executor.stream(
                {"messages": messages},
                stream_mode="values",
                config=config,
            ):
                last_msg = event["messages"][-1]
                if hasattr(last_msg, 'content') and last_msg.content:
                    response_parts.append(last_msg.content)
            
            full_response = "\n".join(response_parts) if response_parts else "No response generated"
            
            # Try to parse and validate the FSM
            validation_results = self._validate_generated_fsm(full_response)
            
            return {
                "query": query,
                "enhanced_query": enhanced_query,
                "response": full_response,
                "validation": validation_results,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Error in advanced FSM generation: {e}")
            return {
                "query": query,
                "response": f"Error generating FSM: {e}",
                "validation": {"is_valid": False, "errors": [str(e)]},
                "success": False
            }
    
    def _validate_generated_fsm(self, response: str) -> Dict[str, Any]:
        """Validate the generated FSM response"""
        try:
            # Extract FSM definition from response
            fsm_text = self._extract_fsm_from_response(response)
            
            if not fsm_text:
                return {
                    "is_valid": False,
                    "errors": ["No FSM definition found in response"],
                    "fsm": None
                }
            
            # Parse and validate FSM
            fsm = FSMParser.parse_fsm_text(fsm_text)
            is_valid, errors = FSMValidator.validate_fsm(fsm)
            
            return {
                "is_valid": is_valid,
                "errors": errors,
                "fsm": fsm,
                "fsm_text": fsm_text
            }
            
        except Exception as e:
            return {
                "is_valid": False,
                "errors": [f"Validation error: {e}"],
                "fsm": None
            }
    
    def _extract_fsm_from_response(self, response: str) -> Optional[str]:
        """Extract FSM definition from LLM response"""
        import re
        
        # Look for FSM definition blocks
        fsm_patterns = [
            r'```\s*FSM_NAME:.*?```',
            r'FSM_NAME:.*?(?=\n\n|\Z)',
            r'```\s*(FSM_NAME:.*?)```'
        ]
        
        for pattern in fsm_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                fsm_text = match.group(0)
                # Clean up formatting
                fsm_text = re.sub(r'^```\s*', '', fsm_text)
                fsm_text = re.sub(r'```\s*$', '', fsm_text)
                return fsm_text.strip()
        
        # If no clear FSM block found, look for FSM_NAME start
        if "FSM_NAME:" in response:
            start_idx = response.find("FSM_NAME:")
            # Find a reasonable end point
            end_markers = ["\n\n## ", "\n\n# ", "```", "---"]
            end_idx = len(response)
            
            for marker in end_markers:
                marker_idx = response.find(marker, start_idx + 100)  # Skip immediate markers
                if marker_idx != -1:
                    end_idx = min(end_idx, marker_idx)
            
            return response[start_idx:end_idx].strip()
        
        return None
    
    def export_fsm(self, fsm_result: Dict[str, Any], export_format: str = "json") -> str:
        """Export generated FSM in various formats"""
        if not fsm_result["success"] or not fsm_result["validation"]["is_valid"]:
            raise ValueError("Cannot export invalid FSM")
        
        fsm = fsm_result["validation"]["fsm"]
        
        if export_format == "json":
            return FSMGenerator.to_json(fsm)
        elif export_format == "python":
            return FSMGenerator.to_python_enum(fsm)
        elif export_format == "mermaid":
            return FSMGenerator.to_mermaid(fsm)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
