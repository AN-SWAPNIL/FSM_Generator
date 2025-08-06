"""
FSM Generation utilities and helper functions
"""

import re
import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

@dataclass
class FSMState:
    """Represents a state in an FSM"""
    name: str
    description: str
    is_initial: bool = False
    is_final: bool = False

@dataclass 
class FSMTransition:
    """Represents a transition between states"""
    from_state: str
    to_state: str
    event: str
    condition: str = None
    action: str = None

@dataclass
class FSMEvent:
    """Represents an event that can trigger transitions"""
    name: str
    description: str
    parameters: List[str] = None

@dataclass
class FSM:
    """Complete FSM definition"""
    name: str
    description: str
    states: List[FSMState]
    transitions: List[FSMTransition]
    events: List[FSMEvent]
    initial_state: str
    
class FSMParser:
    """Parse and validate FSM definitions"""
    
    @staticmethod
    def parse_fsm_text(fsm_text: str) -> FSM:
        """Parse FSM from text format"""
        lines = fsm_text.strip().split('\n')
        
        name = ""
        description = ""
        states = []
        transitions = []
        events = []
        initial_state = ""
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('FSM_NAME:'):
                name = line.split(':', 1)[1].strip()
            elif line.startswith('DESCRIPTION:'):
                description = line.split(':', 1)[1].strip()
            elif line.startswith('STATES:'):
                current_section = 'states'
            elif line.startswith('TRANSITIONS:'):
                current_section = 'transitions'
            elif line.startswith('EVENTS:'):
                current_section = 'events'
            elif line.startswith('INITIAL_STATE:'):
                initial_state = line.split(':', 1)[1].strip()
            elif line.startswith('-') and current_section:
                FSMParser._parse_section_item(line, current_section, states, transitions, events)
        
        return FSM(name, description, states, transitions, events, initial_state)
    
    @staticmethod
    def _parse_section_item(line: str, section: str, states: List, transitions: List, events: List):
        """Parse individual items in each section"""
        content = line[1:].strip()  # Remove the '-' prefix
        
        if section == 'states':
            if ':' in content:
                state_name, state_desc = content.split(':', 1)
                states.append(FSMState(state_name.strip(), state_desc.strip()))
            else:
                states.append(FSMState(content, ""))
                
        elif section == 'transitions':
            # Parse format: FROM: state | EVENT: event | CONDITION: cond | TO: state
            parts = content.split('|')
            from_state = ""
            to_state = ""
            event = ""
            condition = None
            
            for part in parts:
                part = part.strip()
                if part.startswith('FROM:'):
                    from_state = part.split(':', 1)[1].strip()
                elif part.startswith('TO:'):
                    to_state = part.split(':', 1)[1].strip()
                elif part.startswith('EVENT:'):
                    event = part.split(':', 1)[1].strip()
                elif part.startswith('CONDITION:'):
                    condition = part.split(':', 1)[1].strip()
            
            if from_state and to_state and event:
                transitions.append(FSMTransition(from_state, to_state, event, condition))
                
        elif section == 'events':
            if ':' in content:
                event_name, event_desc = content.split(':', 1)
                events.append(FSMEvent(event_name.strip(), event_desc.strip()))
            else:
                events.append(FSMEvent(content, ""))

class FSMValidator:
    """Validate FSM definitions for correctness"""
    
    @staticmethod
    def validate_fsm(fsm: FSM) -> Tuple[bool, List[str]]:
        """Validate FSM and return (is_valid, errors)"""
        errors = []
        
        # Check if FSM has required components
        if not fsm.name:
            errors.append("FSM must have a name")
        if not fsm.states:
            errors.append("FSM must have at least one state")
        if not fsm.initial_state:
            errors.append("FSM must have an initial state")
            
        # Check if initial state exists
        state_names = [state.name for state in fsm.states]
        if fsm.initial_state not in state_names:
            errors.append(f"Initial state '{fsm.initial_state}' not found in states")
            
        # Check transitions reference valid states
        for transition in fsm.transitions:
            if transition.from_state not in state_names:
                errors.append(f"Transition references unknown from_state: {transition.from_state}")
            if transition.to_state not in state_names:
                errors.append(f"Transition references unknown to_state: {transition.to_state}")
                
        # Check for unreachable states (except initial state)
        reachable_states = {fsm.initial_state}
        for transition in fsm.transitions:
            if transition.from_state in reachable_states:
                reachable_states.add(transition.to_state)
                
        unreachable = set(state_names) - reachable_states
        if unreachable:
            errors.append(f"Unreachable states found: {unreachable}")
            
        return len(errors) == 0, errors

class FSMGenerator:
    """Generate FSM code in various formats"""
    
    @staticmethod
    def to_python_enum(fsm: FSM) -> str:
        """Generate Python enum code for FSM states"""
        code = f'class {fsm.name.replace(" ", "")}State(Enum):\n'
        code += f'    """States for {fsm.description}"""\n'
        
        for state in fsm.states:
            state_name = state.name.upper().replace(" ", "_")
            code += f'    {state_name} = "{state.name}"\n'
            
        return code
    
    @staticmethod
    def to_json(fsm: FSM) -> str:
        """Convert FSM to JSON format"""
        fsm_dict = {
            "name": fsm.name,
            "description": fsm.description,
            "initial_state": fsm.initial_state,
            "states": [
                {
                    "name": state.name,
                    "description": state.description,
                    "is_initial": state.is_initial,
                    "is_final": state.is_final
                }
                for state in fsm.states
            ],
            "transitions": [
                {
                    "from_state": trans.from_state,
                    "to_state": trans.to_state,
                    "event": trans.event,
                    "condition": trans.condition,
                    "action": trans.action
                }
                for trans in fsm.transitions
            ],
            "events": [
                {
                    "name": event.name,
                    "description": event.description,
                    "parameters": event.parameters or []
                }
                for event in fsm.events
            ]
        }
        return json.dumps(fsm_dict, indent=2)
    
    @staticmethod
    def to_mermaid(fsm: FSM) -> str:
        """Generate Mermaid state diagram"""
        mermaid = "stateDiagram-v2\n"
        mermaid += f"    title: {fsm.name}\n"
        
        # Add states
        for state in fsm.states:
            if state.name == fsm.initial_state:
                mermaid += f"    [*] --> {state.name.replace(' ', '_')}\n"
        
        # Add transitions
        for transition in fsm.transitions:
            from_state = transition.from_state.replace(' ', '_')
            to_state = transition.to_state.replace(' ', '_')
            label = transition.event
            if transition.condition:
                label += f" [{transition.condition}]"
            mermaid += f"    {from_state} --> {to_state} : {label}\n"
            
        return mermaid

def extract_matter_keywords(text: str) -> List[str]:
    """Extract Matter-specific keywords from text"""
    from fsm_config import MATTER_CONTEXT_KEYWORDS
    
    found_keywords = []
    text_lower = text.lower()
    
    for category, keywords in MATTER_CONTEXT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
                
    return list(set(found_keywords))

def enhance_query_with_context(query: str) -> str:
    """Enhance user query with relevant Matter context"""
    keywords = extract_matter_keywords(query)
    
    context_additions = []
    if any(kw in query.lower() for kw in ['commission', 'pair', 'onboard']):
        context_additions.append("Include device discovery, pairing, and provisioning states.")
    if any(kw in query.lower() for kw in ['connect', 'network', 'session']):
        context_additions.append("Include connection establishment and maintenance states.")
    if any(kw in query.lower() for kw in ['security', 'auth', 'encrypt']):
        context_additions.append("Include security handshake and authentication states.")
    if any(kw in query.lower() for kw in ['command', 'request', 'response']):
        context_additions.append("Include command processing and response generation states.")
        
    if context_additions:
        enhanced = f"{query}\n\nAdditional context: {' '.join(context_additions)}"
        return enhanced
    
    return query
