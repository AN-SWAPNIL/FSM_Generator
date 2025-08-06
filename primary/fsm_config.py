"""
Configuration constants for FSM Generator RAG System
"""

import os
from pathlib import Path

# File types to process for Matter specification
MATTER_SPEC_EXTENSIONS = ['.html', '.md', '.txt', '.pdf']

# Exclude directories and files
EXCLUDE_DIRS = ['__pycache__', '.venv', '.git', '.idea', 'venv', 'env', 'node_modules', 'dist', 'build', '.vscode']
EXCLUDE_FILES = ['requirements.txt', 'package.json', 'package-lock.json']

# Model configuration
DEFAULT_MODEL_CONFIG = {
    "model_name": "deepseek-r1",
    "temperature": 0.1,  # Low temperature for consistent FSM generation
    "max_tokens": 4096,
    "chunk_size": 2000,  # Larger chunks for better context
    "chunk_overlap": 400,
    "k_retrieval": 4,  # Number of documents to retrieve
}

# FSM Generation prompts and templates
FSM_SYSTEM_PROMPT = """You are an expert protocol analysis assistant specialized in generating Finite State Machines (FSMs) from the Matter IoT specification. 

Your expertise includes:
- Deep understanding of Matter protocol architecture
- State machine modeling and formal verification
- Protocol state analysis and transition identification
- IoT device lifecycle management
- Network protocol state modeling

Always provide clear, implementable FSM definitions with proper state identification, transition conditions, and event handling."""

FSM_PATTERNS = {
    "device_commissioning": {
        "description": "Device commissioning and onboarding process",
        "typical_states": ["Unprovisioned", "Discovering", "Pairing", "Authenticating", "Provisioning", "Operational", "Error"],
        "key_events": ["commission_start", "discovery_complete", "pairing_success", "auth_complete", "provision_complete", "error_occurred"]
    },
    "connection_management": {
        "description": "Network connection lifecycle management",
        "typical_states": ["Disconnected", "Connecting", "Connected", "Authenticated", "Secured", "Timeout", "Error"],
        "key_events": ["connect_request", "connection_established", "authentication_success", "security_established", "timeout", "disconnect"]
    },
    "command_processing": {
        "description": "Command and response processing lifecycle",
        "typical_states": ["Idle", "Receiving", "Processing", "Responding", "Error", "Timeout"],
        "key_events": ["command_received", "processing_start", "processing_complete", "response_sent", "error_occurred", "timeout"]
    },
    "security_handshake": {
        "description": "Security establishment and key exchange",
        "typical_states": ["Insecure", "Handshake_Init", "Key_Exchange", "Verification", "Secured", "Failed"],
        "key_events": ["handshake_start", "key_generated", "verification_success", "security_established", "handshake_failed"]
    }
}

# FSM output template
FSM_OUTPUT_TEMPLATE = """
FSM_NAME: {name}
DESCRIPTION: {description}

STATES:
{states}

INITIAL_STATE: {initial_state}

TRANSITIONS:
{transitions}

EVENTS:
{events}

NOTES:
{notes}
"""

# Matter specification context keywords for better retrieval
MATTER_CONTEXT_KEYWORDS = {
    "commissioning": ["commission", "pairing", "onboard", "provision", "setup", "join", "enroll"],
    "connection": ["connect", "establish", "session", "channel", "link", "association"],
    "security": ["secure", "encrypt", "authenticate", "certificate", "key", "credential", "trust"],
    "command": ["command", "request", "response", "invoke", "execute", "operation"],
    "state": ["state", "status", "mode", "phase", "stage", "condition"],
    "error": ["error", "fault", "exception", "failure", "timeout", "abort"],
    "device": ["device", "node", "endpoint", "cluster", "attribute"],
    "network": ["network", "fabric", "topology", "routing", "discovery"]
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "fsm_generator.log"
}
