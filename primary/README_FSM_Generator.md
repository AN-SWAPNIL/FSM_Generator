# FSM Generator RAG System for Matter Specification

A comprehensive Retrieval Augmented Generation (RAG) system for generating Finite State Machines (FSMs) from the Matter IoT specification, based on the ProtocolGPT implementation but specialized for Matter protocol analysis.

## Overview

This system combines the power of Large Language Models with retrieval-augmented generation to analyze Matter specification documents and extract implementable FSM definitions for various protocol aspects like device commissioning, connection management, security procedures, and command processing.

## Architecture

The system is built on several key components:

1. **FSMGeneratorRAG** - Main RAG system with document loading and agent creation
2. **AdvancedFSMGenerator** - Specialized FSM generation with query analysis and validation
3. **FSMParser/Validator** - FSM parsing and correctness validation
4. **Prompt Management** - Specialized prompts for different FSM types

## Features

### 🎯 Intelligent FSM Generation

- **Pattern Recognition**: Automatically identifies FSM patterns (commissioning, connection, security, command processing)
- **Context-Aware Retrieval**: Searches Matter specification for relevant protocol information
- **Comprehensive Analysis**: Extracts states, transitions, events, and error handling mechanisms

### 🔍 Advanced Query Processing

- **Query Enhancement**: Automatically adds relevant context based on query content
- **Specialized Prompts**: Uses different prompting strategies for different FSM types
- **Multi-step Reasoning**: Agent can perform multiple retrieval steps for complex queries

### ✅ FSM Validation & Export

- **Syntax Validation**: Ensures FSM definitions are well-formed
- **Completeness Checking**: Validates state reachability and transition coverage
- **Multiple Export Formats**: JSON, Python enums, Mermaid diagrams

### 💬 Interactive Interface

- **Conversational UI**: Natural language FSM generation requests
- **Real-time Feedback**: Validation results and suggestions
- **Export Commands**: Easy export to different formats

## Installation

### Required Packages

```bash
pip install --upgrade langchain langchain-community langchain-core langchain-ollama langgraph beautifulsoup4
```

### Model Setup

Ensure you have Ollama installed and the DeepSeek-R1 model available:

```bash
ollama serve
ollama pull deepseek-r1
```

## Usage

### Basic Usage

```python
from fsm_generator_rag import FSMGeneratorRAG

# Initialize with Matter specification
fsm_generator = FSMGeneratorRAG("path/to/Matter_Specification.html")

# Generate FSM
result = fsm_generator.generate_fsm("Generate FSM for device commissioning process")
print(result)
```

### Interactive Mode

```python
# Start interactive session
fsm_generator.interactive_fsm_generation()
```

### Export FSMs

```python
# Export in different formats
json_fsm = fsm_generator.export_fsm("commissioning FSM", "json")
python_fsm = fsm_generator.export_fsm("commissioning FSM", "python")
mermaid_fsm = fsm_generator.export_fsm("commissioning FSM", "mermaid")
```

## File Structure

```
demo/
├── fsm_generator_rag.py          # Main RAG system
├── advanced_fsm_generator.py     # Advanced generation with specialized prompts
├── fsm_config.py                 # Configuration constants and patterns
├── fsm_utils.py                  # Utilities, parsing, and validation
├── test_fsm_generator.py         # Tests and examples
└── Matter_Specification.html     # Matter specification document
```

## FSM Output Format

The system generates FSMs in a structured format:

```
FSM_NAME: Device Commissioning FSM
DESCRIPTION: State machine for Matter device commissioning process

STATES:
- Unprovisioned: Device in factory reset state
- Discovering: Device advertising for commissioning
- Pairing: Device undergoing pairing process
- Operational: Device successfully commissioned

INITIAL_STATE: Unprovisioned

TRANSITIONS:
- FROM: Unprovisioned | EVENT: commission_start | TO: Discovering
- FROM: Discovering | EVENT: pairing_request | TO: Pairing
- FROM: Pairing | EVENT: pairing_success | TO: Operational

EVENTS:
- commission_start: User initiates commissioning
- pairing_request: Commissioner requests pairing
- pairing_success: Pairing completed successfully
```

## Supported FSM Patterns

### 🔧 Device Commissioning

- Device discovery and advertisement
- Pairing and authentication procedures
- Network provisioning steps
- Operational state transitions

### 🌐 Connection Management

- Connection establishment procedures
- Session management and keep-alive
- Reconnection strategies and timeout handling
- Error recovery mechanisms

### 🔐 Security Handshake

- Security establishment procedures
- Key exchange and certificate validation
- Trust establishment protocols
- Security error handling

### ⚡ Command Processing

- Command reception and parsing
- Processing workflows and validation
- Response generation and transmission
- Error handling and timeouts

## Example Queries

- "Generate FSM for Matter device commissioning process"
- "Create state machine for secure connection establishment"
- "Extract FSM for command processing with error handling"
- "Generate security handshake state machine for Matter devices"
- "Create FSM for device discovery and pairing"

## Testing

Run the test suite to verify functionality:

```bash
python test_fsm_generator.py
```

This runs:

- Basic FSM generation tests
- FSM parsing and validation tests
- Export format tests
- Interactive demo

## Integration with ProtocolGPT

This system adapts the ProtocolGPT approach for Matter specification analysis:

### Key Adaptations:

1. **Document Focus**: Specialized for Matter HTML specifications instead of protocol implementations
2. **FSM-Specific Prompts**: Tailored prompts for state machine extraction
3. **Pattern Recognition**: Built-in knowledge of common IoT protocol patterns
4. **Validation Pipeline**: Comprehensive FSM validation and export capabilities

### Similarities to ProtocolGPT:

1. **RAG Architecture**: Uses retrieval-augmented generation for context
2. **Conversational Interface**: Interactive chat-based analysis
3. **Memory Management**: Maintains conversation history and context
4. **Modular Design**: Separated concerns for different functionalities

## Advanced Features

### Query Analysis

The system automatically analyzes queries to:

- Identify FSM type (commissioning, connection, security, command)
- Add relevant context and keywords
- Select appropriate specialized prompts

### Multi-Step Reasoning

The agent can:

- Perform multiple retrieval steps for complex queries
- Iterate on FSM generation based on retrieved context
- Handle follow-up questions and refinements

### Validation Pipeline

Comprehensive validation includes:

- Syntax checking for FSM format
- State reachability analysis
- Transition completeness verification
- Event consistency validation

## Troubleshooting

### Common Issues:

1. **Model Not Found**: Ensure DeepSeek-R1 is installed with `ollama pull deepseek-r1`
2. **Import Errors**: Install missing packages with pip
3. **File Not Found**: Verify Matter_Specification.html path is correct
4. **Memory Issues**: Reduce chunk_size in config for large documents

### Performance Tips:

1. **Chunk Size**: Adjust chunk_size and chunk_overlap for optimal context
2. **Retrieval Count**: Modify k_retrieval for more/less context
3. **Temperature**: Lower temperature (0.1) for more consistent FSMs
4. **Model Choice**: Use larger models for more complex FSM generation

This system provides a powerful foundation for extracting and generating FSMs from protocol specifications, with specific optimizations for the Matter IoT protocol.
