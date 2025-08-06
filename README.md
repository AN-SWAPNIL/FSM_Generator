# Matter Protocol FSM Generator

A specialized Finite State Machine (FSM) extraction system for the Matter IoT protocol specification, based on the ProtocolGPT approach but adapted for specification documents rather than implementation code.

## Overview

This system follows the ProtocolGPT methodology for extracting protocol state machines, but instead of analyzing C code implementations, it analyzes the Matter specification document to extract FSMs for various protocol aspects.

## Key Differences from ProtocolGPT

### ProtocolGPT Approach (Original)

- **Input**: Protocol implementation code (C files)
- **Goal**: Extract FSMs from actual code implementation
- **Method**: Code-specific prompts asking about states in implementation
- **Example Query**: "Extract IKE protocol states from these code snippets"

### Our Approach (Matter Specification)

- **Input**: Protocol specification document (HTML)
- **Goal**: Extract FSMs from protocol specification
- **Method**: Specification-specific prompts asking about protocol behavior
- **Example Query**: "Extract device commissioning states from Matter specification"

## Architecture

### Core Components

1. **MatterFSMExtractor** (`main.py`)

   - Main extraction system using conversational retrieval chain
   - Based on ProtocolGPT's LLM + vector store approach
   - Adapted for HTML specification documents

2. **FSM Parsing & Validation** (`fsm_utils.py`)

   - Parse JSON FSM responses from LLM
   - Validate FSM completeness and correctness
   - Generate visual representations (Mermaid, DOT)

3. **Advanced Analysis** (`fsm_analyzer.py`)
   - Cross-FSM pattern analysis
   - Comprehensive FSM generation
   - Detailed reporting

### Query Strategy (Following ProtocolGPT)

Just like ProtocolGPT has specialized queries for different protocol aspects:

```python
# ProtocolGPT queries (for code)
"IKE_States": "Extract IKE protocol states from these code snippets..."
"IKE_Pathrule": "Extract state constraints from IKE implementation..."

# Our queries (for specification)
"Device_Commissioning_States": "Extract commissioning states from Matter spec..."
"Security_Handshake_Transitions": "Extract security transitions from Matter spec..."
```

## Usage

### 1. Basic FSM Extraction

```bash
cd FSM_Generator
python main.py
```

This will start an interactive session where you can:

- Select predefined queries (1-8)
- Run all queries with 'all'
- Enter custom queries with 'custom'
- Save results automatically

### 2. Advanced Analysis

```bash
python fsm_analyzer.py
```

This will:

- Load all extracted FSM results
- Validate each FSM
- Generate pattern analysis
- Create comprehensive merged FSM
- Export visualizations

### 3. Available Predefined Queries

1. **Device_Commissioning_States** - Extract commissioning process states
2. **Device_Commissioning_Transitions** - Extract commissioning transitions
3. **Connection_Management_States** - Extract connection lifecycle states
4. **Connection_Management_Transitions** - Extract connection transitions
5. **Security_Handshake_States** - Extract security establishment states
6. **Security_Handshake_Transitions** - Extract security transitions
7. **Command_Processing_States** - Extract command handling states
8. **Command_Processing_Transitions** - Extract command processing transitions
9. **Complete_Device_Lifecycle_FSM** - Extract comprehensive device FSM

## Strong Prompting Strategy

Following ProtocolGPT's approach, we use strong, specific prompts:

### Example: Device Commissioning States

```
As a Matter protocol specialist, your task is to extract the device commissioning states
from the Matter specification. Identify all states involved in the commissioning process,
including initial states, intermediate states, and final states.
Represent the states and their descriptions in JSON format.

Expected states may include: Unprovisioned, Discovering, Pairing, Authenticating,
Provisioning, Operational, Error states, etc.

Format:
{
    "fsm_name": "Matter Device Commissioning",
    "states": [
        {"name": "state_name", "description": "state description"},
        ...
    ]
}
```

## Output Formats

### JSON FSM Structure

```json
{
  "fsm_name": "Matter Device Commissioning",
  "description": "Complete FSM for device commissioning",
  "initial_state": "Unprovisioned",
  "states": [
    {
      "name": "Unprovisioned",
      "description": "Device in factory reset state",
      "type": "initial"
    }
  ],
  "transitions": [
    {
      "from_state": "Unprovisioned",
      "to_state": "Discovering",
      "trigger": "commission_start",
      "condition": "device ready",
      "action": "start advertising"
    }
  ],
  "events": [
    {
      "name": "commission_start",
      "description": "User initiates commissioning"
    }
  ]
}
```

### Visual Outputs

- **Mermaid diagrams** for web/markdown integration
- **DOT files** for Graphviz rendering
- **Analysis reports** in markdown format

## Key Features

### 1. Protocol-Specific Knowledge

- Pre-defined Matter protocol patterns
- Commissioning, connection, security, command processing
- Error handling and recovery mechanisms

### 2. Comprehensive Validation

- State reachability analysis
- Transition completeness checking
- Initial state validation
- Unreachable state detection

### 3. Pattern Analysis

- Cross-FSM common state identification
- Event frequency analysis
- Complexity metrics calculation
- Comprehensive FSM generation

### 4. Multiple Export Formats

- JSON for programmatic use
- Mermaid for documentation
- DOT for advanced visualization
- Markdown reports for analysis

## Comparison with RAG Implementation

### Basic RAG (`demo/rag_implementation.py`)

- General Q&A about Matter specification
- Open-ended responses
- No structured output
- Interactive chat interface

### FSM Generator (This Implementation)

- **Structured FSM extraction**
- **JSON output format**
- **Validation and analysis**
- **Following ProtocolGPT methodology**
- **Protocol-specific prompts**

## Dependencies

```bash
pip install langchain-ollama langchain-community faiss-cpu
```

## File Structure

```
FSM_Generator/
├── main.py                    # Main FSM extraction system
├── fsm_utils.py              # Parsing, validation, visualization
├── fsm_analyzer.py           # Advanced analysis and reporting
├── Matter_Specification.html # Source document
├── fsm_results/              # Generated JSON results
├── fsm_visualizations/       # Generated diagrams
└── README.md                 # This file
```

## Example Workflow

1. **Extract FSMs**: Run `python main.py` and select "all" to extract all predefined FSMs
2. **Analyze Results**: Run `python fsm_analyzer.py` to validate and analyze
3. **Review Outputs**:
   - `fsm_results/*.json` - Individual FSM extractions
   - `fsm_visualizations/*.md` - Mermaid diagrams
   - `fsm_visualizations/*.dot` - Graphviz files
   - `matter_fsm_analysis_report.md` - Comprehensive analysis

## Integration Notes

This implementation maintains the core ProtocolGPT approach:

- **Conversational retrieval chain** for context-aware extraction
- **Strong prompting** with protocol-specific knowledge
- **Structured output** in standardized format
- **Validation pipeline** for quality assurance

The key adaptation is moving from code analysis to specification analysis while preserving the methodology's effectiveness for FSM extraction.
