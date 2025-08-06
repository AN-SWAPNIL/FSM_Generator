"""
Example usage and test cases for FSM Generator RAG System
"""

import sys
import os

# Add the demo directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fsm_generator_rag import FSMGeneratorRAG
from fsm_utils import FSMParser, FSMValidator, FSMGenerator

def test_basic_fsm_generation():
    """Test basic FSM generation functionality"""
    matter_spec_path = "d:/Thesis_Matter_2025/Codes/demo/Matter_Specification.html"
    
    if not os.path.exists(matter_spec_path):
        print(f"❌ Matter specification not found at: {matter_spec_path}")
        return
    
    print("🧪 Testing Basic FSM Generation...")
    
    try:
        # Initialize the FSM generator
        fsm_generator = FSMGeneratorRAG(matter_spec_path)
        
        # Test queries
        test_queries = [
            "Generate FSM for device commissioning process",
            "Create state machine for connection management",
            "Extract FSM for security handshake procedure",
            "Generate command processing state machine"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Test {i}: {query}")
            print("-" * 50)
            
            response = fsm_generator.generate_fsm(query, f"test_{i}")
            print(f"Response length: {len(response)} characters")
            
            if "FSM_NAME:" in response:
                print("✅ FSM format detected")
            else:
                print("⚠️ FSM format not clearly detected")
                
            print("First 200 characters:")
            print(response[:200] + "..." if len(response) > 200 else response)
        
        print("\n✅ Basic FSM generation test completed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

def test_fsm_validation():
    """Test FSM parsing and validation"""
    print("\n🧪 Testing FSM Validation...")
    
    # Sample FSM for testing
    sample_fsm = """
FSM_NAME: Test Commissioning FSM
DESCRIPTION: Simple commissioning state machine for testing

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
- FROM: Pairing | EVENT: pairing_failed | TO: Discovering

EVENTS:
- commission_start: User initiates commissioning
- pairing_request: Commissioner requests pairing
- pairing_success: Pairing completed successfully
- pairing_failed: Pairing failed, retry needed
"""
    
    try:
        # Parse FSM
        fsm = FSMParser.parse_fsm_text(sample_fsm)
        print(f"✅ Parsed FSM: {fsm.name}")
        print(f"   States: {len(fsm.states)}")
        print(f"   Transitions: {len(fsm.transitions)}")
        print(f"   Events: {len(fsm.events)}")
        
        # Validate FSM
        is_valid, errors = FSMValidator.validate_fsm(fsm)
        
        if is_valid:
            print("✅ FSM validation passed")
        else:
            print("❌ FSM validation failed:")
            for error in errors:
                print(f"   - {error}")
        
        # Test export formats
        print("\n📤 Testing export formats...")
        
        json_export = FSMGenerator.to_json(fsm)
        print(f"✅ JSON export: {len(json_export)} characters")
        
        python_export = FSMGenerator.to_python_enum(fsm)
        print(f"✅ Python enum export: {len(python_export)} characters")
        
        mermaid_export = FSMGenerator.to_mermaid(fsm)
        print(f"✅ Mermaid export: {len(mermaid_export)} characters")
        
        print("\n✅ FSM validation test completed")
        
    except Exception as e:
        print(f"❌ Validation test failed: {e}")

def demo_interactive_session():
    """Demonstrate interactive FSM generation"""
    matter_spec_path = "d:/Thesis_Matter_2025/Codes/demo/Matter_Specification.html"
    
    if not os.path.exists(matter_spec_path):
        print(f"❌ Matter specification not found at: {matter_spec_path}")
        return
    
    print("\n🎯 Demo: Interactive FSM Generation")
    print("This will start the interactive session...")
    
    try:
        fsm_generator = FSMGeneratorRAG(matter_spec_path)
        
        # Run a single demo query instead of full interactive mode
        demo_query = "Generate FSM for Matter device commissioning with error handling"
        print(f"\n📝 Demo Query: {demo_query}")
        print("-" * 60)
        
        response = fsm_generator.generate_fsm(demo_query, "demo_session")
        print(f"\n📋 Generated FSM:\n{response}")
        
        print("\n✅ Demo completed successfully")
        print("💡 To run full interactive mode, use: fsm_generator.interactive_fsm_generation()")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

def main():
    """Run all tests and demos"""
    print("🚀 FSM Generator RAG System - Tests and Demos")
    print("=" * 60)
    
    # Check if required packages are installed
    try:
        import langchain_ollama
        import langgraph
        print("✅ Required packages found")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install: pip install langchain-ollama langgraph")
        return
    
    # Run tests
    test_basic_fsm_generation()
    test_fsm_validation()
    demo_interactive_session()
    
    print("\n🎉 All tests and demos completed!")
    print("\n📚 To use the FSM generator:")
    print("   from fsm_generator_rag import FSMGeneratorRAG")
    print("   fsm_gen = FSMGeneratorRAG('path/to/Matter_Specification.html')")
    print("   fsm_gen.interactive_fsm_generation()")

if __name__ == "__main__":
    main()
