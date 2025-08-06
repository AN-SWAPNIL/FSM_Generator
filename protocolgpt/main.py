#!/usr/bin/env python3
"""
FSM Generator from Matter Specification
Component-wise implementation following ProtocolGPT's proven methodology
Modular architecture with separate LLM, retriever, and extractor components
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Component imports
from matter_clusters import MatterClusters, PROTOCOLGPT_CONFIG
from protocolgpt_llm import create_llm
from protocolgpt_retriever import create_retriever
from protocolgpt_extractor import create_extractor
from protocolgpt_prompts import get_query_types

class MatterProtocolGPTApp:
    """
    Main application class following ProtocolGPT's component-wise architecture
    Orchestrates LLM, retriever, and extractor components
    """
    
    def __init__(self, spec_path: str, model_name: str = "llama3.1", config: Optional[Dict] = None):
        self.spec_path = spec_path
        self.model_name = model_name
        self.config = config or PROTOCOLGPT_CONFIG.copy()
        
        # Component instances
        self.llm = None
        self.retriever = None
        self.extractor = None
        
        # Matter clusters
        self.matter_clusters = MatterClusters.get_all_clusters()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize application
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components following ProtocolGPT pattern"""
        print("🚀 Initializing Matter Cluster ProtocolGPT FSM Extractor")
        print("="*60)
        
        # Check if specification exists
        if not Path(self.spec_path).exists():
            raise FileNotFoundError(f"Matter specification not found at: {self.spec_path}")
        
        try:
            # Initialize LLM
            print("\n🤖 Step 1: Initializing Language Model")
            self.llm = create_llm(self.model_name, self.config)
            
            # Test LLM connection
            if not self.llm.test_connection():
                raise RuntimeError("LLM connection test failed")
            
            # Initialize retriever
            print("\n📚 Step 2: Initializing Document Retriever")
            self.retriever = create_retriever(self.config)
            
            # Load specification
            print("\n📄 Step 3: Loading Matter Specification")
            self.retriever.load_specification(self.spec_path)
            
            # Initialize extractor
            print("\n🔧 Step 4: Initializing FSM Extractor")
            self.extractor = create_extractor(self.llm, self.retriever)
            
            print("\n✅ All components initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error during initialization: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure Ollama is running: ollama serve")
            print(f"2. Make sure the model is available: ollama pull {self.model_name}")
            print("3. Check if all dependencies are installed")
            print("4. Ensure Matter_Specification.html exists in the current directory")
            raise
    
    def print_startup_info(self):
        """Print startup information and available options"""
        print(f"\n📋 Available clusters ({len(self.matter_clusters)}):")
        for i, (cluster_key, cluster) in enumerate(self.matter_clusters.items(), 1):
            print(f"  {i}. {cluster.cluster_name} ({cluster.cluster_id}) - {cluster.category}")
        
        query_types = get_query_types()
        print(f"\n🔧 Available query types:")
        for i, query_type in enumerate(query_types, 1):
            print(f"  {i}. {query_type}")
        
        print("\n" + "="*60)
        print("🚀 Matter Cluster ProtocolGPT FSM Extraction Assistant")
        print("="*60)
        print("Extract FSMs using ProtocolGPT's proven methodology!")
        print("Commands:")
        print("  - 'cluster_name query_type' (e.g., 'on_off complete_fsm')")
        print("  - 'all cluster_name' to extract all components for a cluster")
        print("  - 'chat cluster_name' for interactive analysis")
        print("  - 'batch' to extract complete FSMs for all clusters")
        print("  - 'status' to show system status")
        print("  - 'help' to show this help")
        print("  - 'quit', 'exit' to exit")
        print("="*60)
    
    def run_interactive_session(self):
        """Run interactive command session"""
        self.print_startup_info()
        
        while True:
            try:
                user_input = input("\n🤖 Enter command: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'help':
                    self.print_startup_info()
                elif user_input.lower() == 'status':
                    self._show_status()
                elif user_input.lower() == 'batch':
                    self._handle_batch_processing()
                else:
                    self._handle_user_command(user_input)
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _handle_user_command(self, user_input: str):
        """Handle user command parsing and execution"""
        parts = user_input.split()
        query_types = get_query_types()
        
        if len(parts) == 2:
            if parts[0] == 'all':
                cluster_name = parts[1]
                if cluster_name in self.matter_clusters:
                    result = self.extractor.extract_all_cluster_components(cluster_name)
                    self._save_result(result, f"{cluster_name}_all_components.json")
                else:
                    print(f"❌ Unknown cluster: {cluster_name}")
                    print("Available clusters:", list(self.matter_clusters.keys()))
            
            elif parts[0] == 'chat':
                cluster_name = parts[1]
                if cluster_name in self.matter_clusters:
                    self.extractor.interactive_cluster_chat(cluster_name)
                else:
                    print(f"❌ Unknown cluster: {cluster_name}")
                    print("Available clusters:", list(self.matter_clusters.keys()))
            
            else:
                cluster_name, query_type = parts
                if cluster_name in self.matter_clusters and query_type in query_types:
                    result = self.extractor.extract_cluster_fsm(cluster_name, query_type)
                    self._save_result(result, f"{cluster_name}_{query_type}.json")
                else:
                    print(f"❌ Invalid cluster or query type")
                    print("Available clusters:", list(self.matter_clusters.keys()))
                    print("Available query types:", query_types)
        else:
            print("❌ Invalid command format")
            print("Use: 'cluster_name query_type', 'all cluster_name', 'chat cluster_name', or 'batch'")
    
    def _handle_batch_processing(self):
        """Handle batch processing of all clusters"""
        results = self.extractor.batch_extract_all_clusters("complete_fsm")
        
        # Save individual results
        for result in results:
            if "error" not in result:
                cluster_name = result["cluster_name"]
                self._save_result(result, f"{cluster_name}_complete_fsm.json")
        
        # Save batch summary
        self.extractor.save_results()
    
    def _save_result(self, result: Dict[str, Any], filename: str):
        """Save individual result to file"""
        os.makedirs("protocolgpt_fsm_results", exist_ok=True)
        filepath = f"protocolgpt_fsm_results/{filename}"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved: {filepath}")
    
    def _show_status(self):
        """Show system status"""
        print("\n📊 System Status")
        print("=" * 40)
        
        # LLM status
        llm_info = self.llm.get_model_info()
        print(f"🤖 LLM: {llm_info['model_name']} ({llm_info['status']})")
        print(f"   Temperature: {llm_info['temperature']}")
        print(f"   Max tokens: {llm_info['max_tokens']}")
        
        # Retriever status
        store_info = self.retriever.get_store_info()
        print(f"📚 Vector Store: {store_info['status']}")
        if store_info['status'] == 'initialized':
            print(f"   Index size: {store_info.get('index_size', 'unknown')}")
            print(f"   Chunk size: {store_info['config']['chunk_size']}")
        
        # Extraction summary
        summary = self.extractor.get_extraction_summary()
        if 'total_extractions' in summary:
            print(f"🔧 Extractions performed: {summary['total_extractions']}")
            print(f"   Clusters analyzed: {len(summary['clusters_analyzed'])}")
            print(f"   Successful: {summary['successful_extractions']}")
            print(f"   Failed: {summary['failed_extractions']}")
        else:
            print("🔧 No extractions performed yet")
    
    def extract_single_cluster(self, cluster_name: str, query_type: str = "complete_fsm") -> Dict[str, Any]:
        """
        Extract FSM for a single cluster (programmatic interface)
        
        Args:
            cluster_name: Name of the cluster
            query_type: Type of extraction
        
        Returns:
            Extraction result dictionary
        """
        return self.extractor.extract_cluster_fsm(cluster_name, query_type)
    
    def extract_all_clusters(self, query_type: str = "complete_fsm") -> List[Dict[str, Any]]:
        """
        Extract FSMs for all clusters (programmatic interface)
        
        Args:
            query_type: Type of extraction
        
        Returns:
            List of extraction results
        """
        return self.extractor.batch_extract_all_clusters(query_type)

def main():
    """Main function following ProtocolGPT pattern"""
    spec_path = os.path.join(os.path.dirname(__file__), "Matter_Specification.html")
    
    if not os.path.exists(spec_path):
        print(f"❌ Matter specification not found at: {spec_path}")
        print("Please copy Matter_Specification.html to this directory")
        return 1
    
    try:
        # Create and run application
        app = MatterProtocolGPTApp(spec_path, model_name="llama3.1")
        app.run_interactive_session()
        
        # Save final results
        app.extractor.save_results()
        
        return 0
        
    except Exception as e:
        print(f"❌ Application error: {e}")
        return 1

def create_app(spec_path: str, model_name: str = "llama3.1", config: Optional[Dict] = None) -> MatterProtocolGPTApp:
    """
    Factory function to create application instance
    
    Args:
        spec_path: Path to Matter specification
        model_name: LLM model name
        config: Optional configuration
    
    Returns:
        Initialized MatterProtocolGPTApp instance
    """
    return MatterProtocolGPTApp(spec_path, model_name, config)

if __name__ == "__main__":
    sys.exit(main())
