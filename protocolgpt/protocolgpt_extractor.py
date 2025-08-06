"""
ProtocolGPT FSM Extractor
Main extraction logic following ProtocolGPT's conversational approach
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory

from matter_clusters import MatterClusters, MatterClusterInfo
from protocolgpt_llm import ProtocolGPTLLM
from protocolgpt_retriever import ProtocolGPTRetriever
from protocolgpt_prompts import ProtocolGPTPrompts, get_prompt_for_cluster_and_type, get_query_types

class ProtocolGPTExtractor:
    """
    Main FSM extractor following ProtocolGPT's proven conversational approach
    Combines LLM, retriever, and prompts for comprehensive FSM extraction
    """
    
    def __init__(self, llm: ProtocolGPTLLM, retriever: ProtocolGPTRetriever):
        self.llm = llm
        self.retriever = retriever
        self.matter_clusters = MatterClusters.get_all_clusters()
        
        # Results storage
        self.extraction_results = []
    
    def extract_cluster_fsm(self, cluster_name: str, query_type: str = "complete_fsm") -> Dict[str, Any]:
        """
        Extract FSM for specific cluster using ProtocolGPT's conversational approach
        
        Args:
            cluster_name: Name of the Matter cluster
            query_type: Type of extraction (states, messages, transitions, complete_fsm)
        
        Returns:
            Dictionary containing extraction results
        """
        if cluster_name not in self.matter_clusters:
            raise ValueError(f"Unknown cluster: {cluster_name}. Available: {list(self.matter_clusters.keys())}")
        
        cluster = self.matter_clusters[cluster_name]
        
        # Get the appropriate prompt
        query = get_prompt_for_cluster_and_type(cluster, query_type)
        
        print(f"\n🔍 Extracting {query_type} for {cluster.cluster_name} cluster")
        print("=" * 60)
        
        # Create conversational retrieval chain following ProtocolGPT
        memory = ConversationSummaryMemory(
            llm=self.llm.get_llm(),
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            self.llm.get_llm(),
            retriever=self.retriever.get_retriever(),
            memory=memory,
            return_source_documents=True,
            get_chat_history=lambda h: h,
        )
        
        print("🤔 Processing with conversational retrieval chain...")
        
        # Execute query
        result = qa_chain({"question": query})
        
        # Extract response
        answer = result.get("answer", "")
        source_docs = result.get("source_documents", [])
        
        print(f"\n🤖 FSM Analysis Generated")
        print("=" * 40)
        print(answer[:800] + "..." if len(answer) > 800 else answer)
        
        # Show source files
        if source_docs:
            file_paths = [doc.metadata.get("source", "Unknown") for doc in source_docs]
            print(f"\n📄 Source documents:")
            for file_path in set(file_paths):
                print(f"  - {Path(file_path).name}")
        
        # Create result object
        extraction_result = {
            "cluster_name": cluster_name,
            "cluster_id": cluster.cluster_id,
            "query_type": query_type,
            "answer": answer,
            "raw_response": result,
            "source_documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in source_docs
            ],
            "modeling_framework": "ProtocolGPT",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Store result
        self.extraction_results.append(extraction_result)
        
        return extraction_result
    
    def extract_all_cluster_components(self, cluster_name: str) -> Dict[str, Any]:
        """
        Extract all FSM components for a specific cluster
        
        Args:
            cluster_name: Name of the Matter cluster
        
        Returns:
            Dictionary containing all extraction results for the cluster
        """
        if cluster_name not in self.matter_clusters:
            raise ValueError(f"Unknown cluster: {cluster_name}")
        
        cluster = self.matter_clusters[cluster_name]
        results = {}
        
        query_types = get_query_types()
        
        print(f"\n🚀 Extracting all FSM components for {cluster.cluster_name} cluster...")
        
        for query_type in query_types:
            try:
                print(f"\n--- Processing {query_type} ---")
                result = self.extract_cluster_fsm(cluster_name, query_type)
                results[query_type] = result
                time.sleep(1)  # Brief pause between queries
            except Exception as e:
                print(f"❌ Error extracting {query_type}: {e}")
                results[query_type] = {"error": str(e)}
        
        return {
            "cluster_name": cluster_name,
            "cluster_id": cluster.cluster_id,
            "extraction_results": results,
            "summary": {
                "total_queries": len(query_types),
                "successful_extractions": len([r for r in results.values() if "error" not in r]),
                "failed_extractions": len([r for r in results.values() if "error" in r])
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def batch_extract_all_clusters(self, query_type: str = "complete_fsm") -> List[Dict[str, Any]]:
        """
        Extract FSMs for all clusters in batch
        
        Args:
            query_type: Type of extraction to perform for all clusters
        
        Returns:
            List of extraction results for all clusters
        """
        print(f"\n🚀 Batch processing: extracting {query_type} for all clusters...")
        
        batch_results = []
        
        for cluster_name in self.matter_clusters.keys():
            try:
                print(f"\n{'='*40}")
                print(f"Processing {cluster_name}")
                print(f"{'='*40}")
                
                result = self.extract_cluster_fsm(cluster_name, query_type)
                batch_results.append(result)
                
            except Exception as e:
                print(f"❌ Error processing {cluster_name}: {e}")
                error_result = {
                    "cluster_name": cluster_name,
                    "query_type": query_type,
                    "error": str(e),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                batch_results.append(error_result)
        
        return batch_results
    
    def interactive_cluster_chat(self, cluster_name: str):
        """
        Interactive chat session for cluster analysis
        Following ProtocolGPT's chat_loop pattern
        
        Args:
            cluster_name: Name of the Matter cluster to analyze
        """
        if cluster_name not in self.matter_clusters:
            raise ValueError(f"Unknown cluster: {cluster_name}")
        
        cluster = self.matter_clusters[cluster_name]
        
        print(f"\n🤖 Interactive Matter Cluster Analysis")
        print(f"📋 Cluster: {cluster.cluster_name} ({cluster.cluster_id})")
        print(f"📝 Description: {cluster.description}")
        print(f"🏷️  Category: {cluster.category}")
        print("=" * 60)
        print("Ask questions about this cluster's FSM, states, transitions, etc.")
        print("Type 'exit', 'quit', or 'q' to end the session.")
        print("=" * 60)
        
        # Create conversational chain
        memory = ConversationSummaryMemory(
            llm=self.llm.get_llm(),
            memory_key="chat_history", 
            return_messages=True,
            output_key="answer"
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            self.llm.get_llm(),
            retriever=self.retriever.get_retriever(),
            memory=memory,
            return_source_documents=True,
            get_chat_history=lambda h: h,
        )
        
        while True:
            try:
                query = input(f"\n👉 [{cluster.cluster_name}] ").strip()
                
                if not query:
                    print("🤖 Please enter a question about the cluster")
                    continue
                
                if query.lower() in ('exit', 'quit', 'q'):
                    break
                
                # Add cluster context to the query
                contextualized_query = ProtocolGPTPrompts.get_interactive_prompt(cluster, query)
                
                print("🤔 Analyzing...")
                result = qa_chain({"question": contextualized_query})
                
                # Show source documents
                source_docs = result.get("source_documents", [])
                if source_docs:
                    file_paths = [doc.metadata.get("source", "Unknown") for doc in source_docs]
                    print(f"\n📄 Source references:")
                    for file_path in set(file_paths):
                        print(f"  - {Path(file_path).name}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n👋 Chat session ended!")
    
    def save_results(self, output_dir: str = "protocolgpt_fsm_results"):
        """
        Save all extraction results to files
        
        Args:
            output_dir: Directory to save results
        """
        if not self.extraction_results:
            print("⚠️  No results to save")
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual results
        for result in self.extraction_results:
            cluster_name = result["cluster_name"]
            query_type = result["query_type"]
            filename = f"{output_dir}/{cluster_name}_{query_type}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved: {filename}")
        
        # Save summary
        summary = {
            "total_extractions": len(self.extraction_results),
            "clusters_analyzed": list(set(r["cluster_name"] for r in self.extraction_results)),
            "query_types_used": list(set(r["query_type"] for r in self.extraction_results)),
            "results": self.extraction_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        summary_file = f"{output_dir}/extraction_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved extraction summary: {summary_file}")
    
    def get_extraction_summary(self) -> Dict[str, Any]:
        """
        Get summary of all extractions performed
        
        Returns:
            Dictionary containing extraction summary
        """
        if not self.extraction_results:
            return {"message": "No extractions performed yet"}
        
        return {
            "total_extractions": len(self.extraction_results),
            "clusters_analyzed": list(set(r["cluster_name"] for r in self.extraction_results)),
            "query_types_used": list(set(r["query_type"] for r in self.extraction_results)),
            "successful_extractions": len([r for r in self.extraction_results if "error" not in r]),
            "failed_extractions": len([r for r in self.extraction_results if "error" in r])
        }

def create_extractor(llm: ProtocolGPTLLM, retriever: ProtocolGPTRetriever) -> ProtocolGPTExtractor:
    """
    Factory function to create extractor
    
    Args:
        llm: Initialized ProtocolGPTLLM instance
        retriever: Initialized ProtocolGPTRetriever instance
    
    Returns:
        ProtocolGPTExtractor instance
    """
    return ProtocolGPTExtractor(llm, retriever)
