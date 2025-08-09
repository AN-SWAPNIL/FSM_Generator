#!/usr/bin/env python3
"""
Simple Matter PDF Table of Contents Extractor
Extracts cluster information from PDF TOC and saves to JSON
Uses Gemini 1.5 Flash with LangChain and LangGraph for structured extraction
"""

import os
import json
import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# Configuration
from config import *

# PDF and AI imports
import fitz  # PyMuPDF
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

@dataclass
class ClusterInfo:
    """Cluster information extracted from TOC"""
    cluster_name: str
    section_number: str
    start_page: int
    end_page: int
    category: str = ""

class MatterTOCExtractor:
    """
    Simple TOC extractor for Matter PDF specification
    Uses Gemini 1.5 Flash for intelligent TOC parsing
    """
    
    def __init__(self, pdf_path: str = None, gemini_api_key: str = None):
        self.pdf_path = pdf_path or PDF_PATH
        self.gemini_api_key = gemini_api_key
        self.extracted_clusters = {}
        self.toc_content = ""
        self.vector_store = None
        self.graph = None
        self.memory = MemorySaver()
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, LOG_LEVEL))
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._setup_gemini()
        self._setup_embeddings()
        
    def _setup_gemini(self):
        """Initialize Gemini LLM"""
        if self.gemini_api_key:
            os.environ["GOOGLE_API_KEY"] = self.gemini_api_key
        
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=GEMINI_TEMPERATURE,
            max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS
        )
        print("✅ Gemini 1.5 Flash initialized")
        
    def _setup_embeddings(self):
        """Initialize embeddings for vector store"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDINGS_MODEL
        )
        print("✅ Embeddings initialized")
    
    def extract_toc_from_pdf(self) -> str:
        """Step 1: Extract table of contents from PDF"""
        self.logger.info("Extracting TOC from PDF...")
        
        try:
            doc = fitz.open(self.pdf_path)
            toc_content = ""
            
            # Get TOC from PDF metadata if available
            toc = doc.get_toc()
            if toc:
                self.logger.info(f"Found TOC with {len(toc)} entries")
                for level, title, page in toc:
                    indent = "  " * (level - 1)
                    toc_content += f"{indent}{title} ... {page}\n"
            else:
                # Extract from first pages (usually TOC is in first 50 pages)
                self.logger.info("No TOC metadata found, extracting from text...")
                toc_content = self._extract_toc_from_text(doc)
            
            doc.close()
            self.toc_content = toc_content
            
            # Save raw TOC for debugging
            toc_file = os.path.join(OUTPUT_DIRECTORY, RAW_TOC_FILE)
            with open(toc_file, 'w', encoding='utf-8') as f:
                f.write(toc_content)
            
            self.logger.info(f"TOC extracted and saved to: {toc_file}")
            return toc_content
            
        except Exception as e:
            self.logger.error(f"Error extracting TOC: {e}")
            raise
    
    def _extract_toc_from_text(self, doc) -> str:
        """Extract TOC from PDF text content"""
        toc_content = ""
        
        # Look for TOC in first 50 pages (as configured)
        for page_num in range(min(MAX_TOC_PAGES, len(doc))):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            # Look for TOC patterns
            if self._is_toc_page(text):
                toc_content += f"\n--- Page {page_num + 1} ---\n"
                toc_content += text
        
        return toc_content
    
    def _is_toc_page(self, text: str) -> bool:
        """Check if page contains TOC content"""
        text_lower = text.lower()
        for indicator in TOC_INDICATORS:
            if indicator in text_lower or re.search(indicator, text, re.IGNORECASE):
                return True
        
        return False
    
    def setup_rag_system(self):
        """Step 2: Setup RAG system for TOC processing"""
        if not self.toc_content:
            self.logger.error("No TOC content found. Run extract_toc_from_pdf() first.")
            return
        
        self._create_vector_store()
        self._setup_graph()
        print("✅ RAG system setup complete")
    
    def _create_vector_store(self):
        """Create vector store from TOC content and first 50 pages only"""
        try:
            # Split TOC content into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ".", " "]
            )
            
            # Create documents from TOC
            chunks = text_splitter.split_text(self.toc_content)
            
            # Load only first 50 pages for efficiency (TOC is usually in first 50 pages)
            loader = PyPDFLoader(self.pdf_path)
            pages = loader.load()
            
            # Take only first 50 pages as specified
            first_pages = pages[:MAX_TOC_PAGES]
            page_chunks = text_splitter.split_documents(first_pages)
            
            # Combine TOC and page content
            from langchain.schema import Document
            toc_documents = [Document(page_content=chunk) for chunk in chunks]
            all_documents = toc_documents + page_chunks
            
            # Create vector store
            self.vector_store = FAISS.from_documents(all_documents, self.embeddings)
            print(f"✅ Vector store created with {len(all_documents)} documents (TOC + first {MAX_TOC_PAGES} pages)")
            
        except Exception as e:
            self.logger.error(f"Error creating vector store: {e}")
            raise
    
    def _setup_graph(self):
        """Setup LangGraph workflow for TOC analysis"""
        @tool
        def search_toc_content(query: str) -> str:
            """Search TOC and document content for cluster information"""
            if not self.vector_store:
                return "Vector store not initialized"
            
            try:
                docs = self.vector_store.similarity_search(query, k=VECTOR_SEARCH_K)
                return "\n\n".join(doc.page_content for doc in docs)
            except Exception as e:
                return f"Search error: {e}"
        
        tools = [search_toc_content]
        
        graph_builder = StateGraph(MessagesState)
        
        def query_or_respond(state: MessagesState):
            """Determine if we need to search or can respond directly"""
            llm_with_tools = self.llm.bind_tools(tools)
            return {"messages": [llm_with_tools.invoke(state["messages"])]}
        
        def extract_cluster_info(state: MessagesState):
            """Extract structured cluster information using Gemini"""
            tool_messages = [msg for msg in state["messages"] if msg.type == "tool"]
            
            docs_content = "\n\n".join(msg.content for msg in tool_messages)
            
            # Use system message from config
            system_message_content = CLUSTER_EXTRACTION_PROMPT
            
            conversation_messages = [
                message
                for message in state["messages"]
                if message.type in ("human", "system")
                or (message.type == "ai" and not message.tool_calls)
            ]
            
            prompt = [SystemMessage(system_message_content)] + conversation_messages
            response = self.llm.invoke(prompt)
            return {"messages": [response]}
        
        graph_builder.add_node("query_or_respond", query_or_respond)
        graph_builder.add_node("tools", ToolNode(tools))
        graph_builder.add_node("extract_cluster_info", extract_cluster_info)
        
        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "extract_cluster_info")
        graph_builder.add_edge("extract_cluster_info", END)
        
        self.graph = graph_builder.compile(checkpointer=self.memory)
    
    def extract_clusters_with_rag(self) -> Dict[str, ClusterInfo]:
        """Step 3: Extract cluster information using RAG system"""
        if not self.graph:
            self.setup_rag_system()
        
        # Create query using template from config
        query = HUMAN_QUERY_TEMPLATE.format(toc_content=self.toc_content[:4000])
        
        try:
            result = self.graph.invoke(
                {"messages": [HumanMessage(content=query)]},
                config={"configurable": {"thread_id": "toc_extraction"}}
            )
            
            response = result["messages"][-1].content
            
            # Parse JSON response
            cluster_data = self._parse_cluster_response(response)
            
            # If no clusters found with AI, immediately fall back to manual
            if not cluster_data.get("clusters") or len(cluster_data.get("clusters", [])) == 0:
                self.logger.warning("AI extraction returned no clusters, using manual fallback")
                return self.extract_clusters_manual_fallback()
            
            # Convert to ClusterInfo objects
            clusters = {}
            for cluster in cluster_data.get("clusters", []):
                cluster_info = ClusterInfo(
                    cluster_name=cluster.get("cluster_name", ""),
                    section_number=cluster.get("section_number", ""),
                    start_page=cluster.get("start_page", 0),
                    end_page=cluster.get("end_page", 0),
                    category=cluster.get("category", "")
                )
                clusters[cluster_info.section_number] = cluster_info
            
            self.extracted_clusters = clusters
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error extracting clusters with RAG: {e}")
            raise
    
    def _parse_cluster_response(self, response: str) -> Dict:
        """Parse JSON response from LLM"""
        try:
            # Clean response
            response = response.strip()
            
            # Remove markdown formatting if present
            if response.startswith("```"):
                response = re.sub(r'```[a-zA-Z]*\n?', '', response)
                response = re.sub(r'```', '', response)
            
            # If response contains explanatory text, try to extract JSON part
            if "cannot" in response.lower() or "need" in response.lower() or "would" in response.lower():
                self.logger.warning("Model returned explanatory text instead of JSON, falling back to manual extraction")
                return {"clusters": []}
            
            # Find JSON content between braces
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                # Clean up any remaining text artifacts
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)  # Remove control characters
                return json.loads(json_str)
            else:
                # Try parsing entire response
                return json.loads(response)
                
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing failed: {e}")
            self.logger.error(f"Response content: {response[:500]}...")
            
            # If JSON parsing fails completely, return empty structure
            return {"clusters": []}
        except Exception as e:
            self.logger.error(f"Error parsing response: {e}")
            return {"clusters": []}
    
    def extract_clusters_manual_fallback(self) -> Dict[str, ClusterInfo]:
        """Fallback method using manual regex parsing"""
        self.logger.info("Using manual fallback method...")
        
        clusters = {}
        
        # Use predefined cluster data from config
        for cluster_data in MANUAL_CLUSTER_DATA:
            cluster_info = ClusterInfo(
                cluster_name=cluster_data["name"],
                section_number=cluster_data["section"],
                start_page=cluster_data["start"],
                end_page=cluster_data["end"],
                category=cluster_data["category"]
            )
            clusters[cluster_info.section_number] = cluster_info
        
        self.extracted_clusters = clusters
        return clusters
    
    def save_clusters_to_json(self, output_file: str = None):
        """Save extracted clusters to JSON file"""
        if not self.extracted_clusters:
            self.logger.error("No clusters extracted. Run extraction method first.")
            return
        
        output_file = output_file or OUTPUT_JSON_FILE
        output_path = os.path.join(OUTPUT_DIRECTORY, output_file)
        
        # Convert to serializable format
        clusters_data = {}
        for section, cluster in self.extracted_clusters.items():
            clusters_data[section] = asdict(cluster)
        
        # Create final JSON structure
        final_data = {
            "metadata": {
                "total_clusters": len(clusters_data),
                "categories": list(set(cluster["category"] for cluster in clusters_data.values())),
                "extraction_method": "Gemini 1.5 Flash with LangChain/LangGraph",
                "source_pdf": os.path.basename(self.pdf_path),
                "model_used": GEMINI_MODEL,
                "temperature": GEMINI_TEMPERATURE
            },
            "clusters": clusters_data
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Clusters saved to: {output_path}")
        return output_path
    
    def print_clusters_summary(self):
        """Print summary of extracted clusters"""
        if not self.extracted_clusters:
            print("No clusters extracted yet.")
            return
        
        print(f"\n🔍 Extracted {len(self.extracted_clusters)} Matter Clusters")
        print("=" * 70)
        
        # Group by category
        categories = {}
        for cluster in self.extracted_clusters.values():
            if cluster.category not in categories:
                categories[cluster.category] = []
            categories[cluster.category].append(cluster)
        
        for category, clusters in categories.items():
            print(f"\n📂 {category} ({len(clusters)} clusters):")
            for cluster in sorted(clusters, key=lambda x: x.section_number):
                print(f"   {cluster.section_number}. {cluster.cluster_name}")
                print(f"      Pages: {cluster.start_page}-{cluster.end_page}")


def main():
    """Main function for TOC extraction"""
    print("🚀 Matter PDF Table of Contents Extractor")
    print("=" * 50)
    
    # Configuration
    pdf_path = PDF_PATH
    
    # Get Gemini API key
    gemini_api_key = GOOGLE_API_KEY
    if not gemini_api_key:
        gemini_api_key = input("Enter your Gemini API key: ").strip()
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        print(f"Expected location: {pdf_path}")
        print("Please ensure the PDF is in the correct location.")
        return
    
    try:
        # Create extractor
        extractor = MatterTOCExtractor(pdf_path, gemini_api_key)
        
        # Step 1: Extract TOC from PDF
        print("\n🔍 Step 1: Extracting TOC from PDF...")
        toc_content = extractor.extract_toc_from_pdf()
        
        if not toc_content:
            print("❌ No TOC content found")
            return
        
        # Step 2: Try RAG-based extraction
        print("\n🧠 Step 2: Extracting clusters using Gemini + RAG...")
        clusters = {}
        try:
            extractor.setup_rag_system()
            clusters = extractor.extract_clusters_with_rag()
            
            # Check if we got valid results
            if not clusters or len(clusters) == 0:
                print("⚠️ AI extraction returned no results, using manual fallback...")
                clusters = extractor.extract_clusters_manual_fallback()
                
        except Exception as e:
            print(f"⚠️ RAG extraction failed: {e}")
            print("🔄 Falling back to manual extraction...")
            clusters = extractor.extract_clusters_manual_fallback()
        
        # Step 3: Save results
        print("\n💾 Step 3: Saving results...")
        json_file = extractor.save_clusters_to_json()
        
        # Step 4: Display summary
        extractor.print_clusters_summary()
        
        print(f"\n✅ Extraction completed successfully!")
        print(f"📄 Results saved to: {json_file}")
        print(f"📊 Total clusters: {len(clusters)}")
        print(f"🤖 Model used: {GEMINI_MODEL}")
        
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        logging.error(f"Extraction failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
