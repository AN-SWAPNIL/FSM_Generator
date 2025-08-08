#!/usr/bin/env python3
"""
Matter Cluster PDF Extractor & FSM Generator
Combines PDF cluster extraction with Les modeling framework FSM generation
Automatically extracts cluster names from PDF and generates formal FSM models
Following Les Modeling Framework from USENIX Security Papers
"""

import os
import re
import logging
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# PDF processing imports
import PyPDF2
import fitz  # PyMuPDF for better text extraction
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

@dataclass
class MatterCluster:
    """Simplified Matter cluster information"""
    cluster_id: str
    cluster_name: str
    description: str = ""
    page_number: int = 0

@dataclass
class MatterClusterInfo:
    """Extended cluster information for FSM generation"""
    cluster_id: str
    cluster_name: str
    category: str
    description: str
    attributes: List[str]
    commands: List[str]
    events: List[str]
    
class PDFClusterExtractorAndFSMGenerator:
    """Extract Matter clusters from PDF specification"""
    
    def __init__(self, pdf_path: str, model_name: str = "llama3.1"):
        self.pdf_path = pdf_path
        self.model_name = model_name
        self.clusters = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM
        self.llm = ChatOllama(
            model=self.model_name,
            temperature=0.1,
            num_predict=1000
        )
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
    def extract_clusters_from_pdf(self) -> List[MatterCluster]:
        """Extract cluster information from PDF using multiple methods"""
        self.logger.info(f"Extracting clusters from: {self.pdf_path}")
        
        # Method 1: Extract using PyMuPDF (better text extraction)
        clusters_pymupdf = self._extract_with_pymupdf()
        
        # Method 2: Extract using regex patterns
        clusters_regex = self._extract_with_regex_patterns()
        
        # Method 3: Extract using LangChain + LLM
        clusters_llm = self._extract_with_llm()
        
        # Combine and deduplicate results
        all_clusters = clusters_pymupdf + clusters_regex + clusters_llm
        self.clusters = self._deduplicate_clusters(all_clusters)
        
        self.logger.info(f"Extracted {len(self.clusters)} unique clusters")
        return self.clusters
    
    def _extract_with_pymupdf(self) -> List[MatterCluster]:
        """Extract clusters using PyMuPDF for better text extraction"""
        clusters = []
        
        try:
            doc = fitz.open(self.pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # Look for cluster patterns in text
                cluster_matches = self._find_cluster_patterns(text, page_num + 1)
                clusters.extend(cluster_matches)
                
            doc.close()
            
        except Exception as e:
            self.logger.error(f"Error with PyMuPDF extraction: {e}")
            
        return clusters
    
    def _extract_with_regex_patterns(self) -> List[MatterCluster]:
        """Extract clusters using regex patterns"""
        clusters = []
        
        try:
            # Use PyPDF2 as fallback
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    cluster_matches = self._find_cluster_patterns(text, page_num + 1)
                    clusters.extend(cluster_matches)
                    
        except Exception as e:
            self.logger.error(f"Error with regex extraction: {e}")
            
        return clusters
    
    def _find_cluster_patterns(self, text: str, page_num: int) -> List[MatterCluster]:
        """Find cluster patterns in text using regex"""
        clusters = []
        
        # Enhanced regex patterns for Matter clusters
        patterns = [
            # Pattern 1: "Cluster ID: 0x0006, Name: On/Off"
            r'Cluster\s+ID:\s*0x([0-9A-Fa-f]+).*?Name:\s*([^\n,]+)',
            
            # Pattern 2: "0x0006 On/Off Cluster"
            r'0x([0-9A-Fa-f]+)\s+([A-Z][^0-9\n]*?)\s+Cluster',
            
            # Pattern 3: Table headers with cluster info
            r'([0-9A-Fa-f]{4})\s+([A-Z][A-Za-z\s/]+?)\s+(?:Cluster|Server|Client)',
            
            # Pattern 4: Cluster definitions
            r'(?:^|\n)\s*([0-9A-Fa-f]{4})\s+([A-Z][A-Za-z\s&/\-]+?)(?:\s+[0-9]|\n)',
            
            # Pattern 5: Chapter/section headers
            r'(\d+\.\d+)\s+([A-Z][A-Za-z\s&/\-]+?)\s+Cluster',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                if len(match.groups()) >= 2:
                    cluster_id = match.group(1).upper()
                    cluster_name = match.group(2).strip()
                    
                    # Clean up cluster name
                    cluster_name = self._clean_cluster_name(cluster_name)
                    
                    # Validate cluster
                    if self._is_valid_cluster(cluster_id, cluster_name):
                        clusters.append(MatterCluster(
                            cluster_id=f"0x{cluster_id}",
                            cluster_name=cluster_name,
                            page_number=page_num
                        ))
        
        return clusters
    
    def _extract_with_llm(self) -> List[MatterCluster]:
        """Extract clusters using LLM for better understanding"""
        clusters = []
        
        try:
            # Load PDF with LangChain
            loader = PyPDFLoader(self.pdf_path)
            pages = loader.load()
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(pages)
            
            # Process chunks that likely contain cluster information
            for chunk in chunks[:20]:  # Limit to first 20 chunks for efficiency
                if self._chunk_contains_clusters(chunk.page_content):
                    extracted = self._extract_clusters_from_chunk(chunk.page_content)
                    clusters.extend(extracted)
                    
        except Exception as e:
            self.logger.error(f"Error with LLM extraction: {e}")
            
        return clusters
    
    def _chunk_contains_clusters(self, text: str) -> bool:
        """Check if chunk likely contains cluster information"""
        cluster_keywords = [
            'cluster', 'server', 'client', '0x', 'attribute', 'command'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in cluster_keywords)
    
    def _extract_clusters_from_chunk(self, text: str) -> List[MatterCluster]:
        """Extract clusters from text chunk using LLM"""
        prompt = f"""
Extract Matter cluster information from this text. 
Return ONLY cluster ID and name in this exact format:
CLUSTER: 0x0006, On/Off
CLUSTER: 0x0008, Level Control

Text to analyze:
{text[:1500]}

Focus on finding patterns like:
- Cluster ID numbers (0x0006, 0x0008, etc.)
- Cluster names (On/Off, Level Control, etc.)
- Table entries with cluster information

Output format: CLUSTER: <ID>, <Name>
"""
        
        try:
            response = self.llm.invoke(prompt)
            return self._parse_llm_response(response.content)
        except Exception as e:
            self.logger.error(f"LLM extraction error: {e}")
            return []
    
    def _parse_llm_response(self, response: str) -> List[MatterCluster]:
        """Parse LLM response to extract cluster information"""
        clusters = []
        
        lines = response.split('\n')
        for line in lines:
            if line.strip().startswith('CLUSTER:'):
                try:
                    # Parse: "CLUSTER: 0x0006, On/Off"
                    parts = line.replace('CLUSTER:', '').strip().split(',', 1)
                    if len(parts) == 2:
                        cluster_id = parts[0].strip()
                        cluster_name = parts[1].strip()
                        
                        if self._is_valid_cluster(cluster_id, cluster_name):
                            clusters.append(MatterCluster(
                                cluster_id=cluster_id,
                                cluster_name=cluster_name
                            ))
                except Exception:
                    continue
                    
        return clusters
    
    def _clean_cluster_name(self, name: str) -> str:
        """Clean up cluster name"""
        # Remove unwanted characters and patterns
        name = re.sub(r'\s+', ' ', name)  # Multiple spaces to single
        name = re.sub(r'[^\w\s&/\-()]', '', name)  # Keep only alphanumeric, space, &, /, -, ()
        name = name.strip()
        
        # Remove common suffixes
        suffixes_to_remove = ['Cluster', 'Server', 'Client', 'Rev', 'Version']
        for suffix in suffixes_to_remove:
            if name.endswith(suffix):
                name = name[:-len(suffix)].strip()
        
        return name
    
    def _is_valid_cluster(self, cluster_id: str, cluster_name: str) -> bool:
        """Validate if cluster ID and name are valid"""
        # Check cluster ID format
        if not re.match(r'^(0x)?[0-9A-Fa-f]{4}$', cluster_id.replace('0x', '')):
            return False
            
        # Check cluster name
        if len(cluster_name) < 3 or len(cluster_name) > 50:
            return False
            
        # Skip invalid names
        invalid_names = [
            'reserved', 'rfu', 'tbd', 'n/a', 'none', 'see', 'table', 'figure',
            'chapter', 'section', 'page', 'ref', 'spec', 'standard'
        ]
        
        name_lower = cluster_name.lower()
        if any(invalid in name_lower for invalid in invalid_names):
            return False
            
        return True
    
    def _deduplicate_clusters(self, clusters: List[MatterCluster]) -> List[MatterCluster]:
        """Remove duplicate clusters"""
        seen = set()
        unique_clusters = []
        
        for cluster in clusters:
            # Create a key for deduplication
            key = (cluster.cluster_id.upper(), cluster.cluster_name.lower())
            
            if key not in seen:
                seen.add(key)
                unique_clusters.append(cluster)
        
        # Sort by cluster ID
        unique_clusters.sort(key=lambda x: int(x.cluster_id.replace('0x', ''), 16))
        
        return unique_clusters
    
    def save_clusters_to_file(self, output_file: str = "extracted_clusters.txt"):
        """Save extracted clusters to file"""
        output_path = os.path.join(os.path.dirname(self.pdf_path), output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Matter Clusters Extracted from PDF\n")
            f.write("=" * 50 + "\n\n")
            
            for cluster in self.clusters:
                f.write(f"Cluster ID: {cluster.cluster_id}\n")
                f.write(f"Name: {cluster.cluster_name}\n")
                if cluster.page_number:
                    f.write(f"Page: {cluster.page_number}\n")
                if cluster.description:
                    f.write(f"Description: {cluster.description}\n")
                f.write("-" * 30 + "\n")
        
        self.logger.info(f"Clusters saved to: {output_path}")
        return output_path
    
    def get_cluster_dict(self) -> Dict[str, MatterCluster]:
        """Get clusters as dictionary with cluster_id as key"""
        return {cluster.cluster_id: cluster for cluster in self.clusters}
    
    def print_clusters_summary(self):
        """Print summary of extracted clusters"""
        print(f"\n🔍 Extracted {len(self.clusters)} Matter Clusters")
        print("=" * 60)
        
        for i, cluster in enumerate(self.clusters, 1):
            print(f"{i:2d}. {cluster.cluster_id} - {cluster.cluster_name}")
            if cluster.page_number:
                print(f"     (Page {cluster.page_number})")


def main():
    """Main function to extract clusters from PDF"""
    print("🚀 Matter Cluster PDF Extractor")
    print("=" * 40)
    
    # PDF file path
    pdf_path = os.path.join(os.path.dirname(__file__), "..", "Matter-1.4-Application-Cluster-Specification.pdf")
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    try:
        # Create extractor
        extractor = PDFClusterExtractor(pdf_path)
        
        # Extract clusters
        print("🔍 Extracting clusters from PDF...")
        clusters = extractor.extract_clusters_from_pdf()
        
        # Print summary
        extractor.print_clusters_summary()
        
        # Save to file
        output_file = extractor.save_clusters_to_file()
        print(f"\n💾 Results saved to: {output_file}")
        
        # Show cluster dictionary structure
        cluster_dict = extractor.get_cluster_dict()
        print(f"\n📊 Cluster Dictionary Keys: {list(cluster_dict.keys())[:10]}...")
        
        print("\n✅ Cluster extraction completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        logging.error(f"Extraction failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()

class MatterPDFClusterExtractor:
    """Extract Matter clusters from PDF and generate FSM models"""
    
    def __init__(self, pdf_path: str, model_name: str = "llama3.1"):
        self.pdf_path = pdf_path
        self.model_name = model_name
        self.extracted_clusters = {}
        self.memory = MemorySaver()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._setup_llm()
        self._setup_embeddings()
        self._load_and_process_pdf()
        self._setup_graph()
    
    def _setup_llm(self):
        """Setup language model"""
        print(f"🤖 Initializing {self.model_name}...")
        self.llm = ChatOllama(
            model=self.model_name,
            temperature=0,
            num_predict=2048
        )
        print("✅ Language model initialized")
    
    def _setup_embeddings(self):
        """Setup embeddings model"""
        print("🔤 Initializing embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("✅ Embeddings model initialized")
    
    def _load_and_process_pdf(self):
        """Load and process Matter PDF specification"""
        print(f"📄 Loading PDF document: {self.pdf_path}")
        
        if not Path(self.pdf_path).exists():
            raise FileNotFoundError(f"PDF document not found: {self.pdf_path}")
        
        # Load PDF
        loader = PyPDFLoader(self.pdf_path)
        docs = loader.load()
        
        if not docs:
            raise ValueError("No content loaded from PDF document")
        
        print(f"📖 Loaded {len(docs)} pages from PDF")
        
        # Store page content for analysis
        self.pdf_content = docs
        
        # Split text for vector store
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            length_function=len,
        )
        
        all_splits = text_splitter.split_documents(docs)
        print(f"✂️  Split into {len(all_splits)} chunks")
        
        print("🗃️  Creating vector store...")
        self.vector_store = FAISS.from_documents(all_splits, self.embeddings)
        print("✅ Vector store created and indexed")
    
    def _setup_graph(self):
        """Setup the LangGraph cluster extraction workflow"""
        print("🔗 Setting up cluster extraction workflow...")
        
        @tool(response_format="content_and_artifact")
        def retrieve_cluster_info(query: str):
            """Retrieve cluster-specific information from the Matter PDF specification."""
            retrieved_docs = self.vector_store.similarity_search(query, k=8)
            serialized = "\n\n".join(
                (f"Page: {doc.metadata.get('page', 'Unknown')}\n"
                 f"Content: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs
        
        self.retrieve_tool = retrieve_cluster_info
        
        graph_builder = StateGraph(MessagesState)
        
        def query_or_respond(state: MessagesState):
            """Generate tool call for retrieval or respond directly."""
            llm_with_tools = self.llm.bind_tools([retrieve_cluster_info])
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}
        
        tools = ToolNode([retrieve_cluster_info])
        
        def extract_clusters(state: MessagesState):
            """Extract cluster information using retrieved context."""
            recent_tool_messages = []
            for message in reversed(state["messages"]):
                if message.type == "tool":
                    recent_tool_messages.append(message)
                else:
                    break
            tool_messages = recent_tool_messages[::-1]
            
            docs_content = "\n\n".join(doc.content for doc in tool_messages)
            system_message_content = (
                "You are a Matter protocol expert. Extract cluster information from the PDF content.\n\n"
                "TASK: Find and list ALL Matter clusters mentioned in the provided content.\n\n"
                "OUTPUT FORMAT (JSON only, no explanations):\n"
                "{\n"
                '  "clusters": [\n'
                '    {\n'
                '      "cluster_id": "0x0006",\n'
                '      "cluster_name": "On/Off",\n'
                '      "page": 15,\n'
                '      "description": "Basic on/off functionality"\n'
                '    }\n'
                '  ]\n'
                "}\n\n"
                "RULES:\n"
                "- Extract cluster ID in hex format (0x****)\n"
                "- Extract exact cluster name from specification\n"
                "- Include page number if available\n"
                "- Add brief description if mentioned\n"
                "- Only include valid Matter clusters\n"
                "- Output valid JSON format only\n\n"
                f"PDF Content:\n{docs_content}"
            )
            
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
        graph_builder.add_node("tools", tools)
        graph_builder.add_node("extract_clusters", extract_clusters)
        
        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "extract_clusters")
        graph_builder.add_edge("extract_clusters", END)
        
        self.graph = graph_builder.compile(checkpointer=self.memory)
        print("✅ Cluster extraction workflow setup complete")
    
    def extract_all_clusters(self) -> Dict[str, MatterClusterBasic]:
        """Extract all clusters from the PDF using various search strategies"""
        print("\n🔍 Starting comprehensive cluster extraction...")
        
        # Different search queries to find all clusters
        search_queries = [
            "Matter cluster ID hex 0x list all clusters",
            "cluster identifier specification table clusters",
            "application clusters 0x0000 0x0100 0x0200 0x0300 0x0400",
            "device type clusters on/off level control color control",
            "HVAC clusters thermostat fan control pump",
            "measurement clusters temperature pressure humidity",
            "lighting clusters on off level color",
            "security clusters access control binding",
            "network clusters descriptor basic information",
            "door lock window covering occupancy sensing",
            "cluster definitions matter protocol specification"
        ]
        
        all_clusters = {}
        
        for i, query in enumerate(search_queries, 1):
            print(f"\n🔍 Search {i}/{len(search_queries)}: {query[:50]}...")
            
            try:
                config = {"configurable": {"thread_id": f"search_{i}"}}
                
                final_response = None
                for step in self.graph.stream(
                    {"messages": [{"role": "user", "content": query}]},
                    stream_mode="values",
                    config=config,
                ):
                    final_message = step["messages"][-1]
                    if hasattr(final_message, 'content') and final_message.content:
                        if final_message.type == "ai" and not getattr(final_message, 'tool_calls', None):
                            final_response = final_message.content
                
                if final_response:
                    clusters = self._parse_cluster_response(final_response)
                    for cluster_id, cluster_info in clusters.items():
                        if cluster_id not in all_clusters:
                            all_clusters[cluster_id] = cluster_info
                            print(f"  ✅ Found: {cluster_info.cluster_name} ({cluster_id})")
                        
            except Exception as e:
                print(f"  ❌ Error in search {i}: {e}")
                continue
        
        # Additional regex-based extraction from raw PDF content
        print(f"\n🔍 Performing regex-based extraction...")
        regex_clusters = self._extract_clusters_regex()
        for cluster_id, cluster_info in regex_clusters.items():
            if cluster_id not in all_clusters:
                all_clusters[cluster_id] = cluster_info
                print(f"  ✅ Found (regex): {cluster_info.cluster_name} ({cluster_id})")
        
        self.extracted_clusters = all_clusters
        print(f"\n✅ Total clusters found: {len(all_clusters)}")
        return all_clusters
    
    def _parse_cluster_response(self, response: str) -> Dict[str, MatterClusterBasic]:
        """Parse LLM response to extract cluster information"""
        clusters = {}
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                
                if "clusters" in data:
                    for cluster_data in data["clusters"]:
                        cluster_id = cluster_data.get("cluster_id", "").strip()
                        cluster_name = cluster_data.get("cluster_name", "").strip()
                        
                        if cluster_id and cluster_name:
                            # Normalize cluster ID format
                            if not cluster_id.startswith("0x"):
                                if cluster_id.isdigit():
                                    cluster_id = f"0x{int(cluster_id):04X}"
                                else:
                                    cluster_id = f"0x{cluster_id}"
                            
                            clusters[cluster_id] = MatterClusterBasic(
                                cluster_id=cluster_id,
                                cluster_name=cluster_name,
                                page_number=cluster_data.get("page"),
                                description=cluster_data.get("description", "").strip()
                            )
                            
        except json.JSONDecodeError:
            # Fallback: try to extract using regex patterns
            cluster_patterns = [
                r'(0x[0-9A-Fa-f]{4})\s*[-:]\s*([A-Za-z][A-Za-z\s/]+)',
                r'([A-Za-z][A-Za-z\s/]+)\s*\(0x([0-9A-Fa-f]{4})\)',
                r'Cluster\s+ID:\s*(0x[0-9A-Fa-f]{4})[,\s]*Name:\s*([A-Za-z][A-Za-z\s/]+)'
            ]
            
            for pattern in cluster_patterns:
                matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    if len(match) == 2:
                        if match[0].startswith('0x'):
                            cluster_id, cluster_name = match[0].strip(), match[1].strip()
                        else:
                            cluster_name, cluster_id = match[0].strip(), f"0x{match[1].strip()}"
                        
                        if cluster_id and cluster_name:
                            clusters[cluster_id] = MatterClusterBasic(
                                cluster_id=cluster_id,
                                cluster_name=cluster_name,
                                description=""
                            )
        
        return clusters
    
    def _extract_clusters_regex(self) -> Dict[str, MatterClusterBasic]:
        """Extract clusters using regex patterns from raw PDF content"""
        clusters = {}
        
        # Common patterns found in Matter specifications
        patterns = [
            # Pattern: "0x0006 On/Off"
            r'(0x[0-9A-Fa-f]{4})\s+([A-Za-z][A-Za-z\s/\-&]+(?:Cluster)?)',
            # Pattern: "On/Off (0x0006)"
            r'([A-Za-z][A-Za-z\s/\-&]+)\s*\(0x([0-9A-Fa-f]{4})\)',
            # Pattern: "Cluster ID: 0x0006, Name: On/Off"
            r'Cluster\s+ID:\s*(0x[0-9A-Fa-f]{4})[,\s]*(?:Name:\s*)?([A-Za-z][A-Za-z\s/\-&]+)',
            # Pattern: Table format with hex IDs
            r'(0x[0-9A-Fa-f]{4})\s*\|\s*([A-Za-z][A-Za-z\s/\-&]+)',
        ]
        
        for page_num, doc in enumerate(self.pdf_content):
            content = doc.page_content
            
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    if len(match) == 2:
                        if match[0].startswith('0x'):
                            cluster_id, cluster_name = match[0].strip(), match[1].strip()
                        else:
                            cluster_name, cluster_id = match[0].strip(), f"0x{match[1].strip()}"
                        
                        # Clean up cluster name
                        cluster_name = re.sub(r'\s+', ' ', cluster_name).strip()
                        cluster_name = cluster_name.replace('Cluster', '').strip()
                        
                        # Validate cluster ID format
                        if re.match(r'^0x[0-9A-Fa-f]{4}$', cluster_id) and len(cluster_name) > 2:
                            if cluster_id not in clusters:
                                clusters[cluster_id] = MatterClusterBasic(
                                    cluster_id=cluster_id,
                                    cluster_name=cluster_name,
                                    page_number=page_num + 1,
                                    description=""
                                )
        
        return clusters
    
    def get_cluster_details(self, cluster_id: str) -> Dict[str, Any]:
        """Get detailed information for a specific cluster"""
        if cluster_id not in self.extracted_clusters:
            return {"error": f"Cluster {cluster_id} not found"}
        
        cluster = self.extracted_clusters[cluster_id]
        query = f"Matter cluster {cluster.cluster_name} {cluster_id} attributes commands events detailed specification"
        
        print(f"\n🔍 Getting details for {cluster.cluster_name} ({cluster_id})...")
        
        try:
            config = {"configurable": {"thread_id": f"details_{cluster_id}"}}
            
            final_response = None
            for step in self.graph.stream(
                {"messages": [{"role": "user", "content": query}]},
                stream_mode="values",
                config=config,
            ):
                final_message = step["messages"][-1]
                if hasattr(final_message, 'content') and final_message.content:
                    if final_message.type == "ai" and not getattr(final_message, 'tool_calls', None):
                        final_response = final_message.content
            
            return {
                "cluster_id": cluster_id,
                "cluster_name": cluster.cluster_name,
                "page_number": cluster.page_number,
                "description": cluster.description,
                "detailed_info": final_response
            }
            
        except Exception as e:
            return {
                "cluster_id": cluster_id,
                "cluster_name": cluster.cluster_name,
                "error": str(e)
            }
    
    def save_clusters_to_file(self, filename: str = "extracted_clusters.json"):
        """Save extracted clusters to JSON file"""
        clusters_data = {
            "total_clusters": len(self.extracted_clusters),
            "extraction_source": self.pdf_path,
            "clusters": {
                cluster_id: {
                    "cluster_id": cluster.cluster_id,
                    "cluster_name": cluster.cluster_name,
                    "page_number": cluster.page_number,
                    "description": cluster.description
                }
                for cluster_id, cluster in self.extracted_clusters.items()
            }
        }
        
        os.makedirs("pdf_extraction_results", exist_ok=True)
        filepath = os.path.join("pdf_extraction_results", filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(clusters_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(self.extracted_clusters)} clusters to: {filepath}")
        return filepath

def main():
    """Main function for Matter PDF Cluster Extraction"""
    print("🚀 Initializing Matter PDF Cluster Extractor")
    print("="*60)
    
    pdf_path = r"D:\Term Files\l4-t1\Thesis\adnan_sir\Codes\FSM_Generator\Matter-1.4-Application-Cluster-Specification.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found at: {pdf_path}")
        print("Please check the file path")
        return
    
    try:
        extractor = MatterPDFClusterExtractor(
            pdf_path=pdf_path,
            model_name="llama3.1"
        )
        
        print("\n✅ Initialization complete!")
        print("\n" + "="*60)
        print("🚀 Matter PDF Cluster Extraction Assistant")
        print("="*60)
        print("Commands:")
        print("  - 'extract' - Extract all clusters from PDF")
        print("  - 'list' - List extracted clusters")
        print("  - 'details <cluster_id>' - Get detailed info for cluster")
        print("  - 'save' - Save results to JSON file")
        print("  - 'quit', 'exit' - Exit")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n🤖 Enter command: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                elif user_input.lower() == 'extract':
                    print("\n🚀 Extracting all clusters from PDF...")
                    clusters = extractor.extract_all_clusters()
                    
                    print(f"\n📋 Extracted {len(clusters)} clusters:")
                    for cluster_id, cluster in sorted(clusters.items()):
                        page_info = f" (Page {cluster.page_number})" if cluster.page_number else ""
                        print(f"  {cluster_id}: {cluster.cluster_name}{page_info}")
                
                elif user_input.lower() == 'list':
                    if extractor.extracted_clusters:
                        print(f"\n📋 Found {len(extractor.extracted_clusters)} clusters:")
                        for cluster_id, cluster in sorted(extractor.extracted_clusters.items()):
                            page_info = f" (Page {cluster.page_number})" if cluster.page_number else ""
                            desc_info = f" - {cluster.description[:50]}..." if cluster.description else ""
                            print(f"  {cluster_id}: {cluster.cluster_name}{page_info}{desc_info}")
                    else:
                        print("❌ No clusters extracted yet. Run 'extract' first.")
                
                elif user_input.lower().startswith('details '):
                    cluster_id = user_input[8:].strip()
                    if cluster_id:
                        details = extractor.get_cluster_details(cluster_id)
                        print(f"\n📄 Details for {cluster_id}:")
                        print(json.dumps(details, indent=2))
                    else:
                        print("❌ Please provide cluster ID (e.g., 'details 0x0006')")
                
                elif user_input.lower() == 'save':
                    if extractor.extracted_clusters:
                        filepath = extractor.save_clusters_to_file()
                        print(f"✅ Results saved to: {filepath}")
                    else:
                        print("❌ No clusters to save. Run 'extract' first.")
                
                else:
                    print("❌ Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ Error initializing extractor: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Make sure the model is available: ollama pull llama3.1")
        print("3. Install required dependencies: pip install langchain-community pypdf")
        print("4. Check if the PDF file exists and is readable")
        logging.exception("Initialization error")

if __name__ == "__main__":
    main()
