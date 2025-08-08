#!/usr/bin/env python3
"""
Matter PDF Cluster Extractor with Les FSM Generator
Combines PDF cluster extraction with formal FSM model generation
Following Les Modeling Framework from USENIX Security Papers
"""

import os
import json
import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# PDF processing imports
import PyPDF2
import fitz  # PyMuPDF for better text extraction

# Core LangChain/LangGraph imports
from langchain_ollama.chat_models import ChatOllama
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
class MatterCluster:
    """Matter cluster information extracted from PDF"""
    cluster_id: str
    cluster_name: str
    description: str = ""
    page_number: int = 0
    category: str = "general"
    attributes: List[str] = None
    commands: List[str] = None
    events: List[str] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = []
        if self.commands is None:
            self.commands = []
        if self.events is None:
            self.events = []

class MatterPDFClusterFSMExtractor:
    """
    Combined PDF cluster extractor and Les FSM generator
    1. Extracts clusters from Matter PDF specification
    2. Generates formal FSM models following Les framework
    """
    
    def __init__(self, pdf_path: str, model_name: str = "llama3.1"):
        self.pdf_path = pdf_path
        self.model_name = model_name
        self.extracted_clusters = {}
        self.vector_store = None
        self.graph = None
        self.memory = MemorySaver()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._setup_llm()
        self._setup_embeddings()
        
    def _setup_llm(self):
        """Initialize LLM"""
        self.llm = ChatOllama(
            model=self.model_name,
            temperature=0.1,
            num_predict=2000
        )
        print(f"✅ LLM initialized: {self.model_name}")
        
    def _setup_embeddings(self):
        """Initialize embeddings"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("✅ Embeddings initialized")
    
    def extract_clusters_from_pdf(self) -> Dict[str, MatterCluster]:
        """Step 1: Extract cluster information from PDF using multiple methods"""
        self.logger.info(f"Extracting clusters from: {self.pdf_path}")
        
        # Method 1: Extract using PyMuPDF (better text extraction)
        clusters_pymupdf = self._extract_with_pymupdf()
        
        # Method 2: Extract using regex patterns
        clusters_regex = self._extract_with_regex_patterns()
        
        # Method 3: Extract using LangChain + LLM
        clusters_llm = self._extract_with_llm()
        
        # Combine and deduplicate results
        all_clusters = clusters_pymupdf + clusters_regex + clusters_llm
        unique_clusters = self._deduplicate_clusters(all_clusters)
        
        # Convert to dictionary for easy access
        self.extracted_clusters = {cluster.cluster_id: cluster for cluster in unique_clusters}
        
        self.logger.info(f"Extracted {len(self.extracted_clusters)} unique clusters")
        return self.extracted_clusters
    
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
                        category = self._determine_cluster_category(cluster_name)
                        clusters.append(MatterCluster(
                            cluster_id=f"0x{cluster_id}",
                            cluster_name=cluster_name,
                            page_number=page_num,
                            category=category
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
                            category = self._determine_cluster_category(cluster_name)
                            clusters.append(MatterCluster(
                                cluster_id=cluster_id,
                                cluster_name=cluster_name,
                                category=category
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
    
    def _determine_cluster_category(self, cluster_name: str) -> str:
        """Determine cluster category based on name"""
        name_lower = cluster_name.lower()
        
        if any(keyword in name_lower for keyword in ['light', 'color', 'level', 'on/off', 'dimmer']):
            return "lighting"
        elif any(keyword in name_lower for keyword in ['temp', 'thermal', 'hvac', 'fan', 'humid']):
            return "hvac"
        elif any(keyword in name_lower for keyword in ['door', 'lock', 'window', 'covering']):
            return "closure"
        elif any(keyword in name_lower for keyword in ['measure', 'sensor', 'detect', 'occup']):
            return "sensing"
        elif any(keyword in name_lower for keyword in ['security', 'access', 'auth']):
            return "security"
        else:
            return "general"
    
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
    
    def setup_fsm_generation(self):
        """Step 2: Setup FSM generation workflow after cluster extraction"""
        if not self.extracted_clusters:
            self.logger.error("No clusters extracted. Run extract_clusters_from_pdf() first.")
            return
            
        self._load_and_process_pdf_for_fsm()
        self._setup_fsm_graph()
        print("✅ FSM generation workflow setup complete")
    
    def _load_and_process_pdf_for_fsm(self):
        """Load PDF content for FSM generation"""
        try:
            loader = PyPDFLoader(self.pdf_path)
            documents = loader.load()
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            
            splits = text_splitter.split_documents(documents)
            print(f"✅ PDF processed into {len(splits)} chunks")
            
            # Create vector store
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
            print("✅ Vector store created")
            
        except Exception as e:
            self.logger.error(f"Error processing PDF for FSM: {e}")
            raise
    
    def _setup_fsm_graph(self):
        """Setup LangGraph workflow for FSM generation"""
        @tool
        def matter_spec_search(query: str) -> str:
            """Search Matter specification for cluster information"""
            if not self.vector_store:
                return "Vector store not initialized"
            
            try:
                docs = self.vector_store.similarity_search(query, k=5)
                return "\n\n".join(doc.page_content for doc in docs)
            except Exception as e:
                return f"Search error: {e}"
        
        tools = [matter_spec_search]
        
        graph_builder = StateGraph(MessagesState)
        
        def query_or_respond(state: MessagesState):
            """Determine if we need to search or can respond directly"""
            llm_with_tools = self.llm.bind_tools(tools)
            return {"messages": [llm_with_tools.invoke(state["messages"])]}
        
        def generate_les_model(state: MessagesState):
            """Generate formal FSM model using Les framework"""
            tool_messages = [msg for msg in state["messages"] if msg.type == "tool"]
            
            docs_content = "\n\n".join(msg.content for msg in tool_messages)
            system_message_content = (
                "You are a formal verification specialist implementing modern FSM modeling standards "
                "for IoT protocols. Generate formal models using TLA+, Promela/SPIN, and temporal logic "
                "following industry best practices for protocol verification.\n\n"
                "MODELING STANDARDS:\n"
                "- Use TLA+ syntax for temporal logic and state machines\n"
                "- Follow Promela conventions for model checking with SPIN\n"
                "- Include LTL properties for safety and liveness\n"
                "- Model security properties and threat scenarios\n"
                "- Ensure finite state spaces for model checking\n\n"
                "CRITICAL: Output ONLY the formal model code without explanations, "
                "markdown formatting, or additional text. Follow the exact syntax requirements.\n\n"
                f"Matter Specification Context:\n{docs_content}"
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
        graph_builder.add_node("tools", ToolNode(tools))
        graph_builder.add_node("generate_les_model", generate_les_model)
        
        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "generate_les_model")
        graph_builder.add_edge("generate_les_model", END)
        
        self.graph = graph_builder.compile(checkpointer=self.memory)
    
    def generate_fsm_for_cluster(self, cluster_id: str, query_type: str = "formal_specification", 
                                thread_id: str = None) -> Dict[str, Any]:
        """Step 3: Generate FSM model for specific cluster"""
        if cluster_id not in self.extracted_clusters:
            raise ValueError(f"Cluster {cluster_id} not found in extracted clusters")
        
        cluster = self.extracted_clusters[cluster_id]
        
        if not self.graph:
            self.setup_fsm_generation()
        
        if thread_id is None:
            thread_id = f"fsm_{cluster_id}_{query_type}"
        
        # Generate query based on cluster information
        query = self._generate_cluster_query(cluster, query_type)
        
        try:
            result = self.graph.invoke(
                {"messages": [HumanMessage(content=query)]},
                config={"configurable": {"thread_id": thread_id}}
            )
            
            formal_model = result["messages"][-1].content
            cleaned_model = self._clean_formal_model_output(formal_model)
            
            return {
                "cluster_id": cluster_id,
                "cluster_name": cluster.cluster_name,
                "query_type": query_type,
                "formal_model": cleaned_model,
                "thread_id": thread_id,
                "query": query
            }
            
        except Exception as e:
            self.logger.error(f"Error generating FSM for {cluster_id}: {e}")
            raise
    
    def _generate_cluster_query(self, cluster: MatterCluster, query_type: str) -> str:
        """Generate appropriate query for cluster FSM generation"""
        base_context = f"""
CLUSTER: {cluster.cluster_name} (ID: {cluster.cluster_id})
CATEGORY: {cluster.category.upper()}
DESCRIPTION: {cluster.description}

ATTRIBUTES: {', '.join(cluster.attributes) if cluster.attributes else 'Basic attributes'}
COMMANDS: {', '.join(cluster.commands) if cluster.commands else 'Basic commands'}
EVENTS: {', '.join(cluster.events) if cluster.events else 'Basic events'}

TASK: Generate formal FSM model following IoT protocol verification standards.
"""

        if query_type == "state_transition_rules":
            return f"""{base_context}

Generate State Transition Rules using formal FSM syntax.

OUTPUT FORMAT (NO explanations):

*** Module: {cluster.cluster_name.replace(' ', '')}Cluster

*** Type definitions
DeviceState ::= {{idle, processing, active, error}}
UserRole ::= {{admin, user, guest}}
SecurityLevel ::= {{low, medium, high}}

*** State variables
VARIABLES
    device_state,     \* Current device state
    attributes,       \* Attribute values map
    user_session,     \* Current user session
    security_context  \* Security permissions

*** State predicates
Init == 
    /\\ device_state = "idle"
    /\\ attributes = [attribute |-> "default" : attribute \\in {{"attr1", "attr2"}}]
    /\\ user_session = [role |-> "guest", authenticated |-> FALSE]
    /\\ security_context = [level |-> "low"]

*** Transition predicates for each command
CommandAction(user) ==
    /\\ user_session.authenticated = TRUE
    /\\ device_state \\in {{"idle", "active"}}
    /\\ device_state' = "processing"
    /\\ UNCHANGED <<user_session, security_context>>

*** State invariants
StateInvariant == 
    /\\ device_state \\in DeviceState
    /\\ user_session.role \\in UserRole
    /\\ security_context.level \\in SecurityLevel

Extract all state transitions for {cluster.cluster_name} cluster from Matter specification.
"""

        elif query_type == "security_properties":
            return f"""{base_context}

Generate Security Properties for {cluster.cluster_name} cluster.

OUTPUT FORMAT (NO explanations):

*** Security Model for {cluster.cluster_name.replace(' ', '')}Cluster

*** Security states
SecurityState ::= {{unauthorized, authenticating, authorized, privileged}}
ThreatLevel ::= {{none, low, medium, high, critical}}

*** Security predicates
AuthenticationRequired ==
    \\A cmd \\in commands :
        (device_state = "processing" /\\ current_command = cmd)
        => user_session.authenticated = TRUE

AccessControlCheck ==
    \\A attr \\in attributes :
        (attr_access_requested = attr)
        => security_context.level \\in {{"medium", "high"}}

*** Security violations
SV1_UnauthorizedAccess ==
    /\\ \\neg user_session.authenticated
    /\\ device_state = "processing"
    /\\ <>device_state = "active"

Extract security model for {cluster.cluster_name} cluster following formal verification standards.
"""

        else:  # formal_specification
            return f"""{base_context}

Generate Complete Formal Specification for {cluster.cluster_name} cluster.

OUTPUT FORMAT (NO explanations):

*** Formal Specification: {cluster.cluster_name.replace(' ', '')}Protocol

EXTENDS Naturals, Sequences, FiniteSets, TLC

*** Constants
CONSTANTS
    MAX_USERS,           \* Maximum concurrent users
    MAX_RETRY_ATTEMPTS,  \* Maximum command retry attempts  
    TIMEOUT_DURATION,    \* Command timeout in milliseconds
    CLUSTER_ID           \* Matter cluster identifier

*** Variables  
VARIABLES
    cluster_state,       \* Current cluster state
    attribute_values,    \* Map of attribute names to values
    pending_commands,    \* Queue of pending commands
    user_sessions,       \* Set of active user sessions
    event_history,       \* Sequence of generated events
    error_conditions,    \* Set of current error conditions
    network_status       \* Network connectivity state

*** Type definitions
ClusterState ::= {{"uninitialized", "idle", "processing", "active", "error", "shutdown"}}
CommandType ::= {{"command1", "command2", "command3"}}
AttributeType ::= {{"attr1", "attr2", "attr3"}}

*** Initial state predicate
Init ==
    /\\ cluster_state = "uninitialized"
    /\\ attribute_values = [attr \\in AttributeType |-> "default"]
    /\\ pending_commands = <<>>
    /\\ user_sessions = {{}}
    /\\ event_history = <<>>
    /\\ error_conditions = {{}}
    /\\ network_status = "connected"

Generate complete formal specification for {cluster.cluster_name} cluster.
"""
    
    def _clean_formal_model_output(self, response: str) -> str:
        """Clean up formal model output"""
        # Remove markdown formatting
        response = re.sub(r'```[a-zA-Z]*\n?', '', response)
        
        # Remove excessive newlines
        response = re.sub(r'\n{3,}', '\n\n', response)
        
        # Remove leading/trailing whitespace
        response = response.strip()
        
        return response
    
    def save_extracted_clusters(self, output_file: str = "extracted_clusters.json"):
        """Save extracted clusters to file"""
        output_path = os.path.join(os.path.dirname(self.pdf_path), output_file)
        
        # Convert clusters to serializable format
        clusters_data = {}
        for cluster_id, cluster in self.extracted_clusters.items():
            clusters_data[cluster_id] = {
                "cluster_id": cluster.cluster_id,
                "cluster_name": cluster.cluster_name,
                "description": cluster.description,
                "page_number": cluster.page_number,
                "category": cluster.category,
                "attributes": cluster.attributes,
                "commands": cluster.commands,
                "events": cluster.events
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clusters_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Clusters saved to: {output_path}")
        return output_path
    
    def print_clusters_summary(self):
        """Print summary of extracted clusters"""
        print(f"\n🔍 Extracted {len(self.extracted_clusters)} Matter Clusters")
        print("=" * 60)
        
        for i, (cluster_id, cluster) in enumerate(self.extracted_clusters.items(), 1):
            print(f"{i:2d}. {cluster_id} - {cluster.cluster_name}")
            print(f"     Category: {cluster.category}")
            if cluster.page_number:
                print(f"     Page: {cluster.page_number}")


def main():
    """Main function combining PDF extraction and FSM generation"""
    print("🚀 Matter PDF Cluster Extractor & Les FSM Generator")
    print("=" * 60)
    
    # PDF file path
    pdf_path = os.path.join(os.path.dirname(__file__), "..", "Matter-1.4-Application-Cluster-Specification.pdf")
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    try:
        # Create extractor
        extractor = MatterPDFClusterFSMExtractor(pdf_path)
        
        # Step 1: Extract clusters from PDF
        print("🔍 Step 1: Extracting clusters from PDF...")
        clusters = extractor.extract_clusters_from_pdf()
        
        # Print summary
        extractor.print_clusters_summary()
        
        # Save clusters
        clusters_file = extractor.save_extracted_clusters()
        print(f"\n💾 Clusters saved to: {clusters_file}")
        
        # Step 2: Setup FSM generation
        print("\n🔧 Step 2: Setting up FSM generation...")
        extractor.setup_fsm_generation()
        
        # Step 3: Interactive FSM generation
        print("\n🎯 Step 3: Interactive FSM Generation")
        print("\nAvailable commands:")
        print("- <cluster_id> <query_type>  : Generate specific FSM")
        print("- list                       : Show available clusters")
        print("- all <cluster_id>          : Generate all FSM types for cluster")
        print("- quit                      : Exit")
        print("\nQuery types: formal_specification, state_transition_rules, security_properties")
        
        results = []
        
        while True:
            try:
                user_input = input("\n🤖 Enter command: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if user_input.lower() == 'list':
                    print("\nAvailable clusters:")
                    for cluster_id, cluster in extractor.extracted_clusters.items():
                        print(f"  {cluster_id} - {cluster.cluster_name}")
                    continue
                
                parts = user_input.split()
                if len(parts) == 2:
                    if parts[0] == 'all':
                        cluster_id = parts[1]
                        if cluster_id in extractor.extracted_clusters:
                            print(f"\n🚀 Generating all FSM types for {cluster_id}...")
                            for query_type in ["formal_specification", "state_transition_rules", "security_properties"]:
                                try:
                                    result = extractor.generate_fsm_for_cluster(cluster_id, query_type)
                                    results.append(result)
                                    
                                    # Save result
                                    os.makedirs("fsm_results", exist_ok=True)
                                    filename = f"fsm_results/{cluster_id}_{query_type}.json"
                                    with open(filename, 'w', encoding='utf-8') as f:
                                        json.dump(result, f, indent=2, ensure_ascii=False)
                                    print(f"💾 Saved: {filename}")
                                    
                                except Exception as e:
                                    print(f"❌ Error in {query_type}: {e}")
                        else:
                            print(f"❌ Unknown cluster: {cluster_id}")
                    else:
                        cluster_id, query_type = parts
                        if cluster_id in extractor.extracted_clusters:
                            if query_type in ["formal_specification", "state_transition_rules", "security_properties"]:
                                print(f"\n🚀 Generating {query_type} for {cluster_id}...")
                                result = extractor.generate_fsm_for_cluster(cluster_id, query_type)
                                results.append(result)
                                
                                # Save result
                                os.makedirs("fsm_results", exist_ok=True)
                                filename = f"fsm_results/{cluster_id}_{query_type}.json"
                                with open(filename, 'w', encoding='utf-8') as f:
                                    json.dump(result, f, indent=2, ensure_ascii=False)
                                print(f"💾 Saved: {filename}")
                                
                                # Also save as appropriate file format
                                if query_type == "formal_specification":
                                    ext_filename = f"fsm_results/{cluster_id}_spec.tla"
                                else:
                                    ext_filename = f"fsm_results/{cluster_id}_{query_type}.txt"
                                
                                with open(ext_filename, 'w', encoding='utf-8') as f:
                                    f.write(result["formal_model"])
                                print(f"💾 Saved formal model: {ext_filename}")
                            else:
                                print(f"❌ Invalid query type: {query_type}")
                        else:
                            print(f"❌ Unknown cluster: {cluster_id}")
                else:
                    print("❌ Invalid command format")
                    print("Use: '<cluster_id> <query_type>' or 'all <cluster_id>' or 'list'")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Save all results
        if results:
            os.makedirs("fsm_results", exist_ok=True)
            all_results_file = "fsm_results/all_fsm_results.json"
            with open(all_results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Saved all {len(results)} results to: {all_results_file}")
        
        print("\n✅ PDF cluster extraction and FSM generation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        logging.error(f"Processing failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
