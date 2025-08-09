#!/usr/bin/env python3
"""
Matter Cluster Detail Extractor
Extracts detailed cluster information from Matter specification PDF
"""

import json
import logging
import os
import sys
import signal
from typing import Dict, List, Optional, Any
import fitz  # PyMuPDF
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import (
    GOOGLE_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE, GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_TOP_P, GEMINI_TOP_K, GEMINI_SAFETY_SETTINGS,
    CLUSTER_DETAIL_EXTRACTION_PROMPT, CLUSTER_PAGE_BUFFER, PDF_PAGE_OFFSET
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClusterDetailExtractor:
    """Extracts detailed information from Matter cluster specifications"""
    
    def __init__(self, pdf_path: str, clusters_json_path: str, output_path: str = "matter_clusters_detailed.json"):
        """
        Initialize the cluster detail extractor
        
        Args:
            pdf_path: Path to the Matter specification PDF
            clusters_json_path: Path to the clusters TOC JSON file
            output_path: Path to output JSON file
        """
        self.pdf_path = pdf_path
        self.clusters_json_path = clusters_json_path
        self.output_path = output_path
        self.pdf_doc = None
        self.llm = None
        self.embeddings = None
        self.clusters_data = None
        self.current_results = None
        self.interruption_handler_set = False
        
        # Initialize components
        self._load_pdf()
        self._init_llm()
        self._load_clusters_data()
        self._setup_interruption_handler()
    
    def _setup_interruption_handler(self):
        """Setup signal handlers for graceful interruption"""
        if not self.interruption_handler_set:
            def signal_handler(signum, frame):
                logger.warning(f"Received signal {signum}. Saving current progress...")
                self._save_current_progress()
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
            signal.signal(signal.SIGTERM, signal_handler)  # Termination
            self.interruption_handler_set = True
            logger.info("Interruption handlers set up for fail-safe operation")
    
    def _save_current_progress(self):
        """Save current progress to output file"""
        if self.current_results and len(self.current_results.get('clusters', [])) > 0:
            try:
                backup_path = f"{self.output_path}.backup"
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(self.current_results, f, indent=2, ensure_ascii=False)
                logger.info(f"Progress saved to {backup_path}")
                
                # Also save to main output file
                with open(self.output_path, 'w', encoding='utf-8') as f:
                    json.dump(self.current_results, f, indent=2, ensure_ascii=False)
                logger.info(f"Progress saved to {self.output_path}")
            except Exception as e:
                logger.error(f"Error saving progress: {e}")
    
    def load_existing_results(self) -> Dict[str, Any]:
        """Load existing results from output file if it exists"""
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                logger.info(f"Loaded existing results with {len(existing_results.get('clusters', []))} clusters")
                return existing_results
            except Exception as e:
                logger.warning(f"Error loading existing results: {e}")
        return None
    
    def get_processed_section_numbers(self, existing_results: Dict[str, Any]) -> set:
        """Get set of already processed section numbers"""
        processed = set()
        if existing_results and 'clusters' in existing_results:
            for cluster in existing_results['clusters']:
                metadata = cluster.get('metadata', {})
                section_num = metadata.get('section_number')
                if section_num:
                    processed.add(section_num)
        logger.info(f"Found {len(processed)} already processed clusters: {sorted(processed)}")
        return processed
    
    def _load_pdf(self):
        """Load the PDF document"""
        try:
            self.pdf_doc = fitz.open(self.pdf_path)
            logger.info(f"Loaded PDF with {len(self.pdf_doc)} pages")
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            raise
    
    def _init_llm(self):
        """Initialize the language model and embeddings with enhanced settings"""
        try:
            self.llm = GoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=GOOGLE_API_KEY,
                temperature=GEMINI_TEMPERATURE,
                max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                top_p=GEMINI_TOP_P,
                top_k=GEMINI_TOP_K,
                safety_settings=GEMINI_SAFETY_SETTINGS
            )
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY
            )
            logger.info("Initialized Gemini LLM and embeddings with enhanced settings")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise
    
    def _load_clusters_data(self):
        """Load clusters data from JSON file"""
        try:
            with open(self.clusters_json_path, 'r', encoding='utf-8') as f:
                self.clusters_data = json.load(f)
            logger.info(f"Loaded {len(self.clusters_data.get('clusters', {}))} clusters")
        except Exception as e:
            logger.error(f"Error loading clusters data: {e}")
            raise
    
    def extract_cluster_pages(self, start_page: int, end_page: int) -> str:
        """
        Extract text content from PDF pages for a specific cluster
        
        Args:
            start_page: Starting page number (1-indexed)
            end_page: Ending page number (1-indexed)
            
        Returns:
            Extracted text content
        """
        try:
            # Apply page offset for PDF structure
            actual_start = start_page + PDF_PAGE_OFFSET
            actual_end = end_page + PDF_PAGE_OFFSET
            
            # Add buffer and ensure valid page range
            buffer_start = max(1, actual_start - CLUSTER_PAGE_BUFFER)
            buffer_end = min(len(self.pdf_doc), actual_end + CLUSTER_PAGE_BUFFER)
            
            text_content = []
            for page_num in range(buffer_start - 1, buffer_end):  # Convert to 0-indexed
                page = self.pdf_doc[page_num]
                text = page.get_text()
                text_content.append(f"--- Page {page_num + 1} (Content Page {page_num + 1 - PDF_PAGE_OFFSET}) ---\n{text}")
            
            full_text = "\n\n".join(text_content)
            logger.info(f"Extracted text from PDF pages {buffer_start}-{buffer_end} (content pages {start_page}-{end_page}) ({len(full_text)} characters)")
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting pages {start_page}-{end_page}: {e}")
            return ""
    
    def create_vector_store(self, text_content: str) -> Optional[FAISS]:
        """
        Create a vector store from text content with enhanced chunking for technical specs
        
        Args:
            text_content: Text content to vectorize
            
        Returns:
            FAISS vector store or None if failed
        """
        try:
            # Enhanced text splitter for better table and technical content capture
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,  # Optimized for table structures
                chunk_overlap=250,  # More overlap to ensure table continuity
                separators=[
                    "\n\n\n",  # Major section breaks
                    "\n\n",    # Paragraph breaks
                    "\n---",   # Page breaks
                    "\nID\t",  # Table headers
                    "\nID ",   # Alternative table headers
                    "\n| ",    # Markdown table rows
                    "\n",      # Line breaks
                    ". ",      # Sentence breaks
                    " ",       # Word breaks
                ],
                length_function=len,
                keep_separator=True  # Keep separators to maintain table structure
            )
            
            # Pre-process text to identify and preserve table structures
            enhanced_text = self._enhance_text_for_chunking(text_content)
            
            # Create documents
            documents = [Document(page_content=enhanced_text)]
            texts = text_splitter.split_documents(documents)
            
            if not texts:
                logger.warning("No text chunks created for vector store")
                return None
            
            # Add metadata to chunks for better retrieval
            for i, text in enumerate(texts):
                text.metadata = {
                    "chunk_id": i,
                    "has_table": self._contains_table_structure(text.page_content),
                    "section_type": self._identify_section_type(text.page_content)
                }
            
            # Create vector store
            vector_store = FAISS.from_documents(texts, self.embeddings)
            logger.info(f"Created vector store with {len(texts)} chunks")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            return None
    
    def _enhance_text_for_chunking(self, text: str) -> str:
        """Enhance text to preserve table structures during chunking"""
        lines = text.split('\n')
        enhanced_lines = []
        
        for line in lines:
            # Mark table-like structures
            if (any(indicator in line.lower() for indicator in ['id', 'name', 'type', 'conformance']) and
                (len(line.split()) >= 4 or '\t' in line)):
                enhanced_lines.append(f"[TABLE_ROW] {line}")
            else:
                enhanced_lines.append(line)
        
        return '\n'.join(enhanced_lines)
    
    def _contains_table_structure(self, text: str) -> bool:
        """Check if text contains table-like structures"""
        table_indicators = [
            '[TABLE_ROW]', 'ID\t', 'ID ', '| ', 
            'Name\t', 'Type\t', 'Conformance',
            'Direction', 'Response', 'Access'
        ]
        return any(indicator in text for indicator in table_indicators)
    
    def _identify_section_type(self, text: str) -> str:
        """Identify the type of section based on content"""
        text_lower = text.lower()
        if 'attributes' in text_lower:
            return 'attributes'
        elif 'commands' in text_lower:
            return 'commands'
        elif 'data types' in text_lower:
            return 'data_types'
        elif 'features' in text_lower:
            return 'features'
        elif 'classification' in text_lower:
            return 'classification'
        else:
            return 'general'
    
    def extract_cluster_details_with_rag(self, cluster_text: str, vector_store: FAISS, cluster_name: str = "") -> Dict[str, Any]:
        """
        Extract cluster details using enhanced RAG approach
        
        Args:
            cluster_text: Text content of the cluster
            vector_store: FAISS vector store for retrieval
            cluster_name: Name of the cluster for context
            
        Returns:
            Extracted cluster information as dictionary
        """
        try:
            # Create retriever with more documents for complex content
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 12}  # Increased for comprehensive retrieval
            )
            
            # Enhanced search queries targeting specific technical sections
            search_queries = [
                f"{cluster_name} cluster attributes table ID Name Type Constraint Quality Default Access Conformance",
                f"{cluster_name} cluster commands table ID Name Direction Response Access Conformance",
                f"{cluster_name} cluster data types enumeration values",
                f"{cluster_name} cluster features bit code conformance",
                f"{cluster_name} cluster classification hierarchy role scope PICS",
                f"{cluster_name} cluster revision history",
                f"command fields data fields effect on receipt {cluster_name}",
                f"attribute descriptions {cluster_name}",
                f"state machine {cluster_name}",
                f"scene table extensions {cluster_name}"
            ]
            
            all_docs = []
            for query in search_queries:
                try:
                    docs = retriever.get_relevant_documents(query)
                    all_docs.extend(docs)
                except Exception as e:
                    logger.warning(f"Search query failed: {query} - {e}")
            
            # Remove duplicates and combine context
            unique_docs = []
            seen_content = set()
            for doc in all_docs:
                if doc.page_content not in seen_content:
                    unique_docs.append(doc)
                    seen_content.add(doc.page_content)
            
            # Sort by relevance - prioritize table content
            table_docs = [doc for doc in unique_docs if self._contains_table_structure(doc.page_content)]
            other_docs = [doc for doc in unique_docs if not self._contains_table_structure(doc.page_content)]
            
            # Combine with table content first
            priority_docs = table_docs + other_docs[:8]  # Limit total context
            context = "\n\n=== SECTION ===\n".join([doc.page_content for doc in priority_docs])
            
            # Enhanced extraction prompt with specific instructions
            prompt = f"""{CLUSTER_DETAIL_EXTRACTION_PROMPT}

TARGET CLUSTER: {cluster_name}

CRITICAL EXTRACTION RULES:
1. Find ALL tables - look for patterns like "ID Name Type" or "0x00 CommandName"
2. Extract EVERY row from attribute and command tables
3. Look for command field specifications (especially for complex commands)
4. Include ALL enumeration values for data types
5. Extract complete conformance information
6. Find effect descriptions for commands

Retrieved relevant context:
{context}

IMPORTANT: The context above contains the key sections. Scan ALL sections thoroughly for:
- Complete attributes table with all columns
- Complete commands table with all commands
- Command field specifications 
- Data type value enumerations
- Feature definitions"""
            
            # Extract with LLM
            response = self.llm.invoke(prompt)
            
            # Parse JSON response with better error handling
            try:
                # Clean response by removing code block markers if present
                clean_response = response.strip()
                if clean_response.startswith('```json'):
                    clean_response = clean_response[7:]
                if clean_response.endswith('```'):
                    clean_response = clean_response[:-3]
                clean_response = clean_response.strip()
                
                cluster_info = json.loads(clean_response)
                
                # Validate extracted data
                if self._validate_extraction(cluster_info, cluster_name):
                    return cluster_info
                else:
                    logger.warning(f"Extraction validation failed for {cluster_name}, trying fallback")
                    return self._create_fallback_cluster_info(cluster_text)
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error for {cluster_name}: {e}")
                logger.error(f"LLM response excerpt: {response[:500]}...")
                return self._create_fallback_cluster_info(cluster_text)
                
        except Exception as e:
            logger.error(f"Error in RAG extraction for {cluster_name}: {e}")
            return self._create_fallback_cluster_info(cluster_text)
    
    def _validate_extraction(self, cluster_info: Dict[str, Any], cluster_name: str) -> bool:
        """Validate that extraction contains essential information"""
        try:
            if not cluster_info or 'cluster_info' not in cluster_info:
                return False
            
            info = cluster_info['cluster_info']
            
            # Check essential fields
            required_fields = ['cluster_name', 'cluster_id', 'classification']
            for field in required_fields:
                if field not in info or not info[field]:
                    logger.warning(f"Missing required field '{field}' for {cluster_name}")
                    return False
            
            # Check that we have some substantial content
            has_content = (
                len(info.get('attributes', [])) > 0 or
                len(info.get('commands', [])) > 0 or
                len(info.get('data_types', [])) > 0
            )
            
            if not has_content:
                logger.warning(f"No substantial content extracted for {cluster_name}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error for {cluster_name}: {e}")
            return False
    
    def extract_cluster_details_direct(self, cluster_text: str, cluster_name: str = "") -> Dict[str, Any]:
        """
        Extract cluster details using enhanced direct LLM approach (fallback)
        
        Args:
            cluster_text: Text content of the cluster
            cluster_name: Name of the cluster for context
            
        Returns:
            Extracted cluster information as dictionary
        """
        try:
            # Truncate text if too long, but preserve key sections
            max_chars = 35000
            if len(cluster_text) > max_chars:
                # Find key sections and preserve them
                lines = cluster_text.split('\n')
                key_sections = []
                current_section = []
                
                for line in lines:
                    if any(keyword in line.lower() for keyword in 
                          ['attributes', 'commands', 'data types', 'features', 'classification']):
                        if current_section:
                            key_sections.append('\n'.join(current_section))
                        current_section = [line]
                    else:
                        current_section.append(line)
                
                if current_section:
                    key_sections.append('\n'.join(current_section))
                
                # Combine key sections and truncate if needed
                cluster_text = '\n\n'.join(key_sections)
                if len(cluster_text) > max_chars:
                    cluster_text = cluster_text[:max_chars] + "\n\n[TEXT TRUNCATED]"
            
            # Enhanced extraction prompt with specific guidance
            prompt = f"""{CLUSTER_DETAIL_EXTRACTION_PROMPT}

TARGET CLUSTER: {cluster_name}

ENHANCED EXTRACTION INSTRUCTIONS:
1. SCAN THE ENTIRE TEXT - look for tables that may appear anywhere in the text
2. ATTRIBUTES TABLE: Look for patterns like "ID Name Type Constraint Quality Default Access Conformance"
3. COMMANDS TABLE: Look for patterns like "ID Name Direction Response Access Conformance"
4. COMMAND FIELDS: Look for detailed command specifications with data fields
5. DATA TYPES: Find enumeration values and bitmap definitions
6. EXTRACT ALL ROWS from each table - do not stop at the first few entries

SPECIFIC PATTERNS TO FIND:
- Section headers: "1.X.Y. AttributeName" or "CommandName Command"
- Hex IDs: "0x0000", "0x4000", etc.
- Table separators: pipe characters |, tabs, or aligned spacing
- Conformance codes: M, O, LT, !OFFONLY, etc.
- Command directions: "client ⇒ server", "client => server"

TEXT LENGTH: {len(cluster_text)} characters
CRITICAL: Process the ENTIRE text below, not just the beginning!

Cluster specification text:
{cluster_text}"""
            
            # Extract with LLM
            response = self.llm.invoke(prompt)
            
            # Parse JSON response with enhanced error handling
            try:
                # Clean response by removing code block markers if present
                clean_response = response.strip()
                if clean_response.startswith('```json'):
                    clean_response = clean_response[7:]
                if clean_response.endswith('```'):
                    clean_response = clean_response[:-3]
                clean_response = clean_response.strip()
                
                cluster_info = json.loads(clean_response)
                
                # Validate extraction
                if self._validate_extraction(cluster_info, cluster_name):
                    return cluster_info
                else:
                    logger.warning(f"Direct extraction validation failed for {cluster_name}")
                    return self._create_fallback_cluster_info(cluster_text)
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error for {cluster_name}: {e}")
                logger.error(f"LLM response excerpt: {response[:500]}...")
                return self._create_fallback_cluster_info(cluster_text)
                
        except Exception as e:
            logger.error(f"Error in direct extraction for {cluster_name}: {e}")
            return self._create_fallback_cluster_info(cluster_text)
            logger.error(f"Error in direct extraction for {cluster_name}: {e}")
            return self._create_fallback_cluster_info(cluster_text)
    
    def _create_fallback_cluster_info(self, cluster_text: str) -> Dict[str, Any]:
        """
        Create fallback cluster info when extraction fails
        
        Args:
            cluster_text: Original cluster text
            
        Returns:
            Basic cluster information structure
        """
        return {
            "cluster_info": {
                "cluster_name": "Unknown",
                "cluster_id": "Unknown",
                "classification": {},
                "revision_history": [],
                "features": [],
                "data_types": [],
                "attributes": [],
                "commands": [],
                "events": [],
                "raw_text_preview": cluster_text[:1000] if cluster_text else ""
            }
        }
    
    def process_cluster(self, cluster_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single cluster with enhanced extraction strategies
        
        Args:
            cluster_data: Dictionary containing cluster information
            
        Returns:
            Extracted cluster details
        """
        cluster_name = cluster_data.get('cluster_name', 'Unknown')
        start_page = cluster_data.get('start_page', 0)
        end_page = cluster_data.get('end_page', 0)
        section_number = cluster_data.get('section_number', 'Unknown')
        
        logger.info(f"Processing cluster: {cluster_name} (section {section_number}, pages {start_page}-{end_page})")
        
        # Extract cluster pages with error handling
        try:
            cluster_text = self.extract_cluster_pages(start_page, end_page)
        except Exception as e:
            logger.error(f"Failed to extract pages for {cluster_name}: {e}")
            return self._create_fallback_cluster_info("")
        
        if not cluster_text or len(cluster_text) < 100:
            logger.warning(f"Insufficient text extracted for {cluster_name}")
            return self._create_fallback_cluster_info(cluster_text)
        
        # Try multiple extraction approaches for better results
        cluster_details = None
        
        # Strategy 1: RAG approach (preferred for large content)
        if len(cluster_text) > 8000:
            try:
                vector_store = self.create_vector_store(cluster_text)
                if vector_store:
                    logger.info("Using RAG approach for extraction")
                    cluster_details = self.extract_cluster_details_with_rag(cluster_text, vector_store, cluster_name)
                    
                    # Validate RAG results
                    if not self._validate_extraction(cluster_details, cluster_name):
                        logger.warning(f"RAG extraction validation failed for {cluster_name}, trying direct approach")
                        cluster_details = None
            except Exception as e:
                logger.error(f"RAG extraction failed for {cluster_name}: {e}")
        
        # Strategy 2: Direct approach (fallback or for smaller content)
        if not cluster_details:
            try:
                logger.info("Using direct approach for extraction")
                cluster_details = self.extract_cluster_details_direct(cluster_text, cluster_name)
            except Exception as e:
                logger.error(f"Direct extraction failed for {cluster_name}: {e}")
        
        # Strategy 3: Fallback with basic info extraction
        if not cluster_details or not self._validate_extraction(cluster_details, cluster_name):
            logger.warning(f"All extraction methods failed for {cluster_name}, creating fallback")
            cluster_details = self._create_enhanced_fallback_cluster_info(cluster_text, cluster_name, cluster_data)
        
        # Add comprehensive metadata
        cluster_details['metadata'] = {
            'source_pages': f"{start_page}-{end_page}",
            'extraction_method': self._determine_extraction_method(cluster_details),
            'text_length': len(cluster_text),
            'section_number': section_number,
            'category': cluster_data.get('category', 'Unknown'),
            'processing_timestamp': self._get_timestamp(),
            'validation_passed': self._validate_extraction(cluster_details, cluster_name)
        }
        
        return cluster_details
    
    def _determine_extraction_method(self, cluster_details: Dict[str, Any]) -> str:
        """Determine which extraction method was used based on content quality"""
        if not cluster_details:
            return "Fallback"
        
        info = cluster_details.get('cluster_info', {})
        
        # Check for comprehensive content
        has_attributes = len(info.get('attributes', [])) > 0
        has_commands = len(info.get('commands', [])) > 0
        has_data_types = len(info.get('data_types', [])) > 0
        
        if has_attributes and has_commands and has_data_types:
            return "RAG"
        elif has_attributes or has_commands:
            return "Direct"
        else:
            return "Fallback"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata"""
        import datetime
        return datetime.datetime.now().isoformat()
    
    def _create_enhanced_fallback_cluster_info(self, cluster_text: str, cluster_name: str, cluster_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create enhanced fallback cluster info with basic pattern extraction
        
        Args:
            cluster_text: Original cluster text
            cluster_name: Name of the cluster
            cluster_data: Original cluster data
            
        Returns:
            Enhanced basic cluster information structure
        """
        # Try to extract basic information using simple patterns
        basic_info = {
            "cluster_name": cluster_name,
            "cluster_id": "Unknown",
            "classification": {
                "hierarchy": "Unknown",
                "role": "Unknown", 
                "scope": "Unknown",
                "pics_code": "Unknown"
            },
            "revision_history": [],
            "features": [],
            "data_types": [],
            "attributes": [],
            "commands": [],
            "events": []
        }
        
        # Try to find cluster ID using simple pattern matching
        import re
        cluster_id_match = re.search(r'0x[0-9A-Fa-f]{4}', cluster_text)
        if cluster_id_match:
            basic_info["cluster_id"] = cluster_id_match.group()
        
        # Try to find PICS code
        pics_match = re.search(r'PICS Code\s+(\w+)', cluster_text)
        if pics_match:
            basic_info["classification"]["pics_code"] = pics_match.group(1)
        
        return {
            "cluster_info": basic_info
        }
    
    def process_all_clusters(self, limit: Optional[int] = None, resume: bool = True) -> Dict[str, Any]:
        """
        Process all clusters and extract detailed information with resume capability
        
        Args:
            limit: Optional limit on number of clusters to process
            resume: Whether to resume from existing results
            
        Returns:
            Dictionary containing all cluster details
        """
        clusters_dict = self.clusters_data.get('clusters', {})
        
        # Load existing results if resume is enabled
        existing_results = None
        processed_sections = set()
        
        if resume:
            existing_results = self.load_existing_results()
            if existing_results:
                processed_sections = self.get_processed_section_numbers(existing_results)
        
        # Convert dictionary to list of cluster data, filtering out already processed
        clusters_list = []
        for section_num, cluster_data in clusters_dict.items():
            if not resume or section_num not in processed_sections:
                cluster_data['section_number'] = section_num
                clusters_list.append(cluster_data)
        
        if limit:
            clusters_list = clusters_list[:limit]
            logger.info(f"Processing first {limit} clusters")
        
        # Initialize results structure
        if existing_results and resume:
            results = existing_results
            results['extraction_info']['total_clusters'] = len(processed_sections) + len(clusters_list)
            results['extraction_info']['extraction_timestamp'] = __import__('datetime').datetime.now().isoformat()
            logger.info(f"Resuming extraction. Already processed: {len(processed_sections)}, Remaining: {len(clusters_list)}")
        else:
            results = {
                'extraction_info': {
                    'total_clusters': len(clusters_list),
                    'pdf_source': self.pdf_path,
                    'extraction_timestamp': __import__('datetime').datetime.now().isoformat()
                },
                'clusters': []
            }
        
        # Store current results for fail-safe
        self.current_results = results
        
        for i, cluster in enumerate(clusters_list, 1):
            logger.info(f"Processing cluster {i}/{len(clusters_list)}")
            
            try:
                cluster_details = self.process_cluster(cluster)
                results['clusters'].append(cluster_details)
                
                # Save progress after each cluster
                self._save_current_progress()
                
            except Exception as e:
                logger.error(f"Error processing cluster {cluster.get('cluster_name', 'Unknown')}: {e}")
                # Add error entry
                error_entry = self._create_fallback_cluster_info("")
                error_entry['error'] = str(e)
                results['clusters'].append(error_entry)
                
                # Save progress even after error
                self._save_current_progress()
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        Save extraction results to JSON file
        
        Args:
            results: Extraction results
            output_path: Output file path
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

def main():
    """Main function"""
    # File paths - PDF is in parent directory
    pdf_path = os.path.join("..", "Matter-1.4-Application-Cluster-Specification.pdf")
    clusters_json_path = "matter_clusters_toc.json"
    output_path = "matter_clusters_detailed.json"
    
    # Check if files exist
    if not os.path.exists(clusters_json_path):
        logger.error(f"Clusters JSON file not found: {clusters_json_path}")
        sys.exit(1)
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        sys.exit(1)
    
    try:
        # Initialize extractor with output path
        extractor = ClusterDetailExtractor(pdf_path, clusters_json_path, output_path)
        
        # Process all clusters with resume capability
        logger.info("Starting cluster detail extraction for ALL clusters")
        results = extractor.process_all_clusters(resume=True)  # Enable resume by default
        
        # Save final results
        extractor.save_results(results, output_path)
        
        logger.info(f"Cluster detail extraction completed successfully!")
        logger.info(f"Processed {len(results['clusters'])} clusters")
        logger.info(f"Results saved to: {output_path}")
        
    except KeyboardInterrupt:
        logger.warning("Extraction interrupted by user. Progress has been saved.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Cluster detail extraction failed: {e}")
        sys.exit(1)
    
    finally:
        # Clean up
        if 'extractor' in locals() and extractor.pdf_doc:
            extractor.pdf_doc.close()

if __name__ == "__main__":
    main()
