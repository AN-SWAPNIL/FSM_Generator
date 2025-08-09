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
    GOOGLE_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE,
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
        """Initialize the language model and embeddings"""
        try:
            self.llm = GoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=GOOGLE_API_KEY,
                temperature=GEMINI_TEMPERATURE
            )
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY
            )
            logger.info("Initialized Gemini LLM and embeddings")
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
        Create a vector store from text content
        
        Args:
            text_content: Text content to vectorize
            
        Returns:
            FAISS vector store or None if failed
        """
        try:
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            
            # Create documents
            documents = [Document(page_content=text_content)]
            texts = text_splitter.split_documents(documents)
            
            if not texts:
                logger.warning("No text chunks created for vector store")
                return None
            
            # Create vector store
            vector_store = FAISS.from_documents(texts, self.embeddings)
            logger.info(f"Created vector store with {len(texts)} chunks")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            return None
    
    def extract_cluster_details_with_rag(self, cluster_text: str, vector_store: FAISS, cluster_name: str = "") -> Dict[str, Any]:
        """
        Extract cluster details using RAG approach
        
        Args:
            cluster_text: Text content of the cluster
            vector_store: FAISS vector store for retrieval
            cluster_name: Name of the cluster for context
            
        Returns:
            Extracted cluster information as dictionary
        """
        try:
            # Create retriever
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # Retrieve relevant documents
            docs = retriever.get_relevant_documents(f"cluster information for {cluster_name}")
            
            # Combine retrieved context
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate extraction prompt with retrieved context
            prompt = f"""{CLUSTER_DETAIL_EXTRACTION_PROMPT}

TARGET CLUSTER: {cluster_name}

Retrieved relevant context:
{context}"""
            
            # Extract with LLM
            response = self.llm.invoke(prompt)
            
            # Parse JSON response
            try:
                # Clean response by removing code block markers if present
                clean_response = response.strip()
                if clean_response.startswith('```json'):
                    clean_response = clean_response[7:]  # Remove ```json
                if clean_response.endswith('```'):
                    clean_response = clean_response[:-3]  # Remove ```
                clean_response = clean_response.strip()
                
                cluster_info = json.loads(clean_response)
                return cluster_info
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error for {cluster_name}: {e}")
                logger.error(f"LLM response: {response[:500]}...")
                return self._create_fallback_cluster_info(cluster_text)
                
        except Exception as e:
            logger.error(f"Error in RAG extraction for {cluster_name}: {e}")
            return self._create_fallback_cluster_info(cluster_text)
    
    def extract_cluster_details_direct(self, cluster_text: str, cluster_name: str = "") -> Dict[str, Any]:
        """
        Extract cluster details using direct LLM approach (fallback)
        
        Args:
            cluster_text: Text content of the cluster
            cluster_name: Name of the cluster for context
            
        Returns:
            Extracted cluster information as dictionary
        """
        try:
            # Generate enhanced extraction prompt with cluster name and more context
            prompt = f"""{CLUSTER_DETAIL_EXTRACTION_PROMPT}

TARGET CLUSTER: {cluster_name}

IMPORTANT: Look for these specific patterns in the text:
1. Tables with pipe delimiters (|) or formatted spacing
2. Hex IDs starting with 0x
3. Section headers like "1.X.X" numbered sections
4. Words like "Attributes", "Commands", "Data Types", "Classification"
5. Direction indicators like → or ⇒ for commands
6. Conformance codes: M (Mandatory), O (Optional), F (Feature-dependent)

SPECIAL ATTENTION: This cluster has {len(cluster_text)} characters of text. 
Make sure to scan the ENTIRE text for Commands section, not just the beginning.
Commands section typically appears after Data Types and Attributes sections.

Cluster specification text:
{cluster_text}"""  # Remove character limit to process full text
            
            # Extract with LLM
            response = self.llm.invoke(prompt)
            
            # Parse JSON response
            try:
                # Clean response by removing code block markers if present
                clean_response = response.strip()
                if clean_response.startswith('```json'):
                    clean_response = clean_response[7:]  # Remove ```json
                if clean_response.endswith('```'):
                    clean_response = clean_response[:-3]  # Remove ```
                clean_response = clean_response.strip()
                
                cluster_info = json.loads(clean_response)
                return cluster_info
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error for {cluster_name}: {e}")
                logger.error(f"LLM response: {response[:500]}...")
                return self._create_fallback_cluster_info(cluster_text)
                
        except Exception as e:
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
        Process a single cluster
        
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
        
        # Extract cluster pages
        cluster_text = self.extract_cluster_pages(start_page, end_page)
        
        if not cluster_text:
            return self._create_fallback_cluster_info("")
        
        # Try to create vector store and use RAG if successful
        vector_store = self.create_vector_store(cluster_text)
        
        if vector_store:
            logger.info("Using RAG approach for extraction")
            cluster_details = self.extract_cluster_details_with_rag(cluster_text, vector_store, cluster_name)
        else:
            logger.info("Using direct approach for extraction")
            cluster_details = self.extract_cluster_details_direct(cluster_text, cluster_name)
        
        # Add metadata
        cluster_details['metadata'] = {
            'source_pages': f"{start_page}-{end_page}",
            'extraction_method': 'RAG' if vector_store else 'Direct',
            'text_length': len(cluster_text),
            'section_number': section_number,
            'category': cluster_data.get('category', 'Unknown')
        }
        
        return cluster_details
    
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
