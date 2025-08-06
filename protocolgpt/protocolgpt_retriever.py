"""
ProtocolGPT Document Retriever and Vector Store Management
Following ProtocolGPT's proven approach for document loading and retrieval
"""

import os
import time
from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import UnstructuredHTMLLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from matter_clusters import PROTOCOLGPT_CONFIG, MATTER_PATTERNS, ALLOW_FILES
import re

class ProtocolGPTRetriever:
    """
    Document retriever following ProtocolGPT's proven approach
    Handles document loading, processing, and vector store creation
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or PROTOCOLGPT_CONFIG.copy()
        self.embeddings = None
        self.vector_store = None
        self.retriever = None
        
        # Initialize embeddings
        self._setup_embeddings()
    
    def _setup_embeddings(self):
        """Setup embeddings model following ProtocolGPT pattern"""
        print("🔤 Initializing embeddings model...")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            print("✅ Embeddings model initialized")
        except Exception as e:
            print(f"❌ Error initializing embeddings: {e}")
            raise
    
    def _get_all_files(self, folder: str) -> List[str]:
        """Get all files in folder recursively (following ProtocolGPT pattern)"""
        file_paths = []
        for root, directories, files in os.walk(folder):
            for file in files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
        return file_paths
    
    def _get_matter_fsm_files(self, root_dir: str) -> List[str]:
        """
        Find Matter cluster-related files using pattern matching
        Following ProtocolGPT's code filtering approach
        """
        files = self._get_all_files(root_dir)
        fsm_files = []
        
        # Combine all Matter patterns
        combined_pattern = '|'.join(MATTER_PATTERNS.values())
        
        for filename in files:
            # Check file extension
            if any(filename.endswith(ext) for ext in ALLOW_FILES):
                try:
                    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
                        text = file.read()
                    # Look for Matter cluster patterns
                    if re.search(combined_pattern, text, re.IGNORECASE):
                        fsm_files.append(filename)
                        print(f"📄 Found Matter-related file: {filename}")
                except Exception as e:
                    # Skip files that can't be read
                    continue
        
        return fsm_files
    
    def load_specification(self, spec_path: str):
        """Load and process Matter specification following ProtocolGPT pattern"""
        print(f"📄 Loading document: {spec_path}")
        
        if not Path(spec_path).exists():
            raise FileNotFoundError(f"Document not found: {spec_path}")
        
        # Load HTML specification
        loader = UnstructuredHTMLLoader(spec_path)
        docs = loader.load()
        
        if not docs:
            raise ValueError("No content loaded from document")
        
        print(f"📖 Loaded {len(docs)} document(s)")
        
        # Use RecursiveCharacterTextSplitter for better processing (following ProtocolGPT)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"],
            length_function=len,
        )
        
        all_splits = text_splitter.split_documents(docs)
        print(f"✂️  Split into {len(all_splits)} chunks")
        
        # Create vector store following ProtocolGPT incremental approach
        self._create_vector_store(all_splits)
        
        return docs, all_splits
    
    def load_code_files(self, root_dir: str):
        """Load and process code files following ProtocolGPT pattern"""
        print(f"📂 Loading code files from: {root_dir}")
        
        # Find Matter-related files
        fsm_files = self._get_matter_fsm_files(root_dir)
        
        if not fsm_files:
            print("⚠️  No Matter-related files found")
            return [], []
        
        # Load files
        docs = []
        for file_path in fsm_files:
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                file_docs = loader.load()
                docs.extend(file_docs)
            except Exception as e:
                print(f"⚠️  Could not load {file_path}: {e}")
        
        if not docs:
            raise ValueError("No documents could be loaded")
        
        print(f"📖 Loaded {len(docs)} code document(s)")
        
        # Use language-aware splitting for code
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.CPP,  # Good for C/C++ and similar languages
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"]
        )
        
        all_splits = text_splitter.split_documents(docs)
        print(f"✂️  Split into {len(all_splits)} chunks")
        
        # Create vector store
        self._create_vector_store(all_splits)
        
        return docs, all_splits
    
    def _create_vector_store(self, splits: List):
        """Create vector store following ProtocolGPT incremental approach"""
        print("🗃️  Creating vector store...")
        
        if not splits:
            raise ValueError("No documents to create vector store")
        
        # Create initial vector store
        self.vector_store = FAISS.from_documents([splits[0]], self.embeddings)
        
        # Add remaining documents incrementally
        for i, text in enumerate(splits[1:]):
            print(f'\r🗃️  Creating vector store ({i + 1}/{len(splits)})', end='')
            self.vector_store.add_documents([text])
            time.sleep(0.01)  # Small delay to prevent overwhelming
        
        print("\n✅ Vector store created and indexed")
        
        # Create retriever
        self._create_retriever()
    
    def _create_retriever(self):
        """Create retriever following ProtocolGPT pattern"""
        print("🔗 Setting up retriever...")
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.config["k"]}
        )
        print("✅ Retriever setup complete")
    
    def get_retriever(self):
        """Get the retriever instance"""
        if self.retriever is None:
            raise RuntimeError("Retriever not initialized. Load documents first.")
        return self.retriever
    
    def search_documents(self, query: str, k: Optional[int] = None) -> List:
        """Search for relevant documents"""
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized. Load documents first.")
        
        k = k or self.config["k"]
        return self.vector_store.similarity_search(query, k=k)
    
    def save_vector_store(self, path: str):
        """Save vector store to disk"""
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized.")
        
        self.vector_store.save_local(path)
        print(f"💾 Vector store saved to: {path}")
    
    def load_vector_store(self, path: str):
        """Load vector store from disk"""
        try:
            self.vector_store = FAISS.load_local(path, self.embeddings)
            self._create_retriever()
            print(f"📁 Vector store loaded from: {path}")
            return True
        except Exception as e:
            print(f"❌ Could not load vector store: {e}")
            return False
    
    def get_store_info(self) -> dict:
        """Get information about the vector store"""
        if self.vector_store is None:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "index_size": self.vector_store.index.ntotal if hasattr(self.vector_store, 'index') else "unknown",
            "config": self.config
        }

def create_retriever(config: Optional[dict] = None) -> ProtocolGPTRetriever:
    """
    Factory function to create retriever following ProtocolGPT pattern
    
    Args:
        config: Configuration dictionary (optional)
    
    Returns:
        Initialized ProtocolGPTRetriever instance
    """
    return ProtocolGPTRetriever(config)
