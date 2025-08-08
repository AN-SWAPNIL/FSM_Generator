#!/usr/bin/env python3
"""
RAG Agent Chat App using Ollama with Matter Specification document
Compatible with models that don't support tool calling
"""

import os
import getpass
from pathlib import Path
from typing import List

# Core imports
from langchain_ollama.chat_models import ChatOllama
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
import re

# Set up LangSmith (optional but recommended)
def setup_langsmith():
    """Setup LangSmith for tracing"""
    if not os.environ.get("LANGSMITH_API_KEY"):
        api_key = getpass.getpass("Enter your LangSmith API key (or press Enter to skip): ")
        if api_key:
            os.environ["LANGSMITH_API_KEY"] = api_key
            os.environ["LANGSMITH_TRACING"] = "true"
            print("✅ LangSmith tracing enabled")
        else:
            print("⚠️  LangSmith tracing disabled")
    else:
        os.environ["LANGSMITH_TRACING"] = "true"
        print("✅ LangSmith tracing enabled")

class RAGChatApp:
    def __init__(self, model_name="deepseek-r1", document_path="Matter_Specification.html"):
        """Initialize the RAG Chat App"""
        self.model_name = model_name
        self.document_path = document_path
        self.vector_store = None
        self.memory = None
        self.qa_chain = None
        
        # Initialize components
        self._setup_llm()
        self._setup_embeddings()
        self._load_and_index_document()
        self._setup_rag_chain()
    
    def _setup_llm(self):
        """Setup the language model"""
        print(f"🤖 Initializing {self.model_name}...")
        self.llm = ChatOllama(
            model=self.model_name,
            temperature=0.7,
            num_predict=1024
        )
        print("✅ Language model initialized")
    
    def _setup_embeddings(self):
        """Setup embeddings model"""
        print("🔤 Initializing embeddings model...")
        # Using a lightweight but effective model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("✅ Embeddings model initialized")
    
    def _load_and_index_document(self):
        """Load and index the Matter specification document"""
        print(f"📄 Loading document: {self.document_path}")
        
        # Check if document exists
        if not Path(self.document_path).exists():
            raise FileNotFoundError(f"Document not found: {self.document_path}")
        
        # Load HTML document
        loader = UnstructuredHTMLLoader(self.document_path)
        docs = loader.load()
        
        if not docs:
            raise ValueError("No content loaded from document")
        
        print(f"📖 Loaded {len(docs)} document(s)")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        all_splits = text_splitter.split_documents(docs)
        
        print(f"✂️  Split into {len(all_splits)} chunks")
        
        # Create vector store
        print("🗃️  Creating vector store...")
        self.vector_store = FAISS.from_documents(all_splits, self.embeddings)
        print("✅ Vector store created and indexed")
    
    def _setup_rag_chain(self):
        """Setup the RAG chain without tool calling"""
        print("🔗 Setting up RAG chain...")
        
        # Create memory for conversation history
        self.memory = ConversationBufferWindowMemory(
            k=5,  # Remember last 5 exchanges
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create custom prompt template
        system_template = """You are an expert assistant for questions about the Matter specification. 
Use the following pieces of context from the Matter specification to answer the question at the end. 
If the context doesn't contain enough information to fully answer the question, say so and provide what 
information you can from the context. Keep your answer clear, well-organized, and technically accurate.

Context from Matter Specification:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:"""
        
        # Create the conversational retrieval chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
            memory=self.memory,
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={
                "prompt": ChatPromptTemplate.from_template(system_template)
            }
        )
        
        print("✅ RAG chain setup complete")
    
    def _detect_fsm_request(self, question: str) -> bool:
        """Detect if the user is asking for FSM/state machine creation"""
        fsm_keywords = [
            'fsm', 'finite state machine', 'state machine', 'state diagram', 
            'states', 'transitions', 'workflow', 'process flow', 'state flow'
        ]
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in fsm_keywords)
    
    def _create_enhanced_fsm_prompt(self, question: str, context: str) -> str:
        """Create an enhanced prompt for FSM generation"""
        return f"""Based on the Matter specification context provided, create a comprehensive finite state machine (FSM) for the requested process. 

CONTEXT FROM MATTER SPECIFICATION:
{context}

ORIGINAL QUESTION: {question}

Please provide:
1. **FSM Overview**: Brief description of what this FSM represents
2. **States**: List all states with clear descriptions
3. **Transitions**: Define all possible transitions between states with triggers/conditions
4. **Initial State**: Specify the starting state
5. **Final/Terminal States**: Identify end states if any
6. **State Diagram Format**: Present the FSM in a clear textual format
7. **Implementation Notes**: Any relevant technical details from the Matter spec

Format your response as a structured FSM specification that could be implemented in code or visualized as a diagram."""
    
    def ask_question(self, question: str) -> dict:
        """Ask a question and get response with sources"""
        try:
            # Check if this is an FSM-related request
            if self._detect_fsm_request(question):
                # First get relevant context
                docs = self.vector_store.similarity_search(question, k=6)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Create enhanced FSM prompt
                enhanced_question = self._create_enhanced_fsm_prompt(question, context)
                
                # Get response using the chain
                result = self.qa_chain({"question": enhanced_question})
            else:
                # Regular RAG query
                result = self.qa_chain({"question": question})
            
            return {
                "answer": result["answer"],
                "sources": result.get("source_documents", []),
                "success": True
            }
        except Exception as e:
            return {
                "answer": f"Error processing question: {str(e)}",
                "sources": [],
                "success": False
            }
    
    def interactive_chat(self):
        """Start interactive chat with RAG capabilities"""
        print("\n" + "="*60)
        print("🚀 Matter Specification RAG Chat Assistant")
        print("="*60)
        print("Ask questions about the Matter specification!")
        print("Special: Ask for 'FSM' or 'finite state machine' for detailed state diagrams")
        print("Commands: 'quit', 'exit', 'clear' (to clear history), 'sources' (show last sources)")
        print("="*60)
        
        last_sources = []
        
        while True:
            try:
                user_input = input("\n🙋 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    # Clear conversation memory
                    self.memory.clear()
                    print("🗑️  Chat history cleared!")
                    continue
                
                if user_input.lower() == 'sources':
                    if last_sources:
                        print("\n📚 Sources from last response:")
                        for i, doc in enumerate(last_sources, 1):
                            source = doc.metadata.get('source', 'Unknown')
                            preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                            print(f"   {i}. {source}")
                            print(f"      Preview: {preview}\n")
                    else:
                        print("No sources from previous response.")
                    continue
                
                if not user_input:
                    continue
                
                print("\n🤔 Thinking...")
                
                # Get response
                result = self.ask_question(user_input)
                last_sources = result["sources"]
                
                if result["success"]:
                    print(f"\n🤖 Assistant: {result['answer']}")
                    
                    # Show source count
                    if result["sources"]:
                        print(f"\n📚 (Based on {len(result['sources'])} relevant sections from the Matter spec)")
                        print("💡 Type 'sources' to see source details")
                else:
                    print(f"\n❌ {result['answer']}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or check if Ollama is running.")
    
    def get_document_stats(self):
        """Get statistics about the loaded document"""
        if self.vector_store:
            # This is a rough estimate since FAISS doesn't directly expose document count
            return {
                "chunks_indexed": "~2270 (as reported during loading)",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "chunk_size": 1000,
                "chunk_overlap": 200
            }
        return None

def main():
    """Main function"""
    print("🚀 Initializing Matter Specification RAG Chat App")
    print("="*50)
    
    # Setup LangSmith if available
    # setup_langsmith()
    
    try:
        # Initialize the RAG chat app
        app = RAGChatApp(
            model_name="deepseek-r1",
            document_path="Matter_Specification.html"
        )
        
        # Show document stats
        stats = app.get_document_stats()
        if stats:
            print(f"\n📊 Document Statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
        
        # Start interactive chat
        app.interactive_chat()
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Please make sure 'Matter_Specification.html' is in the same directory as this script.")
    except Exception as e:
        print(f"❌ Error initializing app: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Make sure the model is available: ollama pull deepseek-r1")
        print("3. Check if all dependencies are installed")
        print("4. Ensure Matter_Specification.html exists in the current directory")

if __name__ == "__main__":
    main()