#!/usr/bin/env python3
"""
RAG Agent Chat App using Ollama with Matter Specification document
Based on LangChain/LangGraph RAG implementation
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
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

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
        self.graph = None
        self.memory = MemorySaver()
        
        # Initialize components
        self._setup_llm()
        self._setup_embeddings()
        self._load_and_index_document()
        self._setup_graph()
    
    def _setup_llm(self):
        """Setup the language model"""
        print(f"🤖 Initializing {self.model_name}...")
        self.llm = ChatOllama(
            model=self.model_name,
            temperature=0.7,
            num_predict=512
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
    
    def _setup_graph(self):
        """Setup the LangGraph RAG workflow"""
        print("🔗 Setting up RAG workflow...")
        
        # Create the retrieval tool
        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """Retrieve information related to a query from the Matter specification."""
            retrieved_docs = self.vector_store.similarity_search(query, k=3)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata.get('source', 'Matter Specification')}\n"
                 f"Content: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs
        
        self.retrieve_tool = retrieve
        
        # Create graph builder
        graph_builder = StateGraph(MessagesState)
        
        # Node 1: Query or respond directly
        def query_or_respond(state: MessagesState):
            """Generate tool call for retrieval or respond directly."""
            llm_with_tools = self.llm.bind_tools([retrieve])
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}
        
        # Node 2: Tool execution
        tools = ToolNode([retrieve])
        
        # Node 3: Generate final response
        def generate(state: MessagesState):
            """Generate answer using retrieved context."""
            # Get generated ToolMessages
            recent_tool_messages = []
            for message in reversed(state["messages"]):
                if message.type == "tool":
                    recent_tool_messages.append(message)
                else:
                    break
            tool_messages = recent_tool_messages[::-1]
            
            # Format into prompt
            docs_content = "\n\n".join(doc.content for doc in tool_messages)
            system_message_content = (
                "You are an expert assistant for questions about the Matter specification. "
                "Use the following retrieved context from the Matter specification to answer "
                "the question accurately and comprehensively. If the context doesn't contain "
                "enough information to fully answer the question, say so and provide what "
                "information you can from the context. Keep your answer clear and well-organized."
                "\n\n"
                f"Retrieved Context:\n{docs_content}"
            )
            
            # Get conversation messages (excluding tool calls and tool messages)
            conversation_messages = [
                message
                for message in state["messages"]
                if message.type in ("human", "system")
                or (message.type == "ai" and not message.tool_calls)
            ]
            
            prompt = [SystemMessage(system_message_content)] + conversation_messages
            
            # Generate response
            response = self.llm.invoke(prompt)
            return {"messages": [response]}
        
        # Add nodes to graph
        graph_builder.add_node("query_or_respond", query_or_respond)
        graph_builder.add_node("tools", tools)
        graph_builder.add_node("generate", generate)
        
        # Set entry point and edges
        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)
        
        # Compile with memory
        self.graph = graph_builder.compile(checkpointer=self.memory)
        print("✅ RAG workflow setup complete")
    
    def interactive_chat(self, thread_id="default"):
        """Start interactive chat with RAG capabilities"""
        print("\n" + "="*60)
        print("🚀 Matter Specification RAG Chat Assistant")
        print("="*60)
        print("Ask questions about the Matter specification!")
        print("Commands: 'quit', 'exit', 'clear' (to clear history)")
        print("="*60)
        
        config = {"configurable": {"thread_id": thread_id}}
        
        while True:
            try:
                user_input = input("\n🙋 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    # Create new thread ID to clear history
                    thread_id = f"thread_{len(str(hash(user_input)))}"
                    config = {"configurable": {"thread_id": thread_id}}
                    print("🗑️  Chat history cleared!")
                    continue
                
                if not user_input:
                    continue
                
                print("\n🤔 Thinking...")
                
                # Stream the response
                messages_seen = 0
                for step in self.graph.stream(
                    {"messages": [{"role": "user", "content": user_input}]},
                    stream_mode="values",
                    config=config,
                ):
                    # Only print new messages
                    current_messages = len(step["messages"])
                    if current_messages > messages_seen:
                        latest_message = step["messages"][-1]
                        if hasattr(latest_message, 'content') and latest_message.content:
                            if latest_message.type == "ai" and not getattr(latest_message, 'tool_calls', None):
                                print(f"\n🤖 Assistant: {latest_message.content}")
                        messages_seen = current_messages
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or check if Ollama is running.")
    
    def ask_question(self, question: str, thread_id="default"):
        """Ask a single question and get response"""
        config = {"configurable": {"thread_id": thread_id}}
        
        for step in self.graph.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
            config=config,
        ):
            final_message = step["messages"][-1]
            if hasattr(final_message, 'content') and final_message.content:
                if final_message.type == "ai" and not getattr(final_message, 'tool_calls', None):
                    return final_message.content
        
        return "No response generated"

def main():
    """Main function"""
    print("🚀 Initializing Matter Specification RAG Chat App")
    print("="*50)
    
    # Setup LangSmith if available
    # setup_langsmith()
    
    try:
        # Initialize the RAG chat app
        app = RAGChatApp(
            model_name="llama3.1",
            document_path="Matter_Specification.html"
        )
        
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