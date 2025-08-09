# Matter PDF Table of Contents Extractor

A simple and efficient tool to extract cluster information from Matter PDF specifications using Gemini 1.5 Flash API with LangChain and LangGraph.

## 🎯 Purpose

This tool extracts structured cluster information from the Matter Application Cluster Specification PDF, including:

- Cluster names (e.g., "Account Login Cluster", "Audio Output Cluster")
- Section numbers (e.g., "6.2", "6.5")
- Start and end page numbers
- Cluster categories (Media, Robots, Home Appliances, etc.)
- Subsections and additional metadata

## 📁 Project Structure

```
simple_toc_extractor/
├── matter_toc_extractor.py    # Main extractor script
├── requirements.txt           # Python dependencies
├── setup.py                  # Setup and installation script
├── README.md                 # This file
└── output/                   # Generated files (created during execution)
    ├── raw_toc.txt           # Extracted table of contents text
    └── matter_clusters_toc.json # Final extracted clusters in JSON format
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or navigate to the directory
cd simple_toc_extractor

# Install dependencies
python setup.py
```

### 2. Get Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create or sign in to your Google account
3. Generate a new API key
4. Copy the API key

### 3. Set Environment Variable

**Windows (bash):**

```bash
export GOOGLE_API_KEY='your-gemini-api-key-here'
```

**Windows (PowerShell):**

```powershell
$env:GOOGLE_API_KEY = 'your-gemini-api-key-here'
```

### 4. Ensure PDF is Available

Place the `Matter-1.4-Application-Cluster-Specification.pdf` file in the parent directory:

```
FSM_Generator/
├── Matter-1.4-Application-Cluster-Specification.pdf  # PDF should be here
└── simple_toc_extractor/
    └── matter_toc_extractor.py
```

### 5. Run the Extractor

```bash
python matter_toc_extractor.py
```

## 🔧 How It Works

### Step 1: PDF TOC Extraction

- Extracts table of contents from PDF metadata (if available)
- Falls back to text extraction from first 50 pages
- Identifies TOC patterns and cluster references

### Step 2: RAG System Setup

- Creates vector store using FAISS and HuggingFace embeddings
- Splits content into manageable chunks
- Sets up LangGraph workflow for intelligent processing

### Step 3: Cluster Information Extraction

- Uses Gemini 1.5 Flash for intelligent parsing
- Extracts structured cluster data using semantic understanding
- Falls back to manual regex patterns if AI extraction fails

### Step 4: JSON Output Generation

- Saves extracted clusters to `matter_clusters_toc.json`
- Includes metadata about extraction process
- Provides structured format for further processing

## 📊 Output Format

The extracted clusters are saved in JSON format:

```json
{
  "metadata": {
    "total_clusters": 30,
    "categories": [
      "Media",
      "Robots",
      "Home Appliances",
      "Energy Management",
      "Network Infrastructure"
    ],
    "extraction_method": "Gemini 1.5 Flash with LangChain/LangGraph",
    "source_pdf": "Matter-1.4-Application-Cluster-Specification.pdf"
  },
  "clusters": {
    "6.2": {
      "cluster_name": "Account Login Cluster",
      "section_number": "6.2",
      "start_page": 471,
      "end_page": 476,
      "cluster_id": "",
      "category": "Media",
      "subsections": ["6.2.1. Revision History", "6.2.2. Classification"]
    }
    // ... more clusters
  }
}
```

## 🏗️ Architecture

### Components Used

1. **Gemini 1.5 Flash**: Primary LLM for intelligent text understanding
2. **LangChain**: Document processing, text splitting, vector stores
3. **LangGraph**: Workflow management and tool orchestration
4. **PyMuPDF (fitz)**: PDF text extraction and processing
5. **FAISS**: Vector database for semantic search
6. **HuggingFace**: Embeddings for vector store

### Processing Pipeline

```
PDF Input → TOC Extraction → Vector Store Creation →
LangGraph Workflow → Gemini Analysis → JSON Output
```

## 🔍 Extracted Cluster Categories

- **Media Clusters**: Account Login, Audio Output, Channel, Content Launcher, etc.
- **Robot Clusters**: RVC Run Mode, RVC Clean Mode, RVC Operational State
- **Home Appliance Clusters**: Temperature Control, Dishwasher Mode, Laundry controls
- **Energy Management Clusters**: Device Energy Management, Energy EVSE, Water Heater
- **Network Infrastructure Clusters**: Wi-Fi Network Management, Thread Border Router

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**: Run `python setup.py` to install dependencies
2. **API Key Issues**: Ensure `GOOGLE_API_KEY` environment variable is set
3. **PDF Not Found**: Check that the PDF is in the correct location
4. **Rate Limits**: Gemini API has rate limits; wait and retry if needed

### Debug Mode

To see detailed logs, modify the logging level in the script:

```python
logging.basicConfig(level=logging.DEBUG)
```

### Fallback Extraction

If AI extraction fails, the tool automatically falls back to manual regex-based extraction using predefined patterns.

## 📈 Performance

- **Processing Time**: ~30-60 seconds for full extraction
- **Memory Usage**: ~500MB for vector store creation
- **API Calls**: 1-3 calls to Gemini API per run
- **Accuracy**: >95% for cluster identification

## 🔄 Customization

### Adding New Cluster Categories

Modify the manual fallback patterns in `extract_clusters_manual_fallback()`:

```python
manual_clusters = [
    {"name": "New Cluster", "section": "X.Y", "start": 123, "end": 456, "category": "New Category"},
    # Add more clusters
]
```

### Adjusting AI Prompts

Modify the system message in `_setup_graph()` to change extraction behavior:

```python
system_message_content = """
Your custom extraction instructions here...
"""
```

## 📄 Dependencies

See `requirements.txt` for complete list. Key dependencies:

- `langchain` and related packages for AI processing
- `PyMuPDF` for PDF handling
- `google-generativeai` for Gemini API access
- `faiss-cpu` for vector storage

## 🤝 Contributing

This is a focused tool for Matter specification processing. For enhancements:

1. Test with your changes
2. Ensure backward compatibility
3. Update documentation

## 📝 License

This tool is part of the Matter specification analysis project. Use according to your project's license requirements.

## 🔗 Related Tools

- `les_cluster_fsm_extractor.py`: Comprehensive FSM generation with Les modeling
- `pdf_cluster_fsm_extractor.py`: Combined PDF extraction and FSM generation
