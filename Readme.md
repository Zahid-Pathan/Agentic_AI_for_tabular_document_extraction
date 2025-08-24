# Enhanced Table Extraction Workflow

## Overview

This system extracts tables from PDF documents using a multi-agent architecture powered by Hugging Face LLMs and LangGraph orchestration. The workflow intelligently combines PDF parsing, OCR fallback, and LLM-powered decision making.

<img width="2076" height="1086" alt="image" src="https://github.com/user-attachments/assets/5df8a07b-8bf1-46da-a29f-95fe8e53debb" />




## Architecture Components

### 1. **Enhanced Master Agent** 🤖
- **Role**: Orchestrates the entire workflow using LangGraph
- **LLM**: Hugging Face model (default: microsoft/DialoGPT-small)
- **Features**: 
  - Intelligent decision making
  - Tool calling capabilities
  - Conversation history tracking
  - Quality analysis and summarization

### 2. **PDF Agent** 📄
- **Role**: Primary table extraction using pdfplumber
- **Method**: Vector-based PDF parsing
- **Output**: Structured table data with cell metadata
- **Features**:
  - Header detection (row/column)
  - Cell-level metadata with coordinates
  - Confidence scoring

### 3. **OCR Agent** 👁️
- **Role**: Fallback for image-based PDFs
- **Method**: Enhanced Tesseract OCR with preprocessing
- **Features**:
  - Multi-DPI conversion (300-450 DPI)
  - Advanced image preprocessing
  - Noise filtering and character enhancement

### 4. **LLM Tools** 🛠️
- `analyze_extraction_results`: Quality assessment
- `make_extraction_decision`: Next step determination  
- `generate_final_summary`: Comprehensive summarization

## Workflow Process

### Phase 1: Initialization
1. Validate PDF file format
2. Initialize agents and LLM
3. Create conversation context

### Phase 2: PDF Extraction
1. Extract tables using pdfplumber
2. Generate cell metadata with headers
3. Calculate confidence scores
4. Log extraction statistics

### Phase 3: Smart Decision Making
```
IF PDF_TABLES_FOUND > 0:
    → Proceed to LLM Analysis
ELSE IF OCR_AVAILABLE:
    → Try OCR Extraction
ELSE:
    → Proceed with empty results
```

### Phase 4: LLM Analysis
1. **Context Building**: Create analysis prompt with extraction results
2. **LLM Reasoning**: Generate quality assessment and recommendations
3. **Decision Logic**: Determine next steps based on LLM analysis
4. **Tool Integration**: Use specialized tools for detailed analysis

### Phase 5: Finalization
1. **Data Filtering**: Keep only non-header cells with valid headers
2. **LLM Summary**: Generate comprehensive extraction summary
3. **Output Structure**: Create final JSON with metadata and conversation log
4. **Quality Metrics**: Include confidence scores and processing statistics

## Output Format

```json
{
  "master_agent_metadata": {
    "extraction_method_used": "pdfplumber|ocr",
    "workflow_completed": true,
    "llm_powered": true,
    "total_agents_involved": 3,
    "final_confidence": 0.85
  },
  "llm_analysis": {
    "analysis": "LLM-generated quality assessment",
    "decision_reasoning": "Step-by-step decision logic",
    "summary": "Final extraction summary"
  },
  "extracted_values": [
    {
      "table_id": "pdf_table_1_0",
      "page": 1,
      "value": "42.5",
      "row_headers": ["Revenue"],
      "column_headers": ["Q1 2024"],
      "row_index": 2,
      "column_index": 1
    }
  ],
  "conversation_log": ["Step-by-step processing log"]
}
```

## Key Features

### 🧠 **LLM-Powered Intelligence**
- Smart decision making based on extraction quality
- Natural language analysis and reasoning
- Adaptive workflow based on document characteristics

### 🔄 **Robust Fallback System** 
- PDF → OCR → Error handling chain
- Multiple OCR configurations for difficult documents
- Graceful degradation with informative error messages

### 📊 **Rich Metadata**
- Cell-level positioning and header information
- Confidence scores at multiple levels
- Processing statistics and quality metrics

### 🛠️ **Tool Integration**
- Specialized analysis tools callable by LLM
- Extensible architecture for additional tools
- Function calling with structured outputs

## Usage

### Basic Usage
```python
from main_agent import EnhancedMasterAgent

# Initialize with default LLM
agent = EnhancedMasterAgent()

# Extract tables
results = agent.extract_tables("document.pdf")

# Access extracted data
for value in results["extracted_values"]:
    print(f"{value['value']} - Headers: {value['row_headers']}, {value['column_headers']}")
```

### Custom LLM Configuration
```python
# Use different model
agent = EnhancedMasterAgent(model_name="gpt2")

# Or lightweight option
agent = EnhancedMasterAgent(model_name="distilbert/distilgpt2")
```

## Model Options

| Model | Size | Description | Best For |
|-------|------|-------------|----------|
| `microsoft/DialoGPT-small` | ~500MB | Conversational model | General use |
| `distilbert/distilgpt2` | ~300MB | Very lightweight | Resource-constrained |
| `gpt2` | ~500MB | Standard GPT-2 | Balanced performance |

## Error Handling

The system includes comprehensive error handling:
- **LLM Fallback**: Rule-based decisions if LLM fails
- **OCR Fallback**: Automatic retry with different methods
- **Graceful Degradation**: Partial results rather than complete failure
- **Detailed Logging**: Full error context and recovery attempts

## Performance Considerations

- **Memory Usage**: ~1-2GB RAM (depending on model choice)
- **Processing Time**: 5-30 seconds per document
- **GPU Acceleration**: Automatic if CUDA available
- **Batch Processing**: Single document focus for accuracy

## Dependencies

**Core Requirements:**
- `transformers` >= 4.36.0
- `torch` >= 2.0.0  
- `langchain-huggingface` >= 0.0.3
- `langgraph` >= 0.0.40

**Existing Dependencies:**
- `pdfplumber`, `pytesseract`, `opencv-python`, `pdf2image`

## Installation

```bash
pip install transformers torch langchain-huggingface accelerate langgraph
```

## Future Enhancements

- **Multi-document batch processing**
- **Custom model fine-tuning**
- **Advanced table relationship detection**
- **Export format options (CSV, Excel, etc.)**
- **Web interface integration**
