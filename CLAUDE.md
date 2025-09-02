# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) evaluation framework built with:
- **Ragas**: For RAG evaluation metrics (faithfulness, answer_relevancy, context_precision, context_recall, etc.)
- **Langfuse**: For tracing and observability 
- **Multi-LLM Support**: OpenAI, Google Gemini, Anthropic Claude, and Ollama for local models
- **SmartRAG Integration**: Evaluate SmartRAG traces via Langfuse API
- **Manual CSV Evaluation**: Evaluate RAG systems using reference datasets

The project evaluates RAG systems using reference datasets and outputs detailed metrics in CSV and JSON formats.

## Project Structure

```
ragas/
├── src/                      # Source code
│   ├── evaluation/          # Evaluation modules
│   │   └── smartrag_evaluator.py
│   ├── utils/              # Utilities
│   └── config/             # Configuration
├── data/                    # Data files
│   ├── reference/          # Reference Q&A datasets
│   └── results/            # Evaluation results
├── notebooks/              # Jupyter notebooks
├── evaluate.py             # Main entry point
├── .env                    # Environment configuration
└── .env.template           # Configuration template
```

## Key Files

**Original Evaluation (Direct RAG Pipeline):**
- `POC_Eval_Ragas_Langfuse.ipynb`: Main evaluation notebook containing the complete workflow
- `Template_gold_POCEval (1).xlsx`: Excel template with gold standard dataset structure
  - **JEU_OR** sheet: Questions and reference answers
  - **SOURCES** sheet: Source documents/contexts
  - **SORTIE_EVALUATIONS** sheet: Generated evaluation results

**SmartRAG Integration (Langfuse Traces):**
- `SmartRAG_Langfuse_Eval.ipynb`: Evaluation of SmartRAG traces stored in Langfuse
- `reference_qa_template.csv`: CSV template for questions/reference answers
- Output files: `smartrag_evaluation_results.csv` and `*_detailed.json`

## Development Commands

### Running the Evaluation

**Direct RAG Pipeline Evaluation:**
```bash
# Start Jupyter notebook for direct evaluation
jupyter notebook POC_Eval_Ragas_Langfuse.ipynb
```

**SmartRAG + Langfuse Evaluation:**
```bash
# Start Jupyter notebook for SmartRAG trace evaluation
jupyter notebook SmartRAG_Langfuse_Eval.ipynb

# Or use Jupyter Lab
jupyter lab SmartRAG_Langfuse_Eval.ipynb
```

### Installing Dependencies
```bash
pip install --upgrade pip
pip install ragas==0.2.6 langfuse==2.46.7 pydantic>=2.7.0 pandas openpyxl xlsxwriter matplotlib seaborn python-dotenv requests
```

### Setting up Ollama
```bash
# Install Ollama (see https://ollama.ai)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (example)
ollama pull llama3.2
ollama pull mistral
```

## Environment Configuration

**Direct RAG Pipeline Evaluation:**

- `EVAL_INPUT_XLSX`: Path to input Excel file with gold dataset
- `EVAL_OUTPUT_XLSX`: Path to output Excel file for results
- `MODEL_PROVIDER`: `bedrock` | `openai` | `custom` | `ollama`
- `BATCH_SIZE`: Batch size for inference (default: 16)

**Ollama Configuration:**
- `OLLAMA_BASE_URL`: Ollama server URL (default: http://localhost:11434)
- `OLLAMA_MODEL`: Model name (e.g., llama3.2, mistral)
- `OLLAMA_TIMEOUT`: Request timeout in seconds (default: 120)
- `OLLAMA_NUM_PREDICT`: Max tokens to generate (default: 512)
- `OLLAMA_TEMPERATURE`: Generation temperature (default: 0.7)
- `OLLAMA_TOP_P`: Top-p sampling (default: 0.9)
- `OLLAMA_TOP_K`: Top-k sampling (default: 40)

**SmartRAG + Langfuse Evaluation:**

- `LANGFUSE_PUBLIC_KEY`: Langfuse public key (required)
- `LANGFUSE_SECRET_KEY`: Langfuse secret key (required)
- `LANGFUSE_BASE_URL`: Langfuse instance URL (default: https://cloud.langfuse.com)
- `REFERENCE_CSV`: Path to CSV file with reference Q&A (default: ./reference_qa.csv)
- `OUTPUT_CSV`: Output file for evaluation results (default: ./smartrag_evaluation_results.csv)
- `SMARTRAG_PROJECT_NAME`: Filter traces by SmartRAG project name (optional)
- `EVALUATION_TIMERANGE`: Evaluation period in days (default: 7)
- `MIN_CONFIDENCE_SCORE`: Minimum confidence threshold (default: 0.0)
- `INCLUDE_FAILED_TRACES`: Include failed traces in evaluation (default: false)

## Architecture

### Core Workflows

**Direct RAG Pipeline Evaluation:**
1. **Data Loading**: Reads gold standard from Excel (`JEU_OR` and `SOURCES` sheets)
2. **RAG Pipeline**: Placeholder functions for retrieval and generation (needs customization)
3. **Evaluation**: Uses Ragas metrics to evaluate RAG outputs
4. **Export**: Writes detailed results back to Excel
5. **Tracing**: Optional Langfuse integration for observability

**SmartRAG + Langfuse Evaluation:**
1. **Trace Retrieval**: Fetches SmartRAG traces from Langfuse API
2. **Data Processing**: Extracts questions, answers, and contexts from traces
3. **Question Matching**: Matches traces to reference questions using similarity
4. **Ragas Evaluation**: Calculates quality metrics on matched pairs
5. **Analysis & Export**: Generates visualizations and exports detailed results

### Key Functions to Customize
- `dummy_retrieve_contexts()`: Replace with actual retrieval logic
- `generate_answer()`: Replace with actual RAG generation pipeline

### Evaluation Metrics
- **faithfulness**: How grounded the answer is in the retrieved context
- **answer_relevancy**: How relevant the answer is to the question
- **context_precision**: Precision of retrieved contexts
- **context_recall**: Recall of retrieved contexts
- **context_entities_recall**: Entity-level recall in contexts
- **noise_sensitivity**: Robustness to noise in contexts

## Data Formats

### Direct RAG Pipeline - Excel Structure
**JEU_OR** sheet must contain:
- `id`: Unique identifier
- `question`: The question to evaluate
- `reponse_reference`: Ground truth answer
- `contexte_attendu`: Expected context (optional)

**SOURCES** sheet contains reference documents/contexts

**SORTIE_EVALUATIONS** sheet includes all evaluation results with:
- Original question/answer pairs
- Model-generated responses
- Retrieved contexts
- All Ragas metric scores
- Performance metadata (latency, tokens, cost estimates)

### SmartRAG + Langfuse - CSV Structure
**reference_qa_template.csv** must contain:
- `question_id`: Unique identifier for each question
- `question`: The reference question text
- `reference_answer`: Expected/correct answer
- `expected_contexts`: Expected contexts (optional, separated by '|||')
- `category`: Question category (optional, for analysis)
- `difficulty`: Difficulty level (optional, for analysis)

**Output files:**
- `smartrag_evaluation_results.csv`: Tabular results with all metrics
- `*_detailed.json`: Complete evaluation data including metadata and trace information

## Model Integration

The notebook includes connectors for:
- **AWS Bedrock**: For AWS-hosted models (placeholder)
- **OpenAI**: For OpenAI API models (placeholder)
- **Custom endpoints**: For internal/proprietary models (placeholder)
- **Ollama**: For local LLM inference (fully implemented)

### Ollama Integration
The Ollama connector is fully functional and includes:
- HTTP API calls to local Ollama instance
- Retry logic with exponential backoff
- Comprehensive error handling
- Token counting and performance metrics
- Configurable generation parameters

Replace the placeholder functions with actual implementations for your specific RAG pipeline.

### SmartRAG Integration
The SmartRAG evaluation workflow automatically:
- Connects to Langfuse to retrieve SmartRAG traces
- Extracts RAG components (questions, answers, contexts) from trace observations
- Matches traces to reference questions using text similarity
- Supports filtering by project name, time range, and confidence scores
- Provides comprehensive analysis including performance metrics and visualizations

## French Language Support

The notebook and documentation are primarily in French, reflecting the target use case. Key terms:
- `jeu d'or`: Gold standard dataset
- `reponse_reference`: Reference/ground truth answer
- `reponse_modele`: Model-generated answer
- `contexte_attendu`: Expected context