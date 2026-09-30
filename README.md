# 🤖 LangGraph RAG Assistant

A Retrieval-Augmented Generation (RAG) based AI assistant built using **LangGraph, LangChain, ChromaDB, BM25, Groq LLM, Tool Calling, Structured Output, Streamlit, and MemorySaver**.

This project combines semantic search and keyword-based search to retrieve relevant information and uses a LangGraph workflow to generate the final response.

---

## 🚀 Features

- 🔍 Retrieval-Augmented Generation (RAG)
- 🧠 LangGraph-based workflow
- 📚 PDF document processing
- 🔎 ChromaDB semantic search
- 🔤 BM25 keyword search
- 🔗 Hybrid Retrieval using Reciprocal Rank Fusion (RRF)
- 🛠️ LLM Tool Calling
- 📦 Structured Output
- 💾 Conversation memory using LangGraph MemorySaver
- ⚡ Groq LLM integration
- 🌐 Streamlit web application
- 🐍 Python-based implementation

---

## 🏗️ Project Architecture

```text
                    User Question
                          │
                          ▼
                  ┌───────────────┐
                  │   Retrieval   │
                  │               │
                  │   ChromaDB    │
                  │      +        │
                  │     BM25      │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │      LLM      │
                  │   + Tools     │
                  └───────┬───────┘
                          │
                    Tool Required?
                       /       \
                     Yes        No
                      │          │
                      ▼          ▼
                ┌──────────┐  ┌──────────────┐
                │ ToolNode │  │  Structured  │
                │          │  │    Output    │
                └────┬─────┘  └──────┬───────┘
                     │               │
                     └───────┬───────┘
                             ▼
                       Final Answer
                             │
                             ▼
                        MemorySaver
```

---

## 🔄 LangGraph Workflow

The application uses LangGraph to control the complete RAG and tool-calling workflow.

```text
START
  │
  ▼
Retrieval
  │
  ▼
LLM + Tools
  │
  ├──────────────► Tool Required
  │                     │
  │                     ▼
  │                  ToolNode
  │                     │
  │                     ▼
  │                    LLM
  │
  └──────────────► No Tool Required
                        │
                        ▼
                 Structured Output
                        │
                        ▼
                       END
```

---

## 🔎 Hybrid Retrieval

The project uses a hybrid retrieval system that combines two retrieval techniques.

### 1. ChromaDB

ChromaDB is used for semantic similarity search using embeddings.

### 2. BM25

BM25 is used for keyword-based retrieval.

### 3. Reciprocal Rank Fusion (RRF)

The results from ChromaDB and BM25 are combined using Reciprocal Rank Fusion.

```text
                User Query
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
      ChromaDB               BM25
   Semantic Search       Keyword Search
          │                   │
          └─────────┬─────────┘
                    ▼
                   RRF
                    │
                    ▼
          Top Relevant Context
                    │
                    ▼
                   LLM
```

This approach combines semantic relevance with keyword matching.

---

## 🛠️ Tool Calling

The LLM can decide whether an available tool is required to answer a question.

```text
User Question
      │
      ▼
     LLM
      │
      ▼
Tool Required?
   │        │
  Yes       No
   │         │
   ▼         ▼
ToolNode   Structured
   │         Output
   ▼
Tool Result
   │
   ▼
   LLM
```

If a tool is required, the tool is executed and the result is returned to the LLM before generating the final response.

---

## 📦 Structured Output

The final response is generated using a structured output schema.

Example:

```python
class Answer(BaseModel):
    answer: str
    summary: str
    sources: list[str]
```

Structured output makes the final response predictable and easier to process in the application.

---

## 💾 Conversation Memory

This project uses LangGraph's in-memory checkpoint system for maintaining conversation state.

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

build = graph.compile(
    checkpointer=memory
)
```

A unique `thread_id` is used for each conversation:

```python
config = {
    "configurable": {
        "thread_id": "user_001"
    }
}
```

Using the same `thread_id` allows the application to continue the conversation within the running application.

> Note: `MemorySaver` stores checkpoints in memory. The memory is lost when the application process is restarted.

---

## 🌐 Streamlit Interface

The project includes a Streamlit-based chat interface.

The application provides:

- 💬 Chat interface
- 🧠 LangGraph workflow execution
- 🔍 RAG-based retrieval
- 🛠️ Tool calling
- 📦 Structured responses
- 💾 Conversation memory
- 🗑️ Clear chat functionality

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming language |
| LangGraph | Workflow orchestration |
| LangChain | LLM and tool integration |
| LangChain Community | Document loaders and integrations |
| Groq | LLM provider |
| ChromaDB | Vector database |
| Sentence Transformers | Text embeddings |
| BM25 | Keyword-based retrieval |
| PyPDF | PDF processing |
| Streamlit | Web interface |
| MemorySaver | Conversation checkpointing |
| Pydantic | Structured output |

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repository.git
```

```bash
cd your-repository
```

---

### 2. Create Virtual Environment

This project uses `uv`.

```bash
uv venv
```

Activate the virtual environment on Windows:

```powershell
.venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
uv add langgraph langchain-community sentence-transformers python-dotenv pypdf langchain-groq langchain-text-splitters chromadb rank-bm25 streamlit
```

For Jupyter Notebook:

```bash
uv add --dev ipykernel
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Load the environment variables:

```python
from dotenv import load_dotenv

load_dotenv()
```

> ⚠️ Never upload your `.env` file or API keys to GitHub.

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
uv run streamlit run app.py
```

After running the command, open the Streamlit URL shown in the terminal.

---

## 📁 Project Structure

```text
LangGraph-Project/
│
├── app.py
├── langgraph_project(4).ipynb
├── pyproject.toml
├── uv.lock
├── README.md
├── .gitignore
│
└── .venv/
```

---

## 🔒 .gitignore

Recommended `.gitignore`:

```gitignore
.venv/
__pycache__/
*.pyc

.env
.env.*

.ipynb_checkpoints/

.pytest_cache/
.mypy_cache/
.ruff_cache/
```

---

## 🧪 Example Questions

You can ask questions related to the documents stored in the RAG knowledge base.

```text
What is the capital of India?
```

```text
Tell me about India's geography.
```

```text
What information is available about Indian sports?
```

The system retrieves relevant context before generating the final response.

---

## 🎯 Project Goals

This project was created to learn and demonstrate:

- Retrieval-Augmented Generation
- Vector databases
- Hybrid search
- BM25 retrieval
- LangGraph state management
- Tool calling
- Structured output
- Conversation memory
- Streamlit application development

---

## 🔮 Future Improvements

Possible future improvements include:

- Add more tools
- Add more document sources
- Improve retrieval accuracy
- Add conversation summarization
- Add user authentication
- Add persistent database-based memory
- Improve Streamlit UI
- Add document upload functionality
- Add chat history management

---

## 👨‍💻 Author

**Sayan Mete**

Computer Science Student | AI & Machine Learning Enthusiast

---

## ⭐ Acknowledgement

This project was built for learning and experimenting with modern AI application development using **RAG, LangGraph, LLM Tool Calling, Structured Output, and Conversation Memory**.