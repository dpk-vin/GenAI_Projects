# ShopAssist Semantic Spotter - Complete Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Project Goals](#project-goals)
4. [System Architecture](#system-architecture)
5. [Data Sources](#data-sources)
6. [Design Choices](#design-choices)
7. [Technology Stack](#technology-stack)
8. [Installation & Setup](#installation--setup)
9. [Usage Guide](#usage-guide)
10. [Challenges & Solutions](#challenges--solutions)
11. [System Flowchart](#system-flowchart)
12. [API Reference](#api-reference)

---

## Project Overview

**ShopAssist Semantic Spotter** is an end-to-end generative AI project built with LangChain that transforms e-commerce product discovery and data analysis. The system combines:

- **Content-Based Product Recommendations**: Semantic similarity search over 5,000 products
- **Conversational Shopping Assistant**: Natural language interface for product queries
- **Pandas Data Analytics Agent**: Natural language questions answered from sales data
- **SQL Database Agent**: Complex SQL queries generated and executed automatically

The project demonstrates how LLMs can be used as reasoning engines to select and orchestrate tools (vector stores, DataFrames, databases) without hand-coded logic.

---

## Problem Statement

E-commerce platforms face several challenges:

1. **Product Discovery Friction**: Users struggle to find products matching their preferences using traditional keyword searches or category filters
2. **Data Accessibility Gap**: Non-technical stakeholders cannot perform ad-hoc sales analysis without SQL/Python skills
3. **Limited Context Understanding**: Traditional rule-based systems fail to understand nuanced user preferences expressed in natural language
4. **Scalability of Support**: Customer service requires human intervention for common product inquiries and analytics questions
5. **Multi-Format Data**: Businesses manage data across products catalogs, sales records, and relational databases without unified access

**Solution**: Semantic Spotter uses LLMs as intelligent agents that understand context and select appropriate tools dynamically.

---

## Project Goals

### Primary Objectives
1. Build a **semantic product recommendation system** using content-based filtering with embeddings
2. Create a **conversational agent** that engages users in natural language shopping interactions
3. Implement a **data analytics agent** that converts natural language queries into insights (Pandas-based)
4. Develop an **SQL agent** for complex database queries across multiple tables
5. Demonstrate **LangChain best practices** for agent design without deprecated APIs
6. Provide **comprehensive documentation** for reproducibility and extension

### Learning Outcomes
- Understand how embeddings enable semantic search at scale
- Master LangChain's agent toolkits for Pandas and SQL
- Learn to compose LLMs with external tools (retrievers, DataFrames, databases)
- Implement production-ready conversational interfaces
- Design modular, maintainable AI system architecture

---

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
│            (Console Chatbot / Interactive Menu)              │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴──────────────┬────────────────┐
        │                           │                │
┌───────▼──────────┐    ┌──────────▼──────┐  ┌─────▼─────────┐
│  ShopAssist Chat │    │ Analytics Agent  │  │  SQL Agent    │
│  (Conversational)│    │  (Pandas-based)  │  │  (Database)   │
└───────┬──────────┘    └──────────┬──────┘  └─────┬─────────┘
        │                         │              │
   ┌────▼──────────┐      ┌───────▼────┐   ┌──────▼──────┐
   │ Tool Selector │      │ DataFrame  │   │ SQLAlchemy  │
   │               │      │ Analysis   │   │ SQLDatabase │
   └────┬──────────┘      └────────────┘   └─────────────┘
        │
   ┌────▼──────────────────────────────┐
   │     Data Layer                     │
   ├────────────────────────────────────┤
   │ • FAISS Vector Store (5000 prod)   │
   │ • Pandas DataFrame (5000 sales)    │
   │ • SQLite DB (products + sales)     │
   │ • products.csv / sales.csv         │
   └────────────────────────────────────┘
```

### Module Breakdown

| Module | Purpose | Key Dependencies |
|--------|---------|------------------|
| **Vector Store** | Semantic product search | FAISS, OpenAIEmbeddings |
| **Recommender Tool** | Content-based recommendations | Vector Store Retriever |
| **Product Info Tool** | Exact product lookups | Pandas DataFrame |
| **ShopAssist Core** | LLM-based routing logic | ChatOpenAI, Tool Functions |
| **Pandas Agent** | Sales data analytics | create_pandas_dataframe_agent |
| **SQL Agent** | Database queries | create_sql_agent, SQLAlchemy |
| **Chatbot Loop** | User interaction | All agents |

---

## Data Sources

### 1. Products Dataset (`products.csv`)
**Format**: CSV with 5,000 rows

**Schema**:
```
id,name,category,description,price,tags
1,Electronics Item 1,Electronics,"A high-quality electronics product...",199.99,"wireless,bluetooth,gaming"
...
```

**Characteristics**:
- **5,000 unique products** across 5 categories
- Categories: Electronics, Sportswear, Home, Books, Beauty
- Price range: $10 - $300
- Semantic-rich descriptions for embeddings
- Multiple tags per product for flexible matching

### 2. Sales Dataset (`sales.csv`)
**Format**: CSV with 5,000 rows

**Schema**:
```
order_id,product_id,quantity,price,category
1,1694,1,223.59,Home
...
```

**Characteristics**:
- **5,000 sales transactions**
- Links to product_id for joins
- Quantity and price per transaction
- Category denormalization for quick aggregations
- Enables revenue calculations, category analysis, etc.

### 3. SQLite Database (`shop.db`)
**Format**: Relational database

**Tables**:
- `products` (5000 rows): id, name, category, description, price, tags
- `sales` (5000 rows): order_id, product_id, quantity, price, category

**Usage**: SQL agent queries this DB to answer complex questions like "top 3 expensive products per category"

### Data Generation Notes
- **Reproducibility**: Using `np.random.default_rng(42)` seed for deterministic generation
- **Realism**: Synthetic but realistic product names and price distributions
- **Scalability**: Easy to increase NUM_PRODUCTS and NUM_SALES in generation code

---

## Design Choices

### 1. **Content-Based Recommendations over Collaborative Filtering**

**Choice**: Use semantic embeddings + FAISS vector store

**Rationale**:
- **Cold start problem solved**: No need for user history; works immediately for new users
- **Semantic understanding**: Handles natural language queries like "gaming headphones" without exact keyword match
- **Explainability**: Can show similar products in embedding space
- **Scalability**: FAISS indexes 5000 products in milliseconds

**Alternative Considered**: Collaborative filtering would need user interaction history, unsuitable for new users.

### 2. **Manual Routing vs. Classic Agents**

**Choice**: Use LLM + simple if-then logic instead of `initialize_agent` (deprecated)

**Rationale**:
- **Stability**: Avoid deprecated APIs that break between LangChain versions
- **Transparency**: Routing logic is explicit and debuggable
- **Control**: Exactly know which tool is called and why
- **Maintenance**: No dependency on evolving agent abstractions

**Code Example**:
```python
if "recommend" in user_query.lower():
    use_recommender_tool()
elif known_product_name in user_query:
    use_product_info_tool()
else:
    use_generic_llm_response()
```

### 3. **LangChain Toolkits for Pandas & SQL**

**Choice**: Use `create_pandas_dataframe_agent` and `create_sql_agent` directly

**Rationale**:
- **Abstraction**: These handle prompt engineering, tool invocation, and result parsing
- **Safety**: SQL agent can be restricted to SELECT-only queries
- **Flexibility**: Easy to add constraints (max results, allowed tables, etc.)

**Example**:
```python
pandas_agent = create_pandas_dataframe_agent(llm, sales_df, verbose=True)
res = pandas_agent.invoke({"input": "What is total revenue by category?"})
```

### 4. **CSV + SQLite Dual Approach**

**Choice**: Load from CSV, then create SQLite for SQL agent

**Rationale**:
- **Simplicity**: CSV is human-readable and portable (no server dependency)
- **Performance**: SQLite provides indexed access for complex queries
- **Flexibility**: Pandas agent works on DataFrames; SQL agent on relational tables
- **Demo-friendly**: No database setup required; SQLite is file-based

### 5. **Modular Function Design**

**Choice**: Separate functions for each agent (`chat_with_shop_assist`, `ask_analytics`, `ask_sql`)

**Rationale**:
- **Reusability**: Each can be exposed as an API endpoint later
- **Testing**: Easy to unit test individual agents
- **Composition**: Simple to combine multiple agents into workflows

---

## Technology Stack

### Core LLM & Embeddings
- **LLM**: `ChatOpenAI` (gpt-4o-mini or equivalent)
- **Embeddings**: `OpenAIEmbeddings`
- **LangChain Version**: 0.2+ (modern Runnable-based APIs)

### Vector Store & Retrieval
- **Vector Store**: `FAISS` (in-memory, CPU-based)
- **Document Format**: `langchain_core.documents.Document`
- **Retriever**: FAISS similarity search with k=3

### Data & Database
- **DataFrame**: `Pandas` (for CSV loading and analytics)
- **Database**: `SQLite` with `SQLAlchemy` ORM
- **CSV Library**: Standard `pandas.read_csv` / `to_csv`

### Agents & Tools
- **Pandas Agent**: `langchain_community.agent_toolkits.create_pandas_dataframe_agent`
- **SQL Agent**: `langchain_community.agent_toolkits.create_sql_agent`
- **Tool Definition**: `langchain.tools.Tool`

### Utilities
- **Environment**: `python-dotenv` (for API key management)
- **Numeric**: `NumPy` (for data generation)

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- OpenAI API key (sign up at https://platform.openai.com/)
- 500MB disk space (for FAISS index + SQLite DB)

### Step 1: Clone or Download Project
```bash
# Create a new directory
mkdir shopAssist_project
cd shopAssist_project
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install \
  langchain \
  langchain-community \
  langchain-openai \
  faiss-cpu \
  sqlalchemy \
  pandas \
  numpy \
  python-dotenv
```

### Step 4: Set Up Environment Variables
Create a `.env` file in your project directory:
```
OPENAI_API_KEY=sk-your-actual-key-here
```

### Step 5: Download or Generate Data
**Option A**: Download pre-generated CSVs (provided with project)
```bash
# Place products.csv and sales.csv in the project directory
ls -la products.csv sales.csv
```

**Option B**: Generate fresh data (optional)
```python
# Run data generation code (see complete_code.ipynb, cell 2)
# This creates products.csv and sales.csv with 5000 rows each
```

### Step 6: Open and Run Notebook
```bash
jupyter notebook ShopAssist_Semantic_Spotter.ipynb
```

Run cells **in order**:
1. Install & imports
2. Load from CSV + create SQLite DB
3. Content-based recommender
4. ShopAssist conversational agent
5. Pandas analytics agent
6. SQL agent
7. Chatbot loop

---

## Usage Guide

### Option 1: Interactive Chatbot Mode

At the end of the notebook, run:
```python
run_chatbot()
```

This starts an interactive loop:
```
=== ShopAssist Chatbot ===
Ask me anything about products, recommendations, or sales.
Type 'exit' to quit.

You: I need running shoes for jogging
ShopAssist: Here are some recommended products:
- Sportswear Item 234 (Category: Sportswear, Price: $79.99)
- Sportswear Item 567 (Category: Sportswear, Price: $89.99)
- Sportswear Item 891 (Category: Sportswear, Price: $74.99)

You: What is the total revenue by category?
ShopAssist: [Pandas agent processes sales.csv and responds with analysis]

You: exit
ShopAssist: Goodbye!
```

### Option 2: Direct Function Calls

Call agents individually:

```python
# 1. Product recommendations
chat_with_shop_assist("I need wireless headphones under $150")

# 2. Product details
chat_with_shop_assist("Tell me about Electronics Item 42")

# 3. Sales analytics
ask_analytics("Which category has the highest average price?")

# 4. Database queries
ask_sql("List all products in the Beauty category with prices > $100")
```

### Option 3: Menu-Driven Interface

Run the demo menu:
```python
run_demo()
```

This presents a numbered menu:
```
=== ShopAssist Semantic Spotter Demo ===
1. Conversational Shopping Assistant
2. Data Analytics Agent (Pandas)
3. SQL Agent
Choose mode (1/2/3): 
```

---

## Challenges & Solutions

### Challenge 1: Deprecated LangChain APIs

**Problem**: `initialize_agent` and `AgentExecutor` were removed between LangChain v0.1 and v1.0

**Solution**:
- Avoided using `initialize_agent`; instead used simple manual routing with `if-elif-else` logic
- Used `create_pandas_dataframe_agent` and `create_sql_agent` directly (these are stable)
- Updated all `.run()` calls to `.invoke({"input": query})` pattern for new Runnable interface

**Code Snippet**:
```python
# Old (deprecated)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
result = agent.run(query)

# New (current)
if "recommend" in query.lower():
    result = llm.invoke(prompt_with_recommendations).content
```

### Challenge 2: Prompt Engineering for Tool Selection

**Problem**: LLM sometimes generates answers without calling available tools

**Solution**:
- Explicitly list keywords that trigger tool usage ("recommend", "suggest", "looking for", "find me")
- Include tool descriptions in system prompts
- Test with multiple query variations during development

### Challenge 3: Handling Large Datasets in Vector Search

**Problem**: Scanning 5000 product names in `shopassist_core()` is inefficient

**Solution**:
- Limit product name scanning to first 500 rows: `products_df["name"].head(500)`
- For full catalog search, would use semantic similarity in production (already implemented)
- Use FAISS index directly for exact-match queries

### Challenge 4: Managing CSV/SQLite Sync

**Problem**: Data in CSV and SQLite DB could diverge if not careful

**Solution**:
- Load from CSV once: `products_df = pd.read_csv("products.csv")`
- Create SQLite directly from loaded DataFrames: `products_df.to_sql(...)`
- Use same DataFrames for both Pandas agent and FAISS indexing
- Single source of truth: CSV files

### Challenge 5: OpenAI API Rate Limits & Costs

**Problem**: High API call volume during development could exceed quota

**Solution**:
- Use `gpt-4o-mini` model (lowest cost tier)
- Set `temperature=0.3` for faster, more deterministic responses
- Cache prompts where possible
- Test with small data samples before scaling
- Monitor costs via OpenAI dashboard

### Challenge 6: FAISS Index Rebuilding

**Problem**: Adding new products requires recreating entire FAISS index

**Solution**:
- For this demo: acceptable to rebuild on startup
- Production approach: Use persistent vector DB (Pinecone, Weaviate) with incremental indexing
- For now, index is rebuilt in milliseconds with 5000 products

---

## System Flowchart

### High-Level Flow

```
                          START
                            │
                            ▼
                    ┌──────────────────┐
                    │  Load CSV Files  │
                    │  (products.csv,  │
                    │   sales.csv)     │
                    └────────┬─────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
        ┌───────▼──────────┐    ┌────────▼─────────┐
        │  Initialize LLM  │    │  Create SQLite   │
        │ & Embeddings     │    │  Database (shop) │
        └────────┬─────────┘    └────────┬─────────┘
                 │                       │
        ┌────────▼──────────┐            │
        │  Build FAISS      │            │
        │  Vector Index     │            │
        │ (5000 products)   │            │
        └────────┬──────────┘            │
                 │                       │
                 └──────────┬────────────┘
                            │
                            ▼
                    ┌──────────────────┐
                    │  Start Chatbot   │
                    │  (run_chatbot)   │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Read User Query  │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
         ┌────▼───┐   ┌─────▼─────┐  ┌────▼─────┐
         │"recom-│   │ Analytics │  │ Database │
         │ mend"?│   │ question? │  │ question?│
         │(Y/N)  │   │(Y/N)      │  │(Y/N)     │
         └────┬──┘   └─────┬─────┘  └────┬─────┘
          Y/  │N       Y/  │N       Y/   │N
         /    │        /   │       /     │
     ┌──▼─┐ ┌─▼──┐  ┌─▼─┐┌──▼──┐ ┌──▼─┐┌──▼──┐
     │USE │ │SKP │  │USE││SKP  │ │USE││SKP  │
     │REC │ │    │  │PAN││     │ │SQL││     │
     │    │ │    │  │DA │      │ │AG │      │
     └──┬─┘ └─┬──┘  └─┬─┘└──┬──┘ └──┬─┘└──┬──┘
        │    │      │    │     │     │
        └────┴──────┴────┴─────┴─────┘
                    │
            ┌───────▼────────┐
            │  LLM Processes │
            │  Tool Output   │
            │  + Generates   │
            │  Response      │
            └───────┬────────┘
                    │
            ┌───────▼────────┐
            │ Display Answer │
            │  to User       │
            └───────┬────────┘
                    │
            ┌───────▼────────┐
            │ Ask Another?   │
            └───────┬────────┘
              Y/    │N
             /      \
          LOOP      END
           │
           └────────→ EXIT
```

### Component Interaction Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                       User (Console)                           │
└──────────────────────────┬─────────────────────────────────────┘
                           │
                    Input Query String
                           │
              ┌────────────▼───────────────┐
              │   shopassist_core()        │
              │   (Manual Tool Routing)    │
              └──┬──┬──────────────┬───────┘
                 │  │              │
        ┌────────┘  │              └─────────┐
        │           │                        │
   ┌────▼──┐   ┌────▼──┐   ┌──────────┐  ┌──▼────┐
   │Recomm │   │Prod   │   │ LLM      │  │Fallbk │
   │ender  │   │Info   │   │ Generic  │  │ Query │
   │Tool   │   │Tool   │   │Response  │  │Handler│
   └────┬──┘   └────┬──┘   └────┬─────┘  └──┬────┘
        │            │           │           │
        │      ┌──────┴───────────┴──────────┴─────┐
        │      │                                   │
   ┌────▼──────▼──────┐               ┌───────────▼──┐
   │ FAISS Vector     │               │ ChatOpenAI   │
   │ Store (Retriever)│               │ (LLM)        │
   └────┬─────────────┘               └───────────┬──┘
        │                                         │
        │              LLM Output                 │
        │                                         │
        └─────────────────┬───────────────────────┘
                          │
                    Response String
                          │
                          ▼
              Display to User (console)
```

---

## API Reference

### Main Functions

#### `chat_with_shop_assist(user_query: str) -> str`
**Purpose**: Conversational shopping assistant with semantic tool routing

**Parameters**:
- `user_query` (str): Natural language question from user

**Returns**: String response from ShopAssist agent

**Example**:
```python
response = chat_with_shop_assist("I need wireless headphones under $150")
print(response)
# Output: "Here are some recommended products: ..."
```

#### `ask_analytics(question: str) -> str`
**Purpose**: Data analytics over sales DataFrame using LLM reasoning

**Parameters**:
- `question` (str): Natural language analytics question

**Returns**: String with analysis results

**Example**:
```python
response = ask_analytics("What is the total revenue by category?")
print(response)
# Output: "Electronics: $..., Sportswear: $..., ..."
```

#### `ask_sql(question: str) -> str`
**Purpose**: SQL query generation and execution over shop.db

**Parameters**:
- `question` (str): Natural language database query

**Returns**: String with query results

**Example**:
```python
response = ask_sql("List all products in Beauty category with price > $100")
print(response)
# Output: "Found X products: ..."
```

#### `recommend_products(preference: str) -> str`
**Purpose**: Content-based product recommendation using FAISS

**Parameters**:
- `preference` (str): User preference description

**Returns**: Formatted string with top-3 recommended products

**Example**:
```python
recs = recommend_products("wireless gaming headphones")
print(recs)
# Output: "- Product A (Category: Electronics, Price: $...)"
```

#### `get_product_details(product_name: str) -> str`
**Purpose**: Exact product lookup from DataFrame

**Parameters**:
- `product_name` (str): Exact or partial product name

**Returns**: Product details (name, category, price, description, tags)

**Example**:
```python
details = get_product_details("Electronics Item 42")
print(details)
# Output: "Product: Electronics Item 42\nCategory: Electronics\n..."
```

#### `run_chatbot()`
**Purpose**: Interactive console chatbot loop

**Parameters**: None

**Returns**: None (runs forever until user types 'exit')

**Example**:
```python
run_chatbot()  # Starts interactive session
```

#### `run_demo()`
**Purpose**: Menu-driven demo interface for all agents

**Parameters**: None

**Returns**: None (runs until user selects exit)

**Example**:
```python
run_demo()  # Shows menu: 1. Chat, 2. Analytics, 3. SQL
```

### Core Objects

#### `llm: ChatOpenAI`
OpenAI GPT model instance for reasoning across all agents

#### `embeddings: OpenAIEmbeddings`
OpenAI embeddings model for semantic similarity (1536 dimensions)

#### `vector_store: FAISS`
In-memory FAISS index with 5000 product embeddings

#### `retriever: Retriever`
Vector store retriever configured to return k=3 similar products

#### `db: SQLDatabase`
SQLAlchemy SQLDatabase connection to shop.db

#### `products_df: pd.DataFrame`
Pandas DataFrame with 5000 products (loaded from CSV)

#### `sales_df: pd.DataFrame`
Pandas DataFrame with 5000 sales records (loaded from CSV)

---

## Example Queries

### Shopping Assistant Examples
```
"I need comfortable running shoes for marathon training"
"Suggest me the best wireless headphones under $150"
"Find me electronics for gaming"
"What beauty products do you have?"
"Tell me about Sportswear Item 234"
```

### Analytics Examples
```
"What is the total revenue for each category?"
"Which product_id generated the highest total revenue?"
"What is the average price of products in each category?"
"How many orders were placed?"
"What is the total quantity sold?"
```

### SQL Query Examples
```
"What are the top 5 most expensive products?"
"List all products in the Books category"
"How many products are in each category?"
"What is the average price for Electronics?"
"Which categories have average price > $150?"
```

---

## Project Structure

```
shopAssist_project/
├── ShopAssist_Semantic_Spotter.ipynb    # Main notebook with all code
├── ShopAssist_Semantic_Spotter_README.md # This file
├── PROJECT_DOCUMENTATION.md              # Detailed design documentation
├── SYSTEM_FLOWCHART.md                   # Flowchart in Mermaid format
├── products.csv                          # 5000 product records
├── sales.csv                             # 5000 sales records
├── shop.db                               # SQLite database (auto-generated)
├── .env                                  # Environment variables (OPENAI_API_KEY)
└── requirements.txt                      # Python dependencies
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'langchain_openai'"
**Solution**: `pip install -U langchain-openai`

### Issue: "OPENAI_API_KEY not set"
**Solution**: Create `.env` file with `OPENAI_API_KEY=sk-...` and run `load_dotenv()`

### Issue: "No such table: products"
**Solution**: Ensure `products.csv` exists in project directory before running cell 2

### Issue: "FAISS index too large"
**Solution**: Reduce NUM_PRODUCTS in data generation, or use GPU FAISS (`faiss-gpu`)

### Issue: "SQL Agent not responding"
**Solution**: Ensure shop.db was created correctly; check database connection string

### Issue: "Slow API responses"
**Solution**: Use `gpt-4o-mini` model; reduce temperature; batch queries

---

## Future Enhancements

1. **Persistent Vector DB**: Replace FAISS with Pinecone/Weaviate for incremental indexing
2. **Multi-turn Conversation**: Add chat history/memory for context-aware responses
3. **Web Interface**: Build Streamlit/FastAPI dashboard for non-technical users
4. **Custom Embeddings**: Fine-tune embeddings on e-commerce product descriptions
5. **A/B Testing**: Measure recommendation quality against collaborative filtering
6. **Cost Optimization**: Implement prompt caching, batch processing
7. **Monitoring**: Add logging, telemetry, cost tracking

---

## References & Resources

- **LangChain Docs**: https://python.langchain.com/
- **FAISS**: https://github.com/facebookresearch/faiss
- **OpenAI Models**: https://platform.openai.com/docs/models
- **SQLAlchemy**: https://docs.sqlalchemy.org/
- **Pandas**: https://pandas.pydata.org/docs/

---

## License & Attribution

This project is provided for educational purposes. LangChain, OpenAI, FAISS, and SQLAlchemy are used under their respective licenses.

---

## Contact & Support

For questions or issues:
1. Check troubleshooting section above
2. Review LangChain documentation
3. Check OpenAI API status
4. Verify environment variables and file paths

---

**Project Completed**: December 2025  
**Framework**: LangChain 0.2+  
**Model**: GPT-4o-mini  
**Data Scale**: 5,000 products × 5,000 sales records

## 👩‍💻 Author
**Deepika Ramesh**  
ShopAssist Semantic Spotter — 2025  
Designed for intelligent, context-aware enterprise search.
