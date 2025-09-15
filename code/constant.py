import os
from dotenv import load_dotenv
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

AZURE_OPENAI_API_EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_API_EMBEDDING_MODEL")
AZURE_OPENAI_API_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_API_EMBEDDING_DEPLOYMENT_NAME")

OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


persist_directory = "./persist_files/hcp_chroma_db"
csv_file_path = 'hcp_dashboard_data.csv'


PROMPT_TEMPLATE_QA = """You are a helpful assistant supporting sales representatives by providing accurate and relevant answers based on the provided data for the question asked.

**Question:** {question}

Use only the information from the vector search context to answer. Do not assume or fabricate any details.

- Extract exact numbers, names, or facts directly from the context.
- If the answer requires calculations (e.g., counting HCPs by specialty), perform them using the data provided.
- If the information is not available or cannot be derived, respond with:  
  **"I'm unable to provide a definitive answer based on the current data context."**

Make no mistakes while providing or calculating the answer.
**Vector Search Context:**  
{vector_context}

Answer:
"""

PROMPT_TEMPLATE_INSIGHT = """You are a helpful assistant providing additional insights based on the data context and the question asked by the sales representative.

**User Question:** {question}

Based only on the vector search context, provide 3-5 concise, factual insights, trends, or correlations. Avoid assumptions or speculation.

- Focus on patterns, comparisons, or notable metrics.
- If no insights can be derived, respond with:  
  **"I cannot provide any additional insights/information for this question at the moment."**

**Vector Search Context:**  
{vector_context}

Insights:
"""