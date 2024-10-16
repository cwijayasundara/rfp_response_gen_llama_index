import warnings
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex

warnings.filterwarnings('ignore')
_ = load_dotenv()

persist_dir = "storage_policy_chroma"

summary_llm = OpenAI(model="gpt-4o-mini")

vector_store = ChromaVectorStore.from_params(
    collection_name="policy_docs",
    persist_dir=persist_dir
)

index = VectorStoreIndex.from_vector_store(vector_store)

# Query Data
query_engine = index.as_query_engine()
response = query_engine.query("Whats the cash back amount for dental expenses?")
print(response)


