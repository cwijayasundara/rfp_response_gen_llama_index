import warnings
from dotenv import load_dotenv
from llama_parse import LlamaParse
from pathlib import Path
from llama_index.llms.openai import OpenAI
from llama_index.core import SummaryIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex

warnings.filterwarnings('ignore')
_ = load_dotenv()

parser = LlamaParse(
    result_type="markdown",
    use_vendor_multimodal_model=True,
    vendor_multimodal_model_name="anthropic-sonnet-3.5",
)

data_dir = "data"
data_out_dir = "data_out_dir"

files = ["pb116349-business-health-select-handbook-1024-pdfa.pdf"]

file_dicts = {}

for f in files:
    file_base = Path(f).stem
    full_file_path = str(Path(data_dir) / f)
    file_docs = parser.load_data(full_file_path)

    # attach metadata
    for idx, d in enumerate(file_docs):
        d.metadata["file_path"] = f
        d.metadata["page_num"] = idx + 1
    file_dicts[f] = {"file_path": full_file_path, "docs": file_docs}

print(">> Parsing complete")

summary_llm = OpenAI(model="gpt-4o-mini")

for f in files:
    print(f">> Generate summary for file {f}")
    index = SummaryIndex(file_dicts[f]["docs"])
    response = index.as_query_engine(llm=summary_llm).query(
        "Generate a short 1-2 line summary of this file to help inform an agent on what this file is about."
    )
    print(f">> Generated summary: {str(response)}")
    file_dicts[f]["summary"] = str(response)

persist_dir = "storage_policy_chroma"

vector_store = ChromaVectorStore.from_params(
    collection_name="policy_docs",
    persist_dir=persist_dir
)

index = VectorStoreIndex.from_vector_store(vector_store)

# run this only if the Chroma index is not already built
all_nodes = [c for d in file_dicts.values() for c in d["docs"]]
index.insert_nodes(all_nodes)
