from time import perf_counter

from didact.docs import Docs
from didact.llms.gemini import GeminiLLM
from didact.config import settings
from didact.stores import SupabaseVectorStore


llm = GeminiLLM(settings.GEMINI_API_KEY)

store = SupabaseVectorStore(
    settings.SUPABASE_URL,
    settings.SUPABASE_SERVICE_KEY,
)

docs = Docs(llm, store)

if __name__ == "__main__":
    start = perf_counter()
    # docs.add("../paper-qa/sample_papers/14_74.pdf")
    # response = docs.query("What is the architecture for a resnet model?", verbose=True)
    response = docs.query(
        "What approaches convert k-space of low field MRI to high resolution images",
        # "What is inverse crime in the context of image upscaling?",
        # "What are the downsides of RAG and the mitigations?",
        # "What is the resnet model meant to solve?",
        k=5,
        fetch_k=10,
        summarize=True, use_ask_llm=True, verbose=True,
    )
    print(response)
    print(f"Duration: {perf_counter() - start}s")
