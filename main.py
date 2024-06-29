from didact.docs import Docs
from didact.llms.gemini import GeminiLLM
from didact.config import settings


llm = GeminiLLM(settings.GEMINI_API_KEY)

docs = Docs(llm)

docs.add("./JBPE-8-127.pdf")
print(f"Embedded {len(docs.store.texts)} chunks")

response = docs.query("How much is the fat content in starved fish?")

print(response)
