from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

document = [
    'I am Guts',
    'I love casca',
    'I will kill griffith'
]

vector = model.embed_documents(document)

print(vector)
print(len(vector))