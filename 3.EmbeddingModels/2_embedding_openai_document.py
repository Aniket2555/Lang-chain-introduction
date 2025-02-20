from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

model = OpenAIEmbeddings(model='text-embedding-3-small',dimensions=32)

document = [
    'I am Guts',
    'I love casca',
    'I will kill griffith'
]

vector = model.embed_documents(document)

print(str(vector))