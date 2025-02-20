from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

model = OpenAIEmbeddings(model='text-embedding-3-small',dimensions=32)

text = 'My name is Guts'

vector = model.embed_query(text)
print(vector)
print(len(vector))