# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

document = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'who is jasprit Bumarh'

# model = OpenAIEmbeddings(model='text-embedding-3-small',dimensions=100)
model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

document_embedding = model.embed_documents(document)
query_embedding = model.embed_query(query)

score = cosine_similarity([query_embedding],document_embedding)[0]

index,score = sorted(list(enumerate(score)),key=lambda x:x[1])[-1]

print(query)
print(document[index])
print(f'Cosine similiarity is: {score}')
