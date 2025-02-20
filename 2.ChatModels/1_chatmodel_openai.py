from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model1 = ChatOpenAI(model='gpt-4o')
model2 = ChatOpenAI(model='gpt-4o',temperature=1.5,max_completion_tokens=10)

result1 = model1.invoke("what is reinforcement learning")
result2 = model2.invoke("what is reinforcement learning")

print(result1.content)
print(result2.content)