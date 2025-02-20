from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = GoogleGenerativeAI(model='gemini-1.5-pro')

result = model.invoke('what is reinforcement learning')

print(result.invoke)