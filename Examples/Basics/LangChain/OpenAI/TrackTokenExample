from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9)

text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."

with get_openai_callback() as cb:
    result = llm(text)
    print(cb)