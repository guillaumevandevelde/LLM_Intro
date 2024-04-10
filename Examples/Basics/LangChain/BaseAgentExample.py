from langchain.llms import OpenAI
from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType


llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

# remember to set the environment variables
# “GOOGLE_API_KEY” and “GOOGLE_CSE_ID” to be able to use
# Google Search via API.
search = GoogleSearchAPIWrapper()

prompt = PromptTemplate(
    input_variables=["query"],
    template="Beschrijf altijd in het Nederlands, spreek de gebruiker aan met Beste burger,: {query}"
)

translate_chain = LLMChain(llm=llm, prompt=prompt)

tools = [
    Tool(
        name = "google-search",
        func=search.run,
        description="useful for when you need to search google to answer questions about current events"
    ),
        Tool(
       name='Translator',
       func=translate_chain.run,
       description='useful for translating text to Dutch and addressing the user as "Beste burger'
    )
]

agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True,
    max_iterations=10)

response = agent("Is het adres van het stadshuis nog steeds botermakerstraat 69?")
print(response['output'])