import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# user input in ragas = query
# response in ragas = response
# reference in ragas = ground truth (expected results)
# retrieved_context in ragas = Top k retrieved documents

def test_context_precision():
  # Create instance of class for specific metric
  # Power of LLM + method metric => score
  llm = ChatOpenAI(model='gpt-4', temperature=0)
  langchain_llm = LangchainLLMWrapper(llm)
  context_precision = LLMContextPrecisionWithoutReference(llm=langchain_llm)

  # Feed data
  SingleTurnSample(
    user_input="How many articles are there in the Selenium webdriver python course",
    response="There are 23 articles in the course.",
    retrieved_contexts=[]

  )

  # Score