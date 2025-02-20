from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pytest
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference
import requests

# Load environment variables from .env file so that the OPENAI_API_KEY and other env variables are
# available on the environment.
load_dotenv()

# Terms and definitions
# user input in ragas = query
# response in ragas = response
# reference in ragas = ground truth (expected results)
# retrieved_context in ragas = Top k retrieved documents


@pytest.mark.asyncio
async def test_context_precision():
  # Create instance of class for specific metric
  # Power of LLM + method metric => score
  llm = ChatOpenAI(model='gpt-4', temperature=0)
  langchain_llm = LangchainLLMWrapper(llm)
  context_precision = LLMContextPrecisionWithoutReference(llm=langchain_llm)
  question = "How many articles are there in the Selenium webdriver python course?"
  response_dict = requests.post('https://rahulshettyacademy.com/rag-llm/ask',
                                json={
                                  "question": question ,
                                  "chat_history": []
                                }).json()

  sample = SingleTurnSample(
    user_input=question,
    response=response_dict['answer'],
    retrieved_contexts=[retrieved_context['page_content'] for retrieved_context in response_dict['retrieved_docs']]
  )

  # Feed data
  # Score
  score = await context_precision.single_turn_ascore(sample)
  print(f"Context Precision Score: {score}")

  # Example assertion
  assert score > 0.8