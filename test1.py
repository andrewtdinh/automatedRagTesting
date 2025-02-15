import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pytest
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


@pytest.mark.asyncio
async def test_context_precision():
  # Create instance of class for specific metric
  # Power of LLM + method metric => score
  llm = ChatOpenAI(model='gpt-4', temperature=0)
  langchain_llm = LangchainLLMWrapper(llm)
  context_precision = LLMContextPrecisionWithoutReference(llm=langchain_llm)

  # Feed data
  sample = SingleTurnSample(
    user_input="How many articles are there in the Selenium webdriver python course?",
    response="There are 23 articles in the Selenium WebDriver Python course. \n",
    retrieved_contexts=[
      "Complete Understanding on Selenium Python API Methods with real time Scenarios on LIVE Websites\n\"Last but not least\" you can clear any Interview and can Lead Entire Selenium Python Projects from Design Stage\nThis course includes:\n17.5 hours on-demand video\nAssignments\n23 articles\n9 downloadable resources\nAccess on mobile and TV\nCertificate of completion\nRequirements"
    ]

  )

  # Score
  score = await context_precision.single_turn_ascore(sample)
  print(f"Context Precision Score: {score}")