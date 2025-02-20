import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall

load_dotenv()

@pytest.mark.asyncio
async def test_context_recall(llm_wrapper, get_data):
    context_recall =  LLMContextRecall(llm=llm_wrapper)
    score = await context_recall.single_turn_ascore(get_data)
    print(f"\nContext Recall Score: {score}")
    assert score > 0.7

@pytest.fixture
def llm_wrapper():
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    langchain_llm = LangchainLLMWrapper(llm)

    return langchain_llm

