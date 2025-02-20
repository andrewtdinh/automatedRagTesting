import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall
import requests

load_dotenv()

@pytest.mark.asyncio
async def test_context_recall(llm_wrapper, get_data):
    context_recall =  LLMContextRecall(llm=llm_wrapper)
    score = await context_recall.single_turn_ascore(get_data)
    print(f"Context Recall Score: {score}")
    assert score > 0.7

@pytest.fixture
def llm_wrapper():
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    langchain_llm = LangchainLLMWrapper(llm)

    return langchain_llm

@pytest.fixture
def get_data():
    question = "How many articles are there in the Selenium webdriver python course?"
    response_dict = requests.post('https://rahulshettyacademy.com/rag-llm/ask',
                                  json={
                                      "question": question,
                                      "chat_history": []
                                  }).json()

    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=[retrieved_context['page_content'] for retrieved_context in response_dict['retrieved_docs']],
        reference="23"
    )

    return sample