import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall
import requests

load_dotenv()

@pytest.mark.asyncio
async def test_context_recall():
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    langchain_llm = LangchainLLMWrapper(llm)
    context_recall =  LLMContextRecall(llm=langchain_llm)
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
    score = await context_recall.single_turn_ascore(sample)
    print(f"Context Recall Score: {score}")
    assert score > 0.7
