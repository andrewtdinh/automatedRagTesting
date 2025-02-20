import pytest
import requests
from ragas import SingleTurnSample

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