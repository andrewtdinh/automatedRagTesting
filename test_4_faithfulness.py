import pytest
from ragas import SingleTurnSample
from ragas.metrics import Faithfulness

from utils import load_test_data, get_llm_response


@pytest.mark.parametrize("get_data", load_test_data(), indirect=True)
@pytest.mark.asyncio
async def test_faithfulness(llm_wrapper, get_data):
    faithful = Faithfulness(llm=llm_wrapper)
    score = faithful.single_turn_ascore(get_data)
    print(f"Faithful score: {score}")
    assert score > 0.8


@pytest.fixture
def get_data(request):
    test_data = request.param
    response_dict = get_llm_response(test_data)
    sample = SingleTurnSample(
        user_input=test_data['question'],
        response=response_dict['answer'],
        retrieved_contexts=[response_dict["retrieved_docs"][0]['page_content']]
        # retrieved_contexts=[retrieved_context['page_content'] for retrieved_context in response_dict['retrieved_docs']],
    )

    return sample