import pytest
from dotenv import load_dotenv
from ragas import SingleTurnSample
from ragas.metrics import LLMContextRecall

from utils import get_llm_response, load_test_data

load_dotenv()

@pytest.mark.asyncio
@pytest.mark.parametrize('get_data', load_test_data(), indirect=True)
async def test_context_recall(llm_wrapper, get_data):
    context_recall =  LLMContextRecall(llm=llm_wrapper)
    score = await context_recall.single_turn_ascore(get_data)
    print(f"\nContext Recall Score: {score}")
    assert score > 0.7


@pytest.fixture
def get_data(request):
    test_data = request.param
    response_dict = get_llm_response(test_data)

    sample = SingleTurnSample(
        user_input=test_data['question'],
        retrieved_contexts=[retrieved_context['page_content'] for retrieved_context in response_dict['retrieved_docs']],
        reference=test_data['reference']
    )

    return sample

