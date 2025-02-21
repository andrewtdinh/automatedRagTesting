import pytest
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper


@pytest.fixture
def llm_wrapper():
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    langchain_llm = LangchainLLMWrapper(llm)

    return langchain_llm