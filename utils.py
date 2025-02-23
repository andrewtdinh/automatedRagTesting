import requests

def get_llm_response(test_data):
    response_dict = requests.post('https://rahulshettyacademy.com/rag-llm/ask',
                                  json={
                                      "question": test_data['question'],
                                      "chat_history": []
                                  }).json()