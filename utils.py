import json

import requests

def load_test_data():
    test_data_path = "/Users/anhbamuoi/Code/automatedRagTesting/testdata/context_recall_framework.json"
    with open(test_data_path) as f:
        return json.load(f)

def get_llm_response(test_data):
    response_dict = requests.post('https://rahulshettyacademy.com/rag-llm/ask',
                                  json={
                                      "question": test_data['question'],
                                      "chat_history": []
                                  }).json()

    return response_dict