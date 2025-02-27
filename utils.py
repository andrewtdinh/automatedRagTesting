import json
from pathlib import Path

import requests

def load_test_data(filename):
    project_dir = Path(__file__).parent.absolute()
    test_data_path = f"{project_dir}/testdata/{filename}"
    with open(test_data_path) as f:
        return json.load(f)

def get_llm_response(test_data):
    response_dict = requests.post('https://rahulshettyacademy.com/rag-llm/ask',
                                  json={
                                      "question": test_data['question'],
                                      "chat_history": []
                                  }).json()

    return response_dict