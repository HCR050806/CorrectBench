import requests
from retrying import retry

class ChatWithOurServer:
    def __init__(self, base_url='http://127.0.0.1:9000', model='facebook/opt-125m'):
        self.base_url = base_url
        self.model = model


    @retry(stop_max_attempt_number=3)
    def create(self, messages, **kwargs):
        response = requests.post(self.base_url+'/chat/completions', json={'messages': messages, 'model': self.model, **kwargs})
        return response.json()['content']#