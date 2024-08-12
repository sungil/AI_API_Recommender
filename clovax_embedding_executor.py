import json
import http.client
import configparser
import logging
import logging_config
from colorama import Fore, Style

logger = logging.getLogger(__name__)

class ClovaxEmbeddingExecutor:
    def __init__(self, host, api_uri, api_key, api_key_primary_val, request_id):
        self._host = host
        self._api_uri = api_uri
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id

    def _send_request(self, request_data):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id
        }

        conn = http.client.HTTPSConnection(self._host)
        conn.request('POST', self._api_uri, json.dumps(request_data), headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode(encoding='utf-8'))
        conn.close()
        return result


    def execute(self, id, request_text):
        result = self._send_request(
            {"text" : request_text}
        )

        return id, result['status']['code'], result['result']['embedding']

# 직접 실행시 입력 text를 clovax와 연동하여 vector 값을 반환함
# python clovax_embedding_executor.py
def main():
    try:
        app_config = configparser.ConfigParser()
        app_config.read('./app.ini', 'utf-8')
        app_common_config = app_config['APP_COMMON']
        clovax_common_config = app_config['CLOVAX_COMMON']
        clovax_embedding_config = app_config['CLOVAX_EMBEDDING']

        _class_clovax_embedding_executor = ClovaxEmbeddingExecutor(
            host = clovax_common_config['host'],
            api_uri = clovax_embedding_config['api_uri'],
            api_key = clovax_common_config['api_key'],
            api_key_primary_val = clovax_common_config['api_key_primary_val'],
            request_id = clovax_common_config['request_id']
        )

        print(f"{Fore.YELLOW}Embedding 할 내용을 입력 하세요.{Style.RESET_ALL}")
        while True:
            req_text = input(">")
            if req_text == 'bye':
                break

            id, result_status, result_vector = _class_clovax_embedding_executor.execute(0, req_text)
            print(f"{Fore.YELLOW}status : {result_status}, dim size : {len(result_vector)}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}{result_vector}{Style.RESET_ALL}")


    except Exception as e:
        logger.error(e, exc_info = True)


if __name__ == "__main__":
    main()