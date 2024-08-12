import json
import requests
import configparser
import logging
import logging_config
from colorama import Fore, Style

logger = logging.getLogger(__name__)

class ClovaxCompletionExecutor:
    def __init__(self, host, api_uri, api_key, api_key_primary_val, request_id):
        self._host = host
        self._api_uri = api_uri
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id


    def execute(self, request_data, response_type):
        headers = {
            "X-NCP-CLOVASTUDIO-API-KEY": self._api_key,
            "X-NCP-APIGW-API-KEY": self._api_key_primary_val,
            "X-NCP-CLOVASTUDIO-REQUEST-ID": self._request_id,
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "text/event-stream"
        }

        response = requests.post(
            self._host + self._api_uri,
            headers = headers,
            json = request_data,
            stream = True
        )

        # 스트림에서 마지막 'data:' 라인을 찾기 위한 로직
        last_data_content = ""

        #todo : stream 옵션일때 동일한 답변이 두번 찍히는 버그 수정 필요, 답변이 중간에 짤리는 이유 확인 필요
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                if '"data":"[DONE]"' in decoded_line: #Done이 나오면 응답의 끝으로 본다.
                    if response_type is not None and response_type.lower() == 'stream':
                        last_data_content = ""
                    break

                if decoded_line.startswith("data:"):
                    last_data_content = json.loads(decoded_line[5:])["message"]["content"]

                    if response_type is not None and response_type.lower() == 'stream':
                        logger.debug(f"completion origin stream : {decoded_line}")
                        print(f"{Fore.GREEN}{last_data_content}{Style.RESET_ALL}", end="")

        return last_data_content


# 직적 실행시 LLM 과 대화 가능
# python clovax_completion_executor.py
def main():
    try:
        app_config = configparser.ConfigParser()
        app_config.read('./app.ini', 'utf-8')
        app_common_config = app_config['APP_COMMON']
        clovax_common_config = app_config['CLOVAX_COMMON']
        clovax_embedding_config = app_config['CLOVAX_EMBEDDING']
        clovax_completion_config = app_config['CLOVAX_COMPLETION']

        _class_clovax_completion_executor = ClovaxCompletionExecutor(
            host = clovax_completion_config['host'],
            api_uri = clovax_completion_config['api_uri'],
            api_key = clovax_common_config['api_key'],
            api_key_primary_val = clovax_common_config['api_key_primary_val'],
            request_id = clovax_common_config['request_id']
        )

        print(f"{Fore.YELLOW}대화 내용을 입력 하세요.{Style.RESET_ALL}")
        while True:
            user_requestion = input(">")
            if user_requestion == 'bye':
                break

            preset_text = [
                {"role" : "user", "content" : user_requestion}
            ]

            request_data = {
                'messages': preset_text,
                'topP': 0.8,
                'topK': 0,
                'maxTokens': 256,
                'temperature': 0.98,
                'repeatPenalty': 5.0,
                'stopBefore': [],
                'includeAiFilters': True,
                'seed': 0
            }

            #completion_result = _class_clovax_completion_executor.execute(request_data, None)
            completion_result = _class_clovax_completion_executor.execute(request_data, 'stream')
            print(f"{Fore.GREEN}{completion_result}{Style.RESET_ALL}")

    except Exception as e:
        logger.error(e, exc_info = True)


if __name__ == "__main__":
    main()