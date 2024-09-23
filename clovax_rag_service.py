import configparser
import argparse
import traceback
import logging
import logging_config
from colorama import Fore, Style
from clovax_embedding_executor import ClovaxEmbeddingExecutor
from clovax_completion_executor import ClovaxCompletionExecutor
from milvus_collection_executor import MilvusCollectionExecutor

logger = logging.getLogger(__name__)
_class_clovax_embedding_executor = None
_class_milvus_collection_executor = None
_class_clovax_completion_executor = None
system_role_contents = None
app_common_config = None
app_ini_system_role_contents_num = None
completion_response_type = None #응답을 스트림 방식으로 받을지 말지


def initiate(is_console_mode = True):
    try:
        app_config = configparser.ConfigParser()
        app_config.read('./app.ini', 'utf-8')
        global app_common_config
        app_common_config = app_config['APP_COMMON']
        clovax_common_config = app_config['CLOVAX_COMMON']
        clovax_embedding_config = app_config['CLOVAX_EMBEDDING']
        clovax_completion_config = app_config['CLOVAX_COMPLETION']

        if is_console_mode:
            parser = argparse.ArgumentParser(description='')
            parser.add_argument('--srcn', type=str, help='sytem role content number(1)')
            parser.add_argument('--restype', type=str, help='response type(stream)')
            args = parser.parse_args()

            global app_ini_system_role_contents_num
            if args.srcn is not None:
                app_ini_system_role_contents_num = args.srcn.strip()

            global completion_response_type
            if args.restype is not None:
                completion_response_type = args.restype
        else:
            app_ini_system_role_contents_num = 1
            completion_response_type = ""

        global _class_milvus_collection_executor
        _class_milvus_collection_executor = MilvusCollectionExecutor(app_common_config["embedding_target_collection_name"])

        global _class_clovax_embedding_executor
        _class_clovax_embedding_executor = ClovaxEmbeddingExecutor(
            host = clovax_common_config['host'],
            api_uri = clovax_embedding_config['api_uri'],
            api_key = clovax_common_config['api_key'],
            api_key_primary_val = clovax_common_config['api_key_primary_val'],
            request_id = clovax_common_config['request_id']
        )

        global _class_clovax_completion_executor
        _class_clovax_completion_executor = ClovaxCompletionExecutor(
            host = clovax_completion_config['host'],
            api_uri = clovax_completion_config['api_uri'],
            api_key = clovax_common_config['api_key'],
            api_key_primary_val = clovax_common_config['api_key_primary_val'],
            request_id = clovax_common_config['request_id']
        )

        #프로퍼티 파일에 등록된 system role content 내용 중 몇 번째를 적용 할지 실행 argument 로 받아 처리함
        global system_role_contents
        system_role_contents = clovax_completion_config[f'system_role_contents_{app_ini_system_role_contents_num}']#.replace("\n", "")
        logger.info(f"System_role_contents_{app_ini_system_role_contents_num} has been applied.")

    except Exception as e:
        raise e


def rag_service(user_request):
    try:
        id, result_status, result_vector = _class_clovax_embedding_executor.execute(0, user_request)
        logger.debug(f"result status : {result_status}, embedding dim size : {len(result_vector)} ")
        logger.debug(f"embedding vector : {result_vector}")
        rag_reference_data = _class_milvus_collection_executor.search_embedding(result_vector,
                                                                                app_common_config["embedding_result_column_name"],
                                                                                ['ID', 'TITLE', 'ORG', 'DESC', 'URL'],
                                                                                20) #todo: limit 갯수에 따라 오류가 나는 이유 확인 필요
        logger.info(f"rag reference data size : {len(rag_reference_data)}")
        logger.debug(f"rag result :")
        logger.debug(f"{rag_reference_data}")

        preset_text = [
            {"role" : "system", "content" : system_role_contents},
            {"role" : "system", "content" : f"reference 데이터: {rag_reference_data}"},
            {"role" : "user", "content" : user_request}
        ]

        request_data = {
            'messages': preset_text,
            'topP': 0.8,
            'topK': 0,
            'maxTokens': (256*5),
            'temperature': 0.98,
            'repeatPenalty': 5.0,
            'stopBefore': [],
            'includeAiFilters': False,
            'seed': 0
        }

        completion_result = _class_clovax_completion_executor.execute(request_data, completion_response_type)
        logger.debug(f"rag completion_result : {completion_result}")
        return completion_result

    except Exception as e:
        raise e


# RAG 기능을 제공하는 서비스
# python clovax_rag_service.py
def main():
    try:
        initiate(is_console_mode=True)

        print(f"{Fore.YELLOW}공공 데이터 검색을 도와 드립니다. 필요한 내용을 구체적으로 입력해 주세요.{Style.RESET_ALL}")
        while True:
            user_request = input(">")
            if user_request == 'bye':
                break

            completion_result = rag_service(user_request)
            print(f"{Fore.GREEN}{completion_result}{Style.RESET_ALL}")

    except Exception as e:
        logger.error(e, exc_info = True)


if __name__ == "__main__":
    main()
