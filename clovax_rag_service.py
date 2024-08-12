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

# RAG 기능을 제공하는 서비스
# python clovax_rag_service.py
def main():
    try:
        app_config = configparser.ConfigParser()
        app_config.read('./app.ini', 'utf-8')
        app_common_config = app_config['APP_COMMON']
        clovax_common_config = app_config['CLOVAX_COMMON']
        clovax_embedding_config = app_config['CLOVAX_EMBEDDING']
        clovax_completion_config = app_config['CLOVAX_COMPLETION']

        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--srcn', type=str, help='sytem role content number(1)')
        parser.add_argument('--restype', type=str, help='response type(stream)')
        args = parser.parse_args()

        _class_milvus_collection_executor = MilvusCollectionExecutor(app_common_config["embedding_target_collection_name"])

        _class_clovax_embedding_executor = ClovaxEmbeddingExecutor(
            host = clovax_common_config['host'],
            api_uri = clovax_embedding_config['api_uri'],
            api_key = clovax_common_config['api_key'],
            api_key_primary_val = clovax_common_config['api_key_primary_val'],
            request_id = clovax_common_config['request_id']
        )

        _class_clovax_completion_executor = ClovaxCompletionExecutor(
            host = clovax_completion_config['host'],
            api_uri = clovax_completion_config['api_uri'],
            api_key = clovax_common_config['api_key'],
            api_key_primary_val = clovax_common_config['api_key_primary_val'],
            request_id = clovax_common_config['request_id']
        )

        #프로퍼티 파일에 등록된 system role content 내용 중 몇 번째를 적용 할지 실행 argument 로 받아 처리함
        srcn = 1
        if args.srcn is not None:
            srcn = args.srcn.strip()

        system_role_contents = clovax_completion_config[f'system_role_contents_{srcn}'].replace("\n", "")
        logger.info(f"System_role_contents_{srcn} has been applied.")

        print(f"{Fore.YELLOW}공공 데이터 검색을 도와 드립니다. 필요한 내용을 구체적으로 입력해 주세요.{Style.RESET_ALL}")
        while True:
            user_request = input(">")
            if user_request == 'bye':
                break

            id, result_status, result_vector = _class_clovax_embedding_executor.execute(0, user_request)
            logger.debug(f"result status : {result_status}, embedding dim size : {len(result_vector)} ")
            logger.debug(f"embedding vector : {result_vector}")
            rag_reference_data = _class_milvus_collection_executor.search_embedding(result_vector,
                                                                                   app_common_config["embedding_result_column_name"],
                                                                                   ['ID', 'TITLE', 'ORG', 'DESC', 'URL'],
                                                                                   10) #todo: limit 갯수에 따라 오류가 나는 이유 확인 필요
            logger.info(f"rag reference data size : {len(rag_reference_data)}")
            logger.debug(f"rag result :")
            logger.debug(f"{rag_reference_data}")

            preset_text = [
                {"role" : "system", "content" : system_role_contents},
                {"role" : "system", "content" : f"질문 답변을 위한 reference : {rag_reference_data}"},
                {"role" : "user", "content" : user_request}
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

            completion_result = _class_clovax_completion_executor.execute(request_data, args.restype)
            print(f"{Fore.GREEN}{completion_result}{Style.RESET_ALL}")

    except Exception as e:
        logger.error(e, exc_info = True)


if __name__ == "__main__":
    main()