import csv
import os
import  itertools
import time
import configparser
import argparse
import logging
import logging_config
from clovax_embedding_executor import ClovaxEmbeddingExecutor
from milvus_collection_executor import MilvusCollectionExecutor
from colorama import Fore, Style

logger = logging.getLogger(__name__)

class CsvVdbConvertor:
    def __init__(self, _class_clovax_embedding_executor,
                 _class_milvus_collection_executor,
                 csv_backdata_file = os.path.join(os.getcwd(), 'backdata.csv'),
                 embedding_id_column_name = "ID",
                 embedding_target_column_name = "DESC",
                 embedding_result_column_name = "EMBEDDING",
                 csv_offset = 0,
                 csv_chunk_size = 100
                 ):

        self._class_clovax_embedding_executor = _class_clovax_embedding_executor
        self._class_milvus_collection_executor = _class_milvus_collection_executor

        self._csv_backdata_file = csv_backdata_file
        self.embedding_id_column_name = embedding_id_column_name
        self.embedding_target_column_name = embedding_target_column_name
        self.embedding_result_column_name = embedding_result_column_name

        self._csv_offset = csv_offset #embedding할 csv의  시작 row
        self._csv_chunk_size = csv_chunk_size #vector DB에 한번에 insert 할 row 수
        self._csv_column_names = []
        self._sub_entities = {}
        self._result_id = 0

    def store_backdata_with_embeddiing(self, csv_backdata_file = None , csv_offset = None, csv_chunk_size = None,
                                       embedding_id_column_name = None, embedding_target_column_name = None, embedding_result_column_name = None):

        try:
            if csv_backdata_file is None:
                csv_backdata_file = self._csv_backdata_file

            if csv_offset is None:
                csv_offset = self._csv_offset

            if csv_chunk_size is None:
                csv_chunk_size = self._csv_chunk_size

            if embedding_id_column_name is None:
                embedding_id_column_name = self.embedding_id_column_name

            if embedding_target_column_name is None:
                embedding_target_column_name = self.embedding_target_column_name

            if embedding_result_column_name is None:
                embedding_result_column_name = self.embedding_result_column_name

            with open(csv_backdata_file, encoding='utf-8-sig', mode='r', newline='') as file:
                reader = csv.reader(file)

                #csv 파일의 첫 라인에서 타이틀을 가져와서 entity의 컬럼명으로 사용
                self._csv_column_names = next(reader)
                for column_name in self._csv_column_names:
                    self._sub_entities.update({column_name : []})

                #embedding 컬럼 별도 추가(csv에는 없는 컬럼임으로)
                self._sub_entities.update({embedding_result_column_name : []})
                print(Fore.YELLOW + "Data check : OK", Style.RESET_ALL)

                #csv에서 데이터 로드
                adjusted_processing_limit = self._csv_chunk_size
                print(Fore.YELLOW + "Processing has started.", Style.RESET_ALL)

                #index 0 시작을 감안하여 csv offset 값 조정 필요
                adjusted_csv_offset = 0
                if csv_offset > 0:
                    adjusted_csv_offset = csv_offset - 1

                #타이틀 라인 다음 부터 시작
                for row_idx, row in enumerate(itertools.islice(reader, adjusted_csv_offset, None)):
                    #기존 데이타 초기화
                    for key in self._sub_entities:
                        self._sub_entities[key] = []

                    try:
                        for column_idx, column_value in enumerate(row):
                            column_values = list(self._sub_entities.values())[column_idx]
                            column_values.append(column_value)

                        #clovaX embedding 처리 요청, 여기서의 csv_offset은 보여지는 값으로 실제 csv 라인수를 마추기 위해 +2함(타이틀 라인과 index 0 시작을 감안하여)
                        self._result_id, result_status_code, result_vector = self._class_clovax_embedding_executor.execute(
                            row_idx + csv_offset + 2, self._sub_entities[embedding_target_column_name][len(self._sub_entities[embedding_target_column_name])-1])

                        #clovax embedding 처리 성공
                        if result_status_code == '20000':
                            print(f'\rEmbedding result : csv line number = {self._result_id}, status = success'
                                  f', id = {self._sub_entities[embedding_id_column_name][len(self._sub_entities[embedding_id_column_name])-1]}'
                                  f', origin = {self._sub_entities[embedding_target_column_name][len(self._sub_entities[embedding_target_column_name])-1][:10]}..'
                                  f', vector = {result_vector[0]}.. ', end='', flush=True)

                            self._sub_entities[embedding_result_column_name].append(result_vector)

                            #DB insert
                            self._class_milvus_collection_executor.insert_embedding(self._sub_entities)

                            #clovax embedding api 의 테스트앱 QPM을 고려해서 대략 QPM=60 정도로 처리
                            time.sleep(1)

                        #clovax embedding 처리 실패
                        else:
                            raise Exception(f'Clovax returned an error during the embedding : idx = {self._result_id}, status = fail({result_status_code})')

                        #_csv_chunk_size 마다 진행을 계속 할지 물음
                        if self._csv_chunk_size != 0 and (row_idx + 1) % self._csv_chunk_size == 0:
                            print(f"\r{Fore.YELLOW}Would you like to continue with another embedding(anyKey)? Or stop here(s)? {Style.RESET_ALL}", end="")
                            user_input = input().strip().lower()
                            if user_input == "s":
                                break

                    except Exception as e:
                        logger.error(e, exc_info = True)

                        #계속 진행을 할지 확인
                        print(f"\r{Fore.YELLOW}Would you like to stop here(s)? Or skip only this data and continue with another embedding(anyKey)? {Style.RESET_ALL}", end="")
                        user_input = input().strip().lower()
                        if user_input == "s":
                            break

        except Exception as e:
            raise e

        #종료 안내, , csv 파일내 컬럼 타이틀 line 감안
        print(f"{Fore.YELLOW}Processing has stopped. The next csv_offset is {self._result_id}.{Style.RESET_ALL}")


# RAG 활용을 위한 레퍼런스 데이터를 생성함 (주어진 csv 파일을 파싱하여 milvus vector DB store 함)
# python csv_vdb_converter.py --csv_offset 1
def main():
    try:
        app_config = configparser.ConfigParser()
        app_config.read('./app.ini', 'utf-8')
        app_common_config = app_config['APP_COMMON']
        clovax_common_config = app_config['CLOVAX_COMMON']
        clovax_embedding_config = app_config['CLOVAX_EMBEDDING']

        parser = argparse.ArgumentParser(description='start options.')
        parser.add_argument('--csv_offset', type=str, help='numeric value')
        args = parser.parse_args()

        csv_offset = int(app_common_config["csv_offset"])
        if args.csv_offset is not None:
            csv_offset = int(args.csv_offset.strip())

        print(f"csv_offset is {csv_offset}")

        _class_csv_vdb_convertor = CsvVdbConvertor(
            _class_clovax_embedding_executor = ClovaxEmbeddingExecutor(
                host = clovax_common_config['host'],
                api_uri = clovax_embedding_config['api_uri'],
                api_key = clovax_common_config['api_key'],
                api_key_primary_val = clovax_common_config['api_key_primary_val'],
                request_id = clovax_common_config['request_id'],
            ),
            _class_milvus_collection_executor = MilvusCollectionExecutor(
                collection_name = app_common_config["embedding_target_collection_name"]
            ),
            csv_backdata_file = app_common_config['csv_backdata_file'],
            embedding_id_column_name = app_common_config['embedding_id_column_name'],
            embedding_target_column_name = app_common_config['embedding_target_column_name'],
            embedding_result_column_name = app_common_config['embedding_result_column_name'],

            csv_offset = csv_offset,
            csv_chunk_size = int(app_common_config['csv_chunk_size'])
        )

        _class_csv_vdb_convertor.store_backdata_with_embeddiing()

    except Exception as e:
        logger.error(e, exc_info=True)


if __name__ == "__main__":
    main()