import time
import sys
import argparse
import configparser
import logging
import logging_config
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from colorama import Fore, Style

logger = logging.getLogger(__name__)

class MilvusCollectionExecutor:
    def __init__(self, collection_name="CLT_EMBEDDING"): #collection default name
        self.connect_to_db(connection_name="default", host="localhost", port="19530")
        self._collection_name = collection_name

    def connect_to_db(self, connection_name="default", host="localhost", port="19530"):
        try:
            # 현재 연결 상태 확인
            active_connections = connections.list_connections()
            logger.info(f"Milvus active_connections : {active_connections}")

            if connection_name in active_connections:
                print(f"Already connected to Milvus with '{connection_name}'.")
                return
            else:
                # 연결이 없거나 오류가 발생한 경우 연결 시도
                connections.connect(connection_name, host=host, port=port)
                print(f"It has been Connected to Milvus with '{connection_name}'.")

        except Exception as e:
            raise e


    def drop_all_collection(self):
        collection_names = utility.list_collections()
        for collection_name in collection_names:
            utility.drop_collection(collection_name)
            print(f"{collection_name} has been dropped.")


    def create_collection(self):
        try:
            if utility.has_collection(self._collection_name):
                print(f"Collection '{self._collection_name}' already exists.")

            else:
                fields = [
                    FieldSchema(name = "ID", dtype=DataType.VARCHAR, max_length = 10, is_primary = True, auto_id = False),
                    FieldSchema(name = "TYPE", dtype=DataType.VARCHAR, max_length = 10),
                    FieldSchema(name = "TITLE", dtype=DataType.VARCHAR, max_length = 256),
                    FieldSchema(name = "DESC", dtype=DataType.VARCHAR, max_length = 1024),
                    FieldSchema(name = "ORG", dtype=DataType.VARCHAR, max_length = 256),
                    FieldSchema(name = "URL", dtype=DataType.VARCHAR, max_length = 256),
                    FieldSchema(name = "EMBEDDING", dtype=DataType.FLOAT_VECTOR, dim = 1024)
                ]
                schema = CollectionSchema(fields, description = self._collection_name)

                embedding_collection = Collection(name = self._collection_name, schema = schema, using='default', shards_num=2)
                print(f"Collection '{self._collection_name}' has been created successfully.")
                time.sleep(0.5)

                #벡터 필드에 대한 index 생성
                index_params = {
                    "metric_type" : "IP",
                    "index_type" : "HNSW",
                    "params": {
                        "M": 8,
                        "efConstruction": 200
                    }
                }
                embedding_collection.create_index(field_name="EMBEDDING", index_params=index_params)

            embedding_collection = Collection(self._collection_name)

            #todo : 로그 처리 수정 필요
            #컬렉션 스키마 정보 확인
            #logger.info(Fore.GREEN + "Collection schema : ", embedding_collection.schema, Style.RESET_ALL)
            time.sleep(0.5)

            #todo : 로그 처리 수정 필요
            #컬렉션 index 정보 확인
            #logger.info(Fore.GREEN + "Collection index : ", [index.params for index in embedding_collection.indexes], Style.RESET_ALL)

        except Exception as e:
            raise e


    def insert_embedding(self, data):
        try:
            collection = Collection(self._collection_name)

            #data['ID'] = numpy.array(data['ID'], dtype = numpy.int64) #int형으로 변경
            insert_data = [data[field.name] for field in collection.schema.fields if field.name in data]
            insert_result = collection.insert(insert_data)
            print(f"Insert result: {insert_result}")

        except Exception as e:
            raise e


    def search_embedding(self, query_vectors, embedding_column_name, select_column_names, limit):
        try:
            collection = Collection(self._collection_name)
            collection.load()

            search_params = {"metric_type": "IP", "params": {"ef": 64}}
            results = collection.search(
                data = [query_vectors],  # 검색할 벡터 데이터
                anns_field = embedding_column_name,  # 검색을 수행할 벡터 필드 지정
                param = search_params,
                limit = limit,
                output_fields = select_column_names
            )

            references = []
            for hit in results[0]:
                distance = hit.distance
                id = hit.entity.get("ID")
                title = hit.entity.get("TITLE")
                org = hit.entity.get("ORG")
                desc = hit.entity.get("DESC")
                url = hit.entity.get("URL")
                references.append({"distance" : distance, "id": id, "title" : title, "org" : org, "desc" : desc, "url" : url})

            return references

        except Exception as e:
            raise e

# 직접 실행시 RAG 구현을 휘한 milvus collection setting을 함
# python milvus_collection_executor.py --option dropall
def main():
    try:
        app_config = configparser.ConfigParser()
        app_config.read('./app.ini', 'utf-8')
        app_common_config = app_config['APP_COMMON']

        parser = argparse.ArgumentParser(description='Processing mode.')
        parser.add_argument('--option', type=str, help='dropall')
        args = parser.parse_args()

        _class_milvus_collection_executor = MilvusCollectionExecutor(app_common_config["embedding_target_collection_name"])
        if args.option is not None and args.option.strip().lower() == 'dropall':
            _class_milvus_collection_executor.drop_all_collection()

        time.sleep(0.5)
        _class_milvus_collection_executor.create_collection()
        time.sleep(0.5)
        print(Fore.YELLOW + "It has been successfully initialized.", Style.RESET_ALL)

    except Exception as e:
        logger.error(e, exc_info = True)


if __name__ == "__main__":
    main()
