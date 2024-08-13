## AI_API_Recommender 소개
- 네이버 Clovax의 Completion, Embedding API를 활용한 RAG 구현
- data.go.kr 에서 제공하는 1만 2천여 건의 공공 API를 Embedding 하여 HCX-003 모델을 통해 질문하는 내용에 대한 API 정보를 제공함
- ex) 아이를 낳고 기르면서 도움이 될만한 앱을 만들고 싶은데 추천해줄 API가 있을까?

## 필요 설정
- Pipfile, Pipfile.lock, requirements.txt 파일을 참조하여 가상 환경을 설정
- API 관련 벡터 데이터를 저장하기 위한 Vector DB 설치 필요
  - `https://github.com/milvus-io/milvus/releases/download/v2.3.1/milvus-standalone-docker-compose.yml -O docker-compose.yml` 에서 Milvus 도커 설치를 위한 compose 파일 다운로드
  - 해당 파일로 Docker 설치를 위해 Docker, Docker Desktop 설치가 필요함 (설명 생략)
  - `docker compose up -d` (compose 파일이 있는 디렉토리에서 실행)
  - Milvus DB Docker 실행 확인
  - **app.ini 파일의 일부 내용(update with your information 로 표기된 부분) 은 clovax 본인 계정 정보에 맞게 수정해야 합니다.**
  - python milvus_collection_executor.py --option dropall 명령을 통해 필요한 collection 과 index를 milvus DB에 생성 합니다.
  - python csv_vdb_converter.py 명령을 통해 csv 파일의 레퍼런스 데이터를 vector화 하여 milvus로 insert 합니다.
  - python clovax_rag_service.py 명령을 통해 RAG 기능을 테스트 할수 있습니다.

## 코드 설명
- **milvus_collection_executor.py**: 다른 모듈에서 활용하는 모듈입니다. Milvus DB에 실제 Insert, Select를 처리합니다. (단독으로도 실행 가능하며 실행 시 프로젝트 실행에 필요한 관련 Collection과 Index를 생성해 줍니다.)
- **csv_vdb_converter.py**: 단독으로 실행하기 위해 만들었으며 clovax_rag_service가 실행되기 전에 CSV 파일의 레퍼런스 데이터를 Milvus DB로 벡터화해 Insert 해 주는 전처리 역할을 위한 모듈입니다.
- **clovax_rag_service.py**: 실제 RAG를 구현한 모듈입니다.
- **clovax_completion_executor.py**: 다른 모듈에서 활용하는 모듈입니다. Clovax와 연동하여 사용자 대화에 응답합니다. (단독으로 실행 가능하며 실행 시 LLM과 대화할 수 있습니다.)
- **clovax_embedding_executor.py**: 다른 모듈에서 활용하는 모듈입니다. Clovax와 연동하여 일반 텍스트를 벡터 데이터로 변경해 줍니다. (단독으로 실행 가능하며 실행 시 입력한 텍스트에 대한 벡터 정보를 보여 줍니다.)
- **logging_config.py**: 로깅 처리를 위한 단순한 기능입니다.

