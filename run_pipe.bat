# 1. 프로젝트 루트 디렉토리 이동
cd ~/work/1_Bootcamp/2_Intermediate_NLP/GIT

# 2. uv venv 가상환경 활성화 (파워 레일 연결)
source .venv/bin/activate

# 3. 환경 변수 레지스터(API Key) 세팅 확인
# (.env 파일에 OPENAI_API_KEY="your_key" 가 저장되어 있어야 합니다)
export $(cat .env | xargs)

python pipeline.py

echo
cat reports/hit_scoreboard.csv

echo
cat reports/ragas_evaluation_result.csv
