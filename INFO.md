# 백엔드 서버 실행
cd ~/Coach
uvicorn server:app --host 0.0.0.0 --port 8000 

# 클라이언트(멘토파이) 연결되어야 영상 및 제어 가능
cd ~/Coach
python3 client.py

# 아래의 웹브라우저로 접속하여 제어 가능
https://awdanmxyaxxcgabw.tunnel.elice.io/