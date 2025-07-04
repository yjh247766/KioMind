#!/bin/bash

export DISPLAY=:0
cd /home/KioMind/kiomind

mkdir -p logs
LOGFILE="logs/flask_$(date +%Y%m%d_%H%M%S).log"
PORT=5000

# Flask 서버가 이미 실행 중인지 확인
PID=$(lsof -ti tcp:$PORT)

if [ -n "$PID" ]; then
  echo " 기존 Flask 프로세스 종료 (PID: $PID)" >> "$LOGFILE"
  kill -9 $PID
  sleep 1
fi


echo "Flask 서버 시작-> 로그저장 : $LOGFILE"
/home/KioMind/.pyenv/versions/3.10.17/bin/python3 main.py > "$LOGFILE" 2>&1 &

sleep 3

firefox --kiosk http://127.0.0.1:5000 || echo "Firefox 실행 실패" >> "$LOGFILE"
