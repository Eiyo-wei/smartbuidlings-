@echo off
start /b ffmpeg -stream_loop -1 -re -i "D:\work\1.mp4" -c copy -f flv rtmp://localhost:1935/live/live01
start /b ffmpeg -stream_loop -1 -re -i "D:\work\2.mp4" -c copy -f flv rtmp://localhost:1935/live/live02
start /b ffmpeg -stream_loop -1 -re -i "D:\work\3.mp4" -c copy -f flv rtmp://localhost:1935/live/live03
start /b ffmpeg -stream_loop -1 -re -i "D:\work\5.mp4" -c copy -f flv rtmp://localhost:1935/live/test01
echo
pause