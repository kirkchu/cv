#!/Users/ckk/venv/cv/bin/python
import os

os.system('pkill -f video_start-cgi.py')
print('Content-Type: text/event-stream')
print('Access-Control-Allow-Origin: *')
print()
print(f'data: stop\n')