#!/usr/bin/env python
import os, sys

os.system('pkill -f video_start-cgi.py')
sys.stdout.write('Content-Type: text/event-stream\n')
sys.stdout.write('Access-Control-Allow-Origin: *\n\n')
sys.stdout.write(f'data: stop\n\n')
sys.stdout.flush()