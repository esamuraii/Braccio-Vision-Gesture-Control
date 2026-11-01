"""Downloads the official gesture model into this folder. Does not open Arduino."""
from pathlib import Path
import urllib.request
import zipfile
import os

URL = 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task'
p = Path(__file__).with_name('gesture_recognizer.task')
if p.is_file() and zipfile.is_zipfile(p):
    print('Gesture model already present:', p)
else:
    temporary = p.with_suffix('.download')
    try:
        print('Downloading the official MediaPipe gesture model...')
        with urllib.request.urlopen(URL, timeout=60) as source, temporary.open('wb') as output:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        if not zipfile.is_zipfile(temporary):
            raise RuntimeError('Unexpected model download; file not installed')
        os.replace(temporary, p)
        print('Model ready:', p)
    finally:
        temporary.unlink(missing_ok=True)
import mediapipe as mp
import cv2
import serial
print('MediaPipe:', getattr(mp, '__version__', 'installed'))
print('OpenCV:', cv2.__version__)
print('Setup complete. Start with --no-arm.')
