import sys

def log(msg):
    frame = sys._getframe(1)  # Get the caller's frame
    print(f'[Info]: {msg} ,File: "{frame.f_code.co_filename}", Line: {frame.f_lineno}')