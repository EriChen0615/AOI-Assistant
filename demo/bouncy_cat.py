import os
import time
import sys

# Define the cat frames (animation frames)
cat_frames = [
    r"""
     /\_/\  
    ( o.o ) 
     > ^ <  
    """,
    r"""
     /\_/\  
    ( -.- ) 
     > ^ <  
    """,
    r"""
     /\_/\  
    ( o_o ) 
     > ^ <  
    """
]

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def animate_cat(bounce_width=20, delay=0.1, repeat=3):
    frame_count = len(cat_frames)
    direction = 1  # 1: right, -1: left
    position = 0
    frame_index = 0

    for _ in range(repeat * bounce_width * 2):
        clear()

        # Calculate padding
        padding = " " * position
        frame = cat_frames[frame_index % frame_count]
        print(padding + frame.replace("\n", "\n" + padding))

        # Update position
        if direction == 1:
            position += 1
            if position >= bounce_width:
                direction = -1
        else:
            position -= 1
            if position <= 0:
                direction = 1

        frame_index += 1
        time.sleep(delay)

try:
    animate_cat()
except KeyboardInterrupt:
    sys.exit(0)
