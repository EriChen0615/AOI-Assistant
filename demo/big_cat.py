import time
import sys

ESC = "\033"
CURSOR_HOME = f"{ESC}[H"
CLEAR_SCREEN = f"{ESC}[2J"
HIDE_CURSOR = f"{ESC}[?25l"
SHOW_CURSOR = f"{ESC}[?25h"

# Two cat frames: normal + blink
cat_frames = [
    r"""
        ／＞　 フ
       | 　_　_| 
     ／` ミ＿xノ 
    /　　　　 |
   /　 ヽ　　 ﾉ
   │　　|　|　|
／￣|　　 |　|　|
(￣ヽ＿_ヽ_)__)
＼二)
""",
    r"""
        ／＞　 フ
       | 　_　_| 
     ／` ミ_－ノ 
    /　 -　 -　|
   /　 ◉　 ◉ ﾉ
   │　　▼ |　|
／￣|　　 |　|　|
(￣ヽ＿_ヽ_)__)
＼二)
"""
]

def print_avatar(frame):
    sys.stdout.write(CURSOR_HOME)
    print(frame, end="", flush=True)

def animate_cat(frames, delay=0.6, repeat=30):
    sys.stdout.write(CLEAR_SCREEN)
    sys.stdout.write(HIDE_CURSOR)
    try:
        for i in range(repeat):
            print_avatar(frames[i % len(frames)])
            time.sleep(delay)
    finally:
        sys.stdout.write(SHOW_CURSOR)
        print()
    
if __name__ == "__main__":
    animate_cat(cat_frames)
