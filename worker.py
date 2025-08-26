import time
from datetime import datetime
from zoneinfo import ZoneInfo
import finall

TZ = ZoneInfo("Asia/Qatar")

def align_to_15m():
    while True:
        now = datetime.now(TZ)
        m = now.minute
        s = now.second
        next_q = ((m // 15) + 1) * 15
        if next_q >= 60:
            next_q -= 60
        sleep_sec = (60 - s) + ((next_q - ((m + 1) % 60)) % 60) * 60
        time.sleep(sleep_sec)
        return

while True:
    align_to_15m()
    print("Running strategy...")
    try:
        res = finall.main_once()
        print(res)
    except Exception as e:
        print("Error:", e)
