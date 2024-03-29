from datetime import datetime

def get_today():
    return datetime.now().strftime("%Y%m%d")