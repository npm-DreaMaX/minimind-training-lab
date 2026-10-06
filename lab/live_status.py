"""Bounded retries for live JSON publication observed through mounted filesystems."""
import json,time
from pathlib import Path

def read_live_json(path,*,missing_ok=False,timeout=30.,interval=.1,report=None):
    path=Path(path);started=time.monotonic();attempts=0
    emit=report or (lambda event:print(json.dumps(event),flush=True))
    while True:
        try:
            value=json.loads(path.read_text())
        except (FileNotFoundError,json.JSONDecodeError) as exc:
            if missing_ok and isinstance(exc,FileNotFoundError):return None
            if time.monotonic()-started>=timeout:
                raise RuntimeError(f'Live JSON unavailable after bounded retry: {path}') from exc
            attempts+=1
            if attempts==1:emit(dict(event='live_json_retry',path=str(path),error=type(exc).__name__))
            time.sleep(interval)
        else:
            if attempts:emit(dict(event='live_json_recovered',path=str(path),attempts=attempts,seconds=time.monotonic()-started))
            return value
