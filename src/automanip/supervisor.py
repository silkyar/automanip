"""Trusted control-plane helpers. Never import this module in training code."""
from dataclasses import dataclass
import json
from pathlib import Path
import re
import urllib.error
import urllib.request


def load_api_key(path):
    values=[]
    for line in Path(path).read_text().splitlines():
        line=line.strip()
        if line.startswith("RUNPOD_API_KEY="):
            value=line.split("=",1)[1].strip()
            if len(value)>=2 and value[0]==value[-1] and value[0] in "\"'":value=value[1:-1]
            values.append(value)
    if len(values)!=1 or not values[0] or any(c.isspace() for c in values[0]):
        raise ValueError("expected one nonempty RUNPOD_API_KEY entry")
    return values[0]


class RunpodClient:
    def __init__(self,key,pod_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}",pod_id):raise ValueError("invalid pod id")
        self.key,self.pod_id=key,pod_id

    def request(self,action=None):
        url="https://api.runpod.io/v2/pods/"+self.pod_id+("/action" if action else "")
        data=json.dumps({"action":"stop"}).encode() if action else None
        request=urllib.request.Request(url,data=data,headers={"Authorization":"Bearer "+self.key,"Content-Type":"application/json"})
        try:
            with urllib.request.urlopen(request,timeout=20) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            # Do not print response bodies, headers, credentials, or Pod environments.
            raise RuntimeError("Runpod API HTTP "+str(exc.code)) from None

    def get(self):return self.request()
    def stop(self):return self.request("stop")


@dataclass(frozen=True)
class StopPolicy:
    started_at: float
    max_seconds: float
    max_usd: float
    hourly_rate: float

    def __post_init__(self):
        import math
        if any(not math.isfinite(v) or v<=0 for v in (self.started_at,self.max_seconds,self.max_usd,self.hourly_rate)):
            raise ValueError("invalid supervisor budget")

    def reason(self,now,completed=False):
        elapsed=max(0,now-self.started_at)
        if elapsed>=self.max_seconds:return "hard_deadline"
        if elapsed*self.hourly_rate/3600>=self.max_usd:return "cost_envelope"
        if completed:return "batch_finished"
        return None
