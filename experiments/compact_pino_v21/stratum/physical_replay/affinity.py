"""Pin each fork worker once after scientific runtimes install their fork-time CPU masks."""
import argparse
import os
from pathlib import Path
from time import sleep, monotonic


SLOTS={"maskB":(4,5,12,13),"combB":(6,7,14,15),"R01":(8,9,16,17,24),"R02":(10,11,18,19,26)}


def configure(out, wait_seconds):
    pending=set(SLOTS)
    end=monotonic()+wait_seconds
    while pending and monotonic()<end:
        for key in tuple(pending):
            expected={"maskB":25,"combB":23,"R01":20,"R02":21}[key]
            if len(list((out/"centers").glob(key+"_*.json")))==expected:
                pending.remove(key)
        for proc in Path("/proc").iterdir():
            if not proc.name.isdecimal():
                continue
            try:
                cmd=(proc/"cmdline").read_bytes().split(b"\0")
                args=[os.fsdecode(arg) for arg in cmd]
                if not any(Path(a).name == "replay_registered.py" for a in args) or "--workers" not in args:
                    continue
                if Path(args[args.index("--out")+1]).resolve()!=out or "--asset" not in args:
                    continue
                key=args[args.index("--asset")+1]
                if key not in pending:
                    continue
                children=sorted(int(p) for p in (proc/"task"/proc.name/"children").read_text().split())
                slots=SLOTS[key]
                if len(children)!=len(slots):
                    continue
                for pid,cpu in zip(children,slots,strict=True):
                    os.sched_setaffinity(pid,{cpu})
                print(key,list(zip(children,slots)),flush=True)
                pending.remove(key)
            except (FileNotFoundError,ProcessLookupError,PermissionError):
                continue
        if pending:
            sleep(1)
    if pending:
        raise SystemExit(f"could not find all launched worker pools: {sorted(pending)}")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    p.add_argument("--wait",type=int,default=180)
    args=p.parse_args()
    configure(args.out.resolve(),args.wait)
