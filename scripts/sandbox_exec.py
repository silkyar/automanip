"""Apply a root-authored Linux Landlock path allowlist, then exec one command."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import sys

READ_EXEC = 1 | 4 | 8
ALL_FS = (1 << 15) - 1  # Through TRUNCATE; requires ABI >= 3.


def restrict(policy):
    if sys.platform != "linux": raise RuntimeError("Linux Landlock required")
    libc = ctypes.CDLL(None, use_errno=True)
    def syscall(number, *args):
        result=libc.syscall(number,*args)
        if result < 0: raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
        return result
    abi=syscall(444,0,0,1)
    if abi<3: raise RuntimeError("Landlock ABI >= 3 required")
    class Ruleset(ctypes.Structure):
        _fields_=[("handled_access_fs",ctypes.c_uint64)]
    class Beneath(ctypes.Structure):
        _pack_=1
        _fields_=[("allowed_access",ctypes.c_uint64),("parent_fd",ctypes.c_int32)]
    attr=Ruleset(ALL_FS)
    ruleset=syscall(444,ctypes.byref(attr),ctypes.sizeof(attr),0)
    try:
        for key,access in (("read_execute",READ_EXEC),("read_write",ALL_FS)):
            for path in policy[key]:
                if not Path(path).is_dir():continue
                fd=os.open(path,os.O_PATH|os.O_CLOEXEC)
                try:
                    rule=Beneath(access,fd)
                    syscall(445,ruleset,1,ctypes.byref(rule),0)
                finally:os.close(fd)
        if libc.prctl(38,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),"no_new_privs failed")
        syscall(446,ruleset,0)
    finally:os.close(ruleset)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy",type=Path,required=True)
    p.add_argument("command",nargs=argparse.REMAINDER)
    args=p.parse_args();command=args.command
    if command and command[0]=="--":command=command[1:]
    if not command:raise ValueError("command required")
    policy=json.loads(args.policy.read_text())
    restrict(policy)
    os.execvpe(command[0],command,os.environ.copy())


if __name__=="__main__":main()
