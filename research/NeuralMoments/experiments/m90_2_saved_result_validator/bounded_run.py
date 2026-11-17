"""CPU-only process-group runner; fresh output directory; 120 s / 1 GiB caps."""
import argparse,json,os,signal,subprocess,sys,time,resource
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    def limits():
        resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
        resource.setrlimit(resource.RLIMIT_CPU,(120,120))
        resource.setrlimit(resource.RLIMIT_FSIZE,(16*1024**2,16*1024**2))
        # OS-level two-CPU affinity also covers libraries ignoring thread env vars.
        cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,cpus[:2])
    start=time.monotonic();reason=None
    with (a.out/'stdout.log').open('xb') as log:
        child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limits)
        while child.poll() is None:
            size=sum(f.stat().st_size for f in a.out.rglob('*') if f.is_file())
            if time.monotonic()-start>=120:reason='120s subprocess deadline'
            elif size>=1024**3:reason='1GiB output limit'
            if reason:os.killpg(child.pid,signal.SIGKILL);break
            time.sleep(.05)
        child.wait()
    size=sum(f.stat().st_size for f in a.out.rglob('*') if f.is_file())
    report={'command':command,'exit_code':child.returncode,'status':'pending' if reason else 'pass' if child.returncode==0 else 'fail','stop_reason':reason,'seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,'output_bytes':size,'memory_limit_bytes':1024**3,'wall_limit_seconds':120,'cpu_affinity_count':2,'thread_environment':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}}
    (a.out/'run.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    return 2 if reason else child.returncode
if __name__=='__main__':sys.exit(main())
