"""Build the Chat audit packet from committed bytes and verify it without a Git checkout."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from experiments.compact_pino_v21.make_review_bundle import ADAPTER, STUDY, selected

SUB=STUDY+'/stratum'
REQUIRED={SUB+'/'+p for p in ('README.md','CHECKLIST.md','protocol.json','restricted_targets.py',
          'review_check.py','evidence/restricted_targets.npz','evidence/restricted_targets.json',
          'evidence/prototype.json','evidence/noodles_event_comparison.json','evidence/event_scale_example.json',
          'physical_replay/evidence/registered_physical_replay.json.gz','physical_replay/evidence/summary.json',
          'physical_replay/evidence/manufactured_plane.json','qualification/evidence/neighbourhoods.json',
          'qualification/native_admission/evidence/replay_inputs.zip','solver_bridge/receipt.json','solver_bridge/replay.py',
          'native_backend/candidate03-stratum.patch','native_backend/build_receipt.json','evidence/regression_tests.json','evidence/decision.json')}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def check(path):
    with zipfile.ZipFile(path) as packet:
        files=[i.filename for i in packet.infolist() if not i.is_dir()]
        if (len(files)!=len(set(files)) or not REQUIRED.issubset(files) or
                any(Path(s).is_absolute() or '..' in Path(s).parts for s in files) or packet.testzip()):
            raise ValueError('invalid ZIP scope, missing members, duplicate name or failed CRC')
        sums=packet.read('SHA256SUMS').decode().splitlines()
        actual=[sha(packet.read(name))+'  '+name for name in sorted(set(files)-{'SHA256SUMS'})]
        if actual!=sums or packet.read('START_HERE.md')!=packet.read(SUB+'/README.md'):
            raise ValueError('review packet members differ from the manifest')
    return sha(path.read_bytes())


def build(path,ref):
    def git(*args):
        return subprocess.check_output(['git','-C',str(ROOT),*args],env={**os.environ,'TZ':'UTC'})
    commit=git('rev-parse','--verify',ref+'^{commit}').decode().strip()
    candidates=git('ls-tree','-r','--name-only','-z',commit,'--',STUDY,ADAPTER,'tests/compact_pino_v21')
    names=sorted(s for s in candidates.decode().split('\0') if s and selected(s) and not s.startswith(SUB+'/review/'))
    if not REQUIRED.issubset(names):
        raise ValueError('missing committed evidence: '+', '.join(sorted(REQUIRED-set(names))))
    path.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temp:
        target=Path(temp)/'review.zip'
        git('archive','--format=zip','-9','-o',str(target),commit,'--',*names)
        with zipfile.ZipFile(target,'a') as packet:
            if set(i.filename for i in packet.infolist() if not i.is_dir())!=set(names):
                raise ValueError('Git archive member selection mismatch')
            def add(name,data):
                entry=zipfile.ZipInfo(name,(1980,1,1,0,0,0))
                entry.external_attr=0o100644<<16
                entry.create_system=3
                packet.writestr(entry,data,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
            add('SOURCE_COMMIT',(commit+'\n').encode())
            add('START_HERE.md',packet.read(SUB+'/README.md'))
            sums=''.join(sha(packet.read(name))+'  '+name+'\n' for name in sorted(names+['SOURCE_COMMIT','START_HERE.md']))
            add('SHA256SUMS',sums.encode())
        checksum=check(target)
        target.replace(path)
    return checksum


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path',type=Path)
    parser.add_argument('--source',default='HEAD')
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    print(check(args.path) if args.verify else build(args.path.resolve(),args.source))
