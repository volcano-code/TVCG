"""Build a deterministic review ZIP from the committed V2.1 tree."""
import argparse
import hashlib
import io
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile
import zipfile


STUDY = 'experiments/compact_pino_v21'
ADAPTER = 'research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter'
REQUIRED = ('README.md','CHECKLIST.md','protocol.json','verify_results.py',
            'native/evidence/native_targets.npz','native/evidence/native_replay.json',
            'calibration/evidence/native_training_receipts.tar.gz','evidence/regression_tests.json',
            'profile_inference/profile.json','profile_inference/summary.md','evidence/gate_status.json')


def selected(name):
    path = PurePosixPath(name)
    if name.startswith(STUDY+'/'):
        return not name.startswith(STUDY+'/review/')
    return (name.startswith('tests/compact_pino_v21/') or
            (str(path.parent)==ADAPTER and (path.name.startswith('v21_') or path.name=='compact_pino_v21.py')) or
            (str(path.parent)==ADAPTER+'/tests' and path.name.startswith('test_v21_')))


def build(output, ref):
    checkout = Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.check_output(['git','-C',str(checkout),*args])
    commit = git('rev-parse','--verify',ref+'^{commit}').decode().strip()
    all_names = git('ls-tree','-r','--name-only','-z',commit,STUDY,ADAPTER,'tests/compact_pino_v21')
    names = sorted(s for s in all_names.decode().split('\0') if selected(s))
    required = {STUDY+'/'+p for p in REQUIRED}
    if not required.issubset(names):
        raise ValueError(f'missing committed results: {sorted(required-set(names))}')
    contents = {'SOURCE_COMMIT':((commit+'\n').encode(),0o644)}
    with tarfile.open(fileobj=io.BytesIO(git('archive','--format=tar',commit,'--',*names)),mode='r:') as archive:
        for member in archive:
            if member.isfile():
                contents[member.name] = (archive.extractfile(member).read(), member.mode)
    if set(contents)!=(set(names)|{'SOURCE_COMMIT'}):
        raise ValueError('Git archive differs from selected file set')
    manifest = ''.join(hashlib.sha256(data).hexdigest()+'  '+name+'\n' for name,(data,_) in sorted(contents.items()))
    contents['SHA256SUMS'] = (manifest.encode(),0o644)
    output = output.resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        finished = Path(temp)/'review.zip'
        with zipfile.ZipFile(finished,'w') as z:
            for name,(data,mode) in sorted(contents.items()):
                info = zipfile.ZipInfo(name, date_time=(1980,1,1,0,0,0))
                info.create_system = 3
                info.external_attr = (0o100000 | (mode & 0o777)) << 16
                z.writestr(info,data,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
        with zipfile.ZipFile(finished) as z:
            if z.testzip() is not None or len(z.namelist())!=len(contents):
                raise ValueError('ZIP member integrity error')
        finished.replace(output)
    return hashlib.sha256(output.read_bytes()).hexdigest()


if __name__=='__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path)
    ap.add_argument('--source',default='HEAD')
    args = ap.parse_args()
    print(build(args.output,args.source), args.output)
