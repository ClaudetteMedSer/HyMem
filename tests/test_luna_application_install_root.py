"""Independent local emulation of source installer, never SSH or inference."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from tools.diagnostics import luna_application_fault_bundle_v1 as bundle
from tools.diagnostics import luna_application_fault_install_v1 as install
from tools.diagnostics import luna_application_fault_install_v2 as install2

ROOT=Path(__file__).resolve().parents[1]
FROZEN=Path('/private/tmp/hymem-lme-instrumented-IjmZdT/bundle')
CANDIDATE=Path('/private/tmp/hymem-staged-proxy-v1-8_ac647k/bundle')


@pytest.fixture(scope='module',params=[install,install2])
def payload(tmp_path_factory,request):
    work=tmp_path_factory.mktemp('root-application-installer')
    target=work/'bundle'
    bundle.assemble(repo=ROOT,accepted_code=FROZEN/'code',candidate=CANDIDATE/'candidate',
        map_path=CANDIDATE/'source-map.json',output=target)
    installer=request.param
    raw,manifest=installer.archive_bytes(target,ROOT/'tools/diagnostics')
    return target,raw,manifest,installer


def script_for(raw,manifest,home,installer):
    pins={key:manifest[key] for key in installer.HELPERS}
    script=installer.REMOTE
    for key,value in {'__ARCHIVE_B64__':repr(base64.b64encode(raw).decode()),
        '__ARCHIVE_SHA__':repr(hashlib.sha256(raw).hexdigest()),
        '__RUNNER_SHA__':repr(installer.RUNNER_SHA),'__MAP_SHA__':repr(installer.MAP_SHA),
        '__MAP_PIN__':repr(installer.MAP_PIN),'__HELPER_PINS__':repr(pins)}.items():
        script=script.replace(key,value)
    # Only host identity/location are substituted. Archive validation and writes
    # remain the exact generated program, in a fresh local test directory.
    script=script.replace('Path("/home/atta")','Path('+repr(str(home))+')')
    script=script.replace('from pathlib import Path','from pathlib import Path\nsys.platform="linux"\nos.getuid=lambda:1000\nos.geteuid=lambda:1000')
    return script


def run(script):
    return subprocess.run([sys.executable,'-I','-B','-'],input=script,text=True,
        capture_output=True,timeout=25)


def test_exact_remote_installer_creates_only_private_source_files(payload,tmp_path):
    _,raw,manifest,installer=payload
    result=run(script_for(raw,manifest,tmp_path,installer))
    assert result.returncode==0,result.stderr[-4000:]
    metadata=json.loads(result.stdout);assert metadata['model_calls']==0
    root=Path(metadata['root']);assert root.parent==tmp_path and root.stat().st_mode&0o777==0o700
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    assert actual==set(manifest)
    for relative,digest in manifest.items():
        path=root/relative
        assert path.stat().st_mode&0o777==0o600
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    assert not (root/'launch-receipt.json').exists()


@pytest.mark.parametrize('mutation',['duplicate','traversal','symlink','drift','extra'])
def test_tampered_archive_rejected_before_root_creation(payload,tmp_path,mutation):
    _,raw,manifest,installer=payload
    source=tarfile.open(fileobj=io.BytesIO(raw),mode='r:')
    output=io.BytesIO()
    with tarfile.open(fileobj=output,mode='w') as target:
        for index,member in enumerate(source.getmembers()):
            data=source.extractfile(member).read()
            if index==1 and mutation=='traversal':member.name='../escape.py'
            if index==1 and mutation=='symlink':member.type=tarfile.SYMTYPE;member.linkname='/etc/passwd';member.size=0;data=b''
            if index==1 and mutation=='drift':data=b'X'+data[1:]
            target.addfile(member,io.BytesIO(data))
            if index==1 and mutation=='duplicate':target.addfile(member,io.BytesIO(data))
        if mutation=='extra':
            extra=tarfile.TarInfo('bundle/extra.py');extra.size=1;target.addfile(extra,io.BytesIO(b'x'))
    result=run(script_for(output.getvalue(),manifest,tmp_path,installer))
    assert result.returncode!=0
    assert list(tmp_path.iterdir())==[]


def test_generated_script_contains_no_launch_or_provider_dispatch(payload,tmp_path):
    target,_,_,installer=payload
    script=installer.render_remote_script(target,ROOT/'tools/diagnostics').decode()
    # The source archive is data, not executing calls. Inspect the remote AST.
    tree=ast.parse(script)
    calls={ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node,ast.Call)}
    assert not calls & {'subprocess.run','subprocess.Popen','exec','eval','os.system'}
    assert 'tempfile.mkdtemp' in calls and 'os.open' in calls
