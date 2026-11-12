"""Build and install a private Qt6 KStars with the native LightPollution module.

Requires Qt6/KF6/INDI and KStars build dependencies. No sudo or system install.
Run with uv run; native dependencies may also live in --dependency-root.
"""

import argparse
import hashlib
import json
import lzma
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from urllib.request import urlopen

from apply_native import prepare
from repair_catalogue import repair,hd_index

REVISION='c46dbb4f0cfe4b8ca190a0c391e975e52a005aa8'
SOURCE_URL='https://api.github.com/repos/KDE/kstars/tarball/'+REVISION
SOURCE_SHA256='ac55ea6d680cbd8dae046c699f7ee821f2ebc5d49cc038df4d87d5a378748a5e'
CATALOGUE_URL='https://free-astro.org/download/kstars-siril-catalogues/deepstars.dat.xz'
CATALOGUE_SHA256='75346d0c1cd58eb237269ff92ecd116accc730dd3c2dcf3c3d412d68eccab213'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def download(url,path,expected):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if digest(path)!=expected:raise ValueError('Cached checksum mismatch: '+str(path))
        return
    temporary=path.with_suffix(path.suffix+'.partial')
    try:
        with urlopen(url,timeout=90) as remote,temporary.open('wb') as stream:
            shutil.copyfileobj(remote,stream)
        if digest(temporary)!=expected:raise ValueError('Download checksum mismatch: '+url)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def native_environment(dependency_root=None):
    env=dict(os.environ)
    if dependency_root:
        root=Path(dependency_root).resolve();prefix=root/'usr'
        lib=prefix/'lib/x86_64-linux-gnu'
        env.update(PATH=str(prefix/'bin')+':'+env['PATH'],
            LD_LIBRARY_PATH=str(lib)+':'+str(prefix/'lib')+':'+env.get('LD_LIBRARY_PATH',''),
            PKG_CONFIG_SYSROOT_DIR=str(root),
            PKG_CONFIG_PATH=str(lib/'pkgconfig')+':'+str(prefix/'share/pkgconfig'))
    return env


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix',type=Path,required=True,help='Private installation directory.')
    parser.add_argument('--source',type=Path,help='Use an existing, explicitly selected KStars source.')
    parser.add_argument('--dependency-root',type=Path,help='Optional relocated native SDK root (containing usr).')
    parser.add_argument('--jobs',type=int,default=1)
    parser.add_argument('--configure-only',action='store_true')
    parser.add_argument('--reconfigure',action='store_true',help='Force CMake configuration instead of reusing an identical one.')
    parser.add_argument('--no-precompiled-headers',action='store_true')
    parser.add_argument('--release-flags',default='-O2 -g0 -DNDEBUG')
    args=parser.parse_args()
    if not 1<=args.jobs<=32:parser.error('--jobs must be in1..32.')
    prefix=args.prefix.resolve();prefix.mkdir(parents=True,exist_ok=True)
    source=args.source.resolve() if args.source else prefix/'sources/KStars'
    if not args.source and not source.exists():
        archive=prefix/'downloads/kstars.tar.gz';download(SOURCE_URL,archive,SOURCE_SHA256)
        unpack=prefix/'sources/unpack';unpack.mkdir(parents=True,exist_ok=True)
        with tarfile.open(archive) as tar:
            tar.extractall(unpack,filter='data')
        directories=list(unpack.iterdir())
        if len(directories)!=1:raise ValueError('Unexpected KStars archive layout.')
        directories[0].rename(source);unpack.rmdir()
    for path,content in prepare(source).items():
        if not path.exists() or path.read_text()!=content:
            temporary=path.with_suffix(path.suffix+'.lightpollution-tmp')
            temporary.write_text(content);temporary.replace(path)
    env=native_environment(args.dependency_root)
    cmake=shutil.which('cmake',path=env['PATH'])
    if not cmake:raise SystemExit('Install CMake and the Qt6/KF6/INDI build dependencies; see docs/KSTARS_INTEGRATION.md.')
    command=[cmake,'-S',str(source),'-B',str(prefix/'build'),'-G','Ninja',
        '-DBUILD_WITH_QT6=ON','-DBUILD_TESTING=OFF','-DENABLE_SENTRY=OFF',
        '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CXX_FLAGS_RELEASE='+args.release_flags,
        '-DLIGHTPOLLUTION_PRECOMPILE_QT='+('OFF' if args.no_precompiled_headers else 'ON'),
        '-DCMAKE_INSTALL_PREFIX='+str(prefix/'installed')]
    if args.dependency_root:
        root=args.dependency_root.resolve()
        command+=['-DCMAKE_PREFIX_PATH='+str(root/'usr'),'-DCMAKE_FIND_ROOT_PATH='+str(root)]
    configured=prefix/'configuration.json'
    configuration={'command':command,'source_cmake_sha256':{str(p.relative_to(source)):digest(p)
        for p in [source/'CMakeLists.txt',source/'kstars/CMakeLists.txt']}}
    cached=json.loads(configured.read_text()) if configured.exists() else None
    if args.reconfigure or configuration!=cached or not (prefix/'build/CMakeCache.txt').exists():
        subprocess.run(command,env=env,check=True)
        configured.write_text(json.dumps(configuration,indent=2)+'\n')
    if args.configure_only:return
    # Fail early on native API compatibility before compiling the entire app.
    subprocess.run([cmake,'--build',str(prefix/'build'),'--target',
        'kstars/CMakeFiles/KStarsLib.dir/tools/lightpollutiondock.cpp.o','--parallel','1'],env=env,check=True)
    # ALL also generates the city database and other installed data targets.
    subprocess.run([cmake,'--build',str(prefix/'build'),'--target','all','--parallel',str(args.jobs)],env=env,check=True)
    subprocess.run([cmake,'--install',str(prefix/'build')],env=env,check=True)
    target=prefix/'catalogues/share/kstars';target.mkdir(parents=True,exist_ok=True)
    report=prefix/'catalogue-repair.json';catalogue=target/'deepstars.dat'
    manifest=prefix/'installation.json'
    existing=json.loads(manifest.read_text()) if manifest.exists() else {}
    fixed_hashes={n:digest(source/'kstars/data'/n) for n in ['namedstars.dat','unnamedstars.dat']}
    reusable=(existing.get('fixed_catalogues')==fixed_hashes and catalogue.exists() and report.exists()
              and existing.get('prepared_catalogue_sha256')==digest(catalogue))
    if not reusable:
        archive=prefix/'downloads/deepstars.dat.xz';download(CATALOGUE_URL,archive,CATALOGUE_SHA256)
        with lzma.open(archive) as remote,catalogue.open('wb') as stream:shutil.copyfileobj(remote,stream)
        repair(source,catalogue,report)
    for name in ['namedstars.dat','unnamedstars.dat','starnames.dat']:
        shutil.copy2(source/'kstars/data'/name,target/name)
    index=hd_index(source,catalogue,target/'Henry-Draper.idx')
    repaired=json.loads(report.read_text())
    catalogue_metadata={'schema_version':1,'maximum_pm_mas_yr':repaired['max_proper_motion_mas_yr'],
        'source_files':{**fixed_hashes,'deepstars.dat':digest(catalogue)},'HD_index':index}
    (target/'catalogue-provenance.json').write_text(json.dumps(catalogue_metadata,indent=2)+'\n')
    project=Path(__file__).resolve().parents[2]
    record={'schema_version':1,'project':str(project),'python':sys.executable,
        'kstars_binary':str(prefix/'installed/bin/kstars'),'source':str(source),
        'source_revision':REVISION if not args.source else 'explicit_checkout_see_source_hashes',
        'dependency_root':str(args.dependency_root.resolve()) if args.dependency_root else None,
        'fixed_catalogues':fixed_hashes,'prepared_catalogue_sha256':digest(catalogue),
        'HD_index':index,
        'binary_sha256':digest(prefix/'installed/bin/kstars'),'Qt_major_requested':6,
        'native_source_sha256':{p.name:digest(p) for p in prepare(source)},
        'system_KStars_replaced':False}
    manifest.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print('Installed. Start backend and KStars together with:')
    print('uv run python integrations/kstars/launch.py --installation',prefix)


if __name__=='__main__':main()
