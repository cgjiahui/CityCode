#!/usr/bin/env python3
"""Assemble the Pages site from its frontend and checksum-pinned preview assets."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(1024*1024):
            result.update(chunk)
    return result.hexdigest()


def safe_name(name):
    path=PurePosixPath(name)
    if path.is_absolute() or '\\' in name or any(part in {'','.','..'} for part in name.split('/')):
        raise ValueError('Unsafe archive path')
    return path


def assemble(config,frontend,output,cache):
    if output.exists():
        raise ValueError('Output exists; choose a fresh output directory')
    metadata=json.loads(config.read_text())
    package=metadata['archive']
    cache.mkdir(parents=True,exist_ok=True)
    archive=cache/(package['sha256']+'.tar.gz')
    if not archive.exists() or archive.stat().st_size!=package['bytes'] or digest(archive)!=package['sha256']:
        temporary=archive.with_suffix('.download')
        request=urllib.request.Request(package['url'],headers={'User-Agent':'CityCode-Pages/1.0'})
        with urllib.request.urlopen(request,timeout=60) as incoming,temporary.open('wb') as outgoing:
            shutil.copyfileobj(incoming,outgoing,1024*1024)
        if temporary.stat().st_size!=package['bytes'] or digest(temporary)!=package['sha256']:
            temporary.unlink()
            raise ValueError('Preview archive does not match its pinned SHA256')
        temporary.replace(archive)

    if not (frontend/'index.html').is_file() or (frontend/'city').exists():
        raise ValueError('Pass only the compiled frontend, without city resources')
    shutil.copytree(frontend,output)
    with tarfile.open(archive,'r|gz') as incoming:
        first=incoming.next()
        if first is None or not first.isfile() or first.name!='asset-manifest.json' or first.size>5_000_000:
            raise ValueError('The archive must begin with its resource manifest')
        manifest=json.load(incoming.extractfile(first))
        if manifest['scene_revision']!=metadata['scene_revision']:
            raise ValueError('Unexpected preview scene revision')
        catalog=manifest['files']
        seen=set()
        for member in incoming:
            if member.name=='asset-manifest.json':
                continue
            name=safe_name(member.name)
            if not member.isfile() or len(name.parts)<2 or name.parts[0]!='city':
                raise ValueError('Only regular city resource files are allowed')
            relative=PurePosixPath(*name.parts[1:]).as_posix()
            if relative not in catalog or relative in seen or member.size!=catalog[relative]['bytes']:
                raise ValueError('Unexpected, duplicate or truncated scene resource')
            target=output.joinpath(*name.parts)
            target.parent.mkdir(parents=True,exist_ok=True)
            result=hashlib.sha256()
            with incoming.extractfile(member) as source,target.open('wb') as destination:
                while chunk:=source.read(1024*1024):
                    destination.write(chunk)
                    result.update(chunk)
            if result.hexdigest()!=catalog[relative]['sha256']:
                raise ValueError('A scene resource failed SHA256 verification')
            seen.add(relative)
    if seen!=set(catalog):
        raise ValueError('The preview archive is missing scene resources')
    (output/'.nojekyll').touch()
    total=sum(path.stat().st_size for path in output.rglob('*') if path.is_file())
    if total>=1_000_000_000:
        raise ValueError('The assembled site exceeds the 1 GB Pages budget')
    result={'site_bytes':total,'scene_files_verified':len(seen),'scene_revision':metadata['scene_revision']}
    print(json.dumps(result))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('homepage/preview-assets.json'))
    parser.add_argument('--frontend',type=Path,default=Path('docs'))
    parser.add_argument('--output',type=Path,default=Path('_site'))
    parser.add_argument('--cache',type=Path,default=Path('.cache/citycode-preview'))
    args=parser.parse_args()
    assemble(args.config,args.frontend,args.output,args.cache)
