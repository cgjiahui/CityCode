"""Opt-in validation top-k retention; standard library only (also used by live watcher)."""
import fcntl
import json
import math
import os
from pathlib import Path
import time
import zipfile

PREFIXES = ("denoiser_step", "denoiser_ema0.9999_step", "misc_step")

def fingerprint(path):
    s = path.stat()
    return (s.st_ino, s.st_size, s.st_mtime_ns)

def complete_files(run, step, min_age):
    files = [run / "ckpts" / f"{prefix}{step:07d}.pt" for prefix in PREFIXES]
    snapshots = []
    for path in files:
        if path.is_symlink() or not path.is_file():
            return None
        sig = fingerprint(path)
        if sig[1] <= 0 or time.time() - path.stat().st_mtime < min_age:
            return None
        # torch.save ZIP central directory is written last. No multi-GB tensor load.
        try:
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                if not any(n.endswith('/data.pkl') for n in names):
                    return None
                if not any('/data/' in n for n in names):
                    return None
        except (OSError, zipfile.BadZipFile):
            return None
        if fingerprint(path) != sig:
            return None
        snapshots.append(sig)
    return files, snapshots

def retain(run, *, apply=False, min_age=60):
    run = Path(run).resolve()
    policy_path = run / "checkpoint_retention_policy.json"
    if not policy_path.exists():
        return {"status": "disabled"}
    policy = json.loads(policy_path.read_text())
    if policy.get("enabled") is not True:
        return {"status": "disabled"}
    assert policy['metric'] == 'val_loss' and policy['mode'] == 'min'
    keep = int(policy['keep'])
    assert keep > 0
    with (run / '.checkpoint_retention.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        history_path = run / 'validation_history.json'
        history_bytes = history_path.read_bytes()
        history = json.loads(history_bytes)
        assert isinstance(history, list)
        rows = {}
        for row in history:
            step = int(row.get('step', 0))
            value = row.get('val_loss')
            if step <= 0 or row.get('metric_scope') != 'validation':
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            if row.get('validation_protocol') != policy['validation_protocol']:
                continue
            rows[step] = row
        eligible = {}
        for step, row in rows.items():
            group = complete_files(run, step, min_age)
            if group:
                eligible[step] = (row, group)
        ranked = sorted(eligible, key=lambda s: (rows[s]['val_loss'], s))
        kept, removed = ranked[:keep], ranked[keep:]
        report = {'time': time.time(), 'run': str(run), 'apply': apply,
                  'metric': 'val_loss ascending; tie: lower step',
                  'kept': [{'step': s, 'val_loss': rows[s]['val_loss']} for s in kept],
                  'removed': [], 'deleted_bytes': 0}
        for step in removed:
            files, sigs = eligible[step][1]
            report['removed'].append({'step': step, 'val_loss': rows[step]['val_loss'],
                                     'files': [str(p) for p in files],
                                     'bytes': sum(sig[1] for sig in sigs)})
        if apply and removed:
            # Recheck both the selection inputs and every file before any deletion.
            if history_path.read_bytes() != history_bytes:
                return {'status': 'retry_history_changed'}
            for step in ranked:
                files, sigs = eligible[step][1]
                if any(fingerprint(p) != sig for p, sig in zip(files, sigs)):
                    return {'status': 'retry_checkpoint_changed'}
            audit = run / 'checkpoint_retention_audit.jsonl'
            with audit.open('a') as f:
                f.write(json.dumps({'event': 'before_delete', **report}) + '\n')
                f.flush()
                os.fsync(f.fileno())
            for item in report['removed']:
                for name in item['files']:
                    path = Path(name)
                    size = path.stat().st_size
                    path.unlink()
                    report['deleted_bytes'] += size
            with audit.open('a') as f:
                f.write(json.dumps({'event': 'after_delete', **report}) + '\n')
                f.flush()
                os.fsync(f.fileno())
        if apply:
            target = run / 'checkpoint_retention_status.json'
            tmp = target.with_suffix('.tmp')
            tmp.write_text(json.dumps(report, indent=2) + '\n')
            os.replace(tmp, target)
        return report
