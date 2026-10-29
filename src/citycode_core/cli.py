"""city-code compile/validate/export-ue/lookup; standard library only."""
import argparse
import json
from pathlib import Path

from .scene import compile_scene, export_ue, load_scene, read, validate


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    c = commands.add_parser("compile")
    c.add_argument("--root", type=Path, default=Path.cwd())
    c.add_argument("--config", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--interchange-root", type=Path)
    for key in ("assembly", "catalog", "binding", "original_binding", "structures", "ground_roads", "elevated_centerlines", "elevated_proposal", "network", "props_groups", "background"):
        c.add_argument("--"+key.replace("_", "-"), type=Path)
    for name in ("validate", "export-ue", "lookup"):
        c = commands.add_parser(name)
        c.add_argument("--scene", type=Path, required=True)
        c.add_argument("--root", type=Path, default=Path.cwd())
        if name == "validate":
            c.add_argument("--verify-files", action="store_true")
        if name == "export-ue":
            c.add_argument("--output", type=Path, required=True)
        if name == "lookup":
            c.add_argument("--id", required=True)
    args = p.parse_args(argv)
    if args.command == "compile":
        config = read(args.config)
        for key in config:
            override = getattr(args, key, None)
            if override is not None:
                config[key] = (override.resolve().relative_to(args.root.resolve()).as_posix() if override.is_absolute() else override.as_posix())
        result = compile_scene(args.root, config, args.output, args.interchange_root)
        result = {k: result[k] for k in ("schema_version", "release_id", "counts", "status")}
    elif args.command == "validate":
        result = validate(args.scene, args.root, args.verify_files)
    elif args.command == "export-ue":
        result = export_ue(args.scene, args.root, args.output)
    else:
        _, layers = load_scene(args.scene)
        result = {"id": args.id, "matches": []}
        for name, rows in (("structure", layers["buildings"]["structures"]), ("entity", layers["buildings"]["entities"]),
                           ("instance", layers["assembly"]["buildings"]), ("native", layers["assembly"]["native"])):
            result["matches"] += [{"kind": name, "record": r} for r in rows if r["id"] == args.id]
        if not result["matches"]:
            p.error("Unknown CityCode ID: "+args.id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
