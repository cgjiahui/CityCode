# Source and Data Distribution

The lightweight source archive contains code, guides, configurations, small catalogs, download tools and the prebuilt viewer frontend. Building models, maps, observations, prediction caches, neural weights and offline dependency wheels are separate resource downloads. The source candidate is prepared for publication. Resource archives have not been uploaded to a public host, and their download base URL is still pending.

For resource locations, see [data/README.md](../data/README.md). For exact package sizes and deduplication totals, see [DATA_REPORT.md](DATA_REPORT.md). The machine-readable entry points are [catalog.json](catalog.json), `manifests/` and [downloads.json](downloads.json).

## Choose a profile

| Profile | Contents | Intended use |
|---|---|---|
| `scene` | Structure, completed dual-LOD buildings, maps, heights, layout inputs and preview models | View the completed city and prepare UE assembly |
| `replay` | Scene resources plus observations, prediction caches and frozen building/background process inputs | Replay supplied results |
| `regeneration-inputs` | Structure and 550 saved generation units; only eight low-LOD component-prior meshes | Check regeneration inputs without the full completed building set |
| `generate` | Regeneration inputs, building-network weights and offline dependencies | Generate new building assets on a prepared GPU, then review them |
| `depth` | The 79 observations and full depth-network checkpoint | Run new depth prediction |
| `all` | Every resource package | Complete offline resources, excluding local UE build copies |

Neural weights are distinct from generated building GLBs. Viewing the supplied scene does not load network weights or run inference. Existing assets provide 0.1M and 0.25M variants; newly exported 1M assets require separate review.

Inspect the actual package closure and download size before installation:

```sh
python release/bundles.py plan --profile scene
python release/bundles.py plan --profile replay
python release/bundles.py plan --package generation-weights
```

## Download layout

```text
source archive/
  release/catalog.json              Package sizes, profiles and dependencies
  release/manifests/<package>.json   Install paths, byte counts and SHA256 values
  release/downloads.json             Archive-part byte counts and SHA256 values
  release/bundles.py                 Standard-library download/install tool

resource hosting root/
  downloads.json
  observations/part-00000.tarpart
  buildings/part-00000.tarpart
  buildings/part-00001.tarpart
  maps/part-00000.tarpart
  ...
```

Parts default to at most 1 GiB and may split a large model across parts. Packages store content by SHA256. Shared large objects normally have one owning download package; eight small component priors are repeated to keep regeneration inputs independent of the full building-model download.

The installer checks both archive parts and individual objects, then restores the existing relative `data/`, `models/` and wheelhouse paths. Existing code and binding references continue to work. HTTP Range resume, completed-part reuse and repeat installations are supported. Servers without Range support restart the affected part. Different existing target files are rejected rather than overwritten.

Restored files are copied, so editing one installed path does not change another path through a shared hard link. Reserve disk space for archive parts, the object cache and restored files. Use persistent disk, not a memory-backed temporary directory. The download cache can be removed after installation.

## Install resources

A publisher must provide the resource-hosting base URL. Set `CITYCODE_DATA_BASE_URL` to that URL, then run:

```sh
python release/bundles.py install --profile scene \
  --downloads release/downloads.json \
  --base-url "$CITYCODE_DATA_BASE_URL"
python web_preview/serve.py --port 8794
```

`--target` selects another complete source directory; `--cache` selects the persistent cache location. The prepared source archive includes `web_preview/dist/`, so it can serve the frontend directly. Rebuilding edited frontend source is covered by the [preview guide](../web_preview/README.md).

The public base URL is not configured yet. Local archives use the same installer:

```sh
python release/bundles.py install --profile scene \
  --downloads release/downloads.json --base-url release/artifacts/data \
  --target /absolute/path/to/another/CityCode \
  --cache /absolute/path/to/disk/cache
```

## GitHub Releases option

The files can be distributed through this project's Releases tab without adding them to Git history. Release assets use flat names: the prepared [asset plan](GITHUB_RELEASE_ASSETS.json) maps local archive paths to unique upload names, and [downloads.github-releases.json](downloads.github-releases.json) is the corresponding installer index. It retains the original part sizes and SHA256 checksums. Use that index with `--downloads` and the release download base URL with `--base-url` once the assets are published. The 60 parts are not uploaded yet; while this repository is private, its release assets also require repository access.

## Prepare a release

These operations create local files; they do not upload resources, call image services or allocate GPUs. Use new output directories for immutable archive snapshots.

```sh
# Re-index actual files after approved changes.
python release/bundles.py index --output release

# Build all packages; users can still download only a selected profile.
python release/bundles.py pack --profile all --output release/artifacts/data

# Ship the small part index with the source archive.
cp release/artifacts/data/downloads.json release/downloads.json
python release/bundles.py source --output release/artifacts/CityCode-source.tar.gz
```

For a new archive snapshot, `pack --reuse /path/to/previous/archives` verifies and reuses unchanged package parts. It still checks object membership and archive-part hashes. Changing documents or control files requires refreshing the relevant proofs/catalogs and rebuilding affected packages; the runtime never silently accepts changed input identities.

The source archive excludes data binaries, weights, wheelhouses, virtual environments, local build/run outputs, old wheels and Node dependencies. The prebuilt `web_preview/dist/` is the specific exception to the build-output exclusion.

GitHub is suitable for the lightweight source. Resource parts can be attached to a GitHub Release without adding them to Git history, or hosted in a Hugging Face dataset repository or versioned HTTPS object storage; weights can also use a separate model repository. A private GitHub repository also restricts access to its release assets. Each prepared part is at most 1 GiB, below the 2 GiB release-asset limit. The installer accepts the corresponding hosting base without binding the resource format to one provider.

See the [GitHub file-size policy](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github) and [Hugging Face storage policy](https://huggingface.co/docs/hub/storage-limits). Project and third-party distribution terms remain as documented in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Building models and local UE copies

Published building GLBs are installed under `data/smallcity_p384_building_assets_completed/`; the `buildings` package supplies them. Stage 02 can instead produce a newly reviewed binding, which stage 04 accepts through `--binding`.

The old local CityCode snapshot references 573 `build/` files: 560 building import copies, background conversion/filter/partition outputs, a small axis fixture and five assembly metadata files. Public CPU preparation code regenerates those import copies from the supplied source models and maps.

Consequently, the old machine-specific `smallcity_p384_citycode/` snapshot is excluded from data downloads. Stage 04 prepare, Windows import and finalize create the user's own import files and CityCode scene. Native Epic assets are resolved from the user's City Sample project.

## Records and maintenance

The completed buildings and replay caches share many identical meshes, images and review records. Download deduplication preserves their logical paths. The superseded lightweight preview is excluded; the 60 background responses and required decisions, inputs and provenance remain available in the appropriate profile.

Current checks and limits are in [STATUS.json](STATUS.json). [LOCAL_DEPENDENCIES.json](LOCAL_DEPENDENCIES.json) records the local snapshot audit, and [CONTROL_REFRESH.json](CONTROL_REFRESH.json) records the earlier control-proof repair. Maintainers may explicitly run `python release/refresh_controls.py --write` after reviewing control-file changes; it does not authorize changes to data or weight binaries.

Public execution consumes supplied resources. Historical annotation-preparation/evaluation tools are outside the entry points, while the actual provenance of supplied example resources remains recorded.
