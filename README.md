# CityCode
### Building Agent-Operable 3D Cities from Sparse Aerial Observations

<p align="center">
  <a href="https://cgjiahui.github.io/CityCode/"><img src="media/live-demo.svg" alt="Explore the live 3D demo" height="44" /></a>
  <a href="https://arxiv.org/abs/2610.18342"><img src="media/arxiv-paper.svg" alt="Read the paper on arXiv" height="44" /></a>
</p>

<p align="center"><b><a href="https://cgjiahui.github.io/CityCode/">Explore the interactive city ↗</a></b> · <a href="https://arxiv.org/abs/2610.18342">Paper</a> · <a href="media/citycode-demo.mp4">Video</a> · <a href="#pipeline">Pipeline</a></p>

CityCode turns sparse low-altitude aerial observations into metric 3D cities that agents can generate, edit, query, and navigate. The input starts with **one aerial image per region**, covering approximately **400 m × 400 m**, together with camera parameters.

- Recover building structure and connect generated assets through a shared, versioned city representation.
- Support city extension, block editing, text/image retrieval, navigation, and inspection.
- Assemble scenes with **high-quality visual effects in Unreal Engine** and explore a city in the browser.

## Video

[![Animated excerpts of generation, editing, retrieval, and planning](media/agentic-preview.gif)](media/citycode-demo.mp4)

[**Watch the full 54-second video**](media/citycode-demo.mp4) — generated cities and four categories of agentic tasks, rendered in Unreal Engine. The animation above contains short excerpts from the original presentation video.

## Scene results

![Scene comparisons on synthetic and real-world aerial observations](media/scene-results.webp)

Comparisons on CS-Aerial, Qingdao Hospital, and Shenzhen Campus. Columns show the comparison methods, CityCode, and reference views. Open the image to inspect the details.

## Agentic operations

![City generation, editing, retrieval, and planning examples](media/agentic-operations.webp)

Selected checkpoints show city extension, building replacement, spatial queries, multimodal retrieval, road navigation, and 3D inspection. Operations update a versioned city state that can be validated and restored.

<details>
<summary><b>Method overview</b></summary>

![CityCode method: aerial input, structure recovery, asset generation, city state, and agent interaction](media/method-overview.webp)

Aerial observations → metric structure → structure-guided assets → shared city state → agent interaction.

</details>

## Pipeline

[Structure recovery](workflow/01_perception_structure/README.md) → [Building assets](workflow/02_building_assets/README.md) → [Background and roads](workflow/03_background_roads/README.md) → [CityCode and UE assembly](workflow/04_citycode_assembly/README.md).

```sh
python -m pip install -e .
python release/bundles.py install --profile scene
python web_preview/serve.py --port 8794
```

Research data and weights are distributed through [Hugging Face](https://huggingface.co/datasets/cgjiahui/CityCode). While private, sign in with `hf auth login` before installation. The installer restores the project directories automatically; see the [download and setup guide](release/README.md) and [data inventory](data/README.md). The repository contains the prepared pipeline code, configurations, UE adapters, tests, and preview source; it does **not** contain the large resource bundles. UE assembly also requires a compatible City Sample installation.

## Release status

Source is available in this repository, which is currently private. The HF resource repository is also private; the installer checks that its upload is complete before downloading. Code and data licensing remain to be finalized; existing [third-party notices](THIRD_PARTY_NOTICES.md) apply to their respective components.
