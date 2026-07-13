# P6_hpc — Re-anchored HPC Congestion-Monitoring Case Study

Rebuilds the Section V-B HPC congestion case study on the **public** Monet Blue
Waters 2017 dataset and adds quantitative evaluation, addressing Reviewer 1's
point 4 ("case studies are qualitative only — no clustering-quality metrics, no
runtime table vs Monet, no accuracy against the ground truth in [28]") and
Reviewer 2's request for stronger, reproducible evaluation.

## Why re-anchor

The figures in the current submission use snapshot timestamps `@1398220860…`,
which decode to **23 Apr 2014** — three years outside the only public release of
this data (**Jan–May 2017**). Re-anchoring on the public dataset makes both the
telemetry *and* the congestion-protection-event markers reproducible from a
citable DOI.

## Dataset

**Monet – Blue Waters Network Dataset**, Illinois Data Bank,
DOI [10.13012/B2IDB-2921318_V1](https://doi.org/10.13012/B2IDB-2921318_V1).
27,648 compute nodes on a 24×24×24 Cray Gemini 3D torus; LDMS/gpcdr counters at
60-second intervals. Released schema (one row per node-sample), from
`github.com/CSLDepend/monet` `src/HEADER`:

```
#Time,
{Z-,Z+,Y-,Y+,X-,X+}_SAMPLE_GEMINI_LINK_CREDIT_STALL (% x1e6),
{...}_INQ_STALL (% x1e6),
{...}_USED_BW  (% x1e6),
nettopo_mesh_coord_Z, nettopo_mesh_coord_Y, nettopo_mesh_coord_X
```

`#Time` is a Unix epoch (America/Chicago). Stall values are **percent × 1e6**
(divide by 1e6 to get PTs in %). Percent-Time-Stalled (PTs) = credit-stall.

### Downloading (you do this once, outside this sandbox)

The release is a single **140 GB** `dataset.tar.gz` plus `README.md`, `HEADER`,
`License.txt`. It is not fetchable here, so download it on a workstation/cluster:

1. Open the dataset page and click **“Open in Globus”** (needs a free Globus
   account + Globus Connect Personal for your machine), or use the direct file
   link. Globus is strongly preferred for a file this size.
2. After transfer, extract only the day(s) you need:
   ```bash
   tar -tzf dataset.tar.gz | head            # inspect the YYYYMMDD layout
   tar -xzf dataset.tar.gz path/to/20170315  # extract selected day files
   ```
3. Pick a few snapshots. Good choices are timestamps where a **congestion-
   protection event (CPE)** was logged (see CPE cross-verification below), plus
   some random control timestamps.
4. Place the extracted files somewhere this project can read, and pass them to
   `run_experiment.py real` as `PATH:TIMESTAMP` specs (epoch seconds).

> If the CPE / `xtnlrd` event log is **not** bundled in the tarball (the public
> file list shows only counters + README), request it from the dataset authors
> (UIUC DEPEND group) or reconstruct approximate event times from sustained
> high-PTs snapshots. The pipeline runs fine without it — CPE cross-check is
> optional.

### Finding CPE days

A real CPE is logged by Cray's `xtnlrd` into the **netwatch logs** (time +
location) when the stall-to-flit ratio crosses a high-watermark — but those logs
are not in the public counter release. Two ways to find CPE days:

1. **Netwatch logs (exact).** Request them from the dataset authors, or use the
   dated congestion examples in [28] / the HOTI'19 / DSN'18 papers. Feed the
   `(time, x, y, z)` events to `cpe_crosscheck.py`.
2. **From the counters (self-contained).** A CPE leaves a signature: many links
   sit at High PTs for a sustained window. `cpe_finder.py` scans downloaded day
   files and ranks candidate windows:
   ```bash
   python cpe_finder.py /data/20170315 /data/20170316 \
       --high-pts 25 --area-frac 0.02 --min-duration 3 --out cpe_candidates.csv
   ```
   Use each `peak_time` as a `PATH:TIMESTAMP` snapshot for `run_experiment.py
   real`. Note (per [28]) only ~8% of high-congestion events actually triggered a
   logged CPE, so these are severe-congestion *candidates* (RCEs); use path 1 for
   a strict CPE match.

## Modules

| file | purpose |
|------|---------|
| `config.yml` | all paths, thresholds (`th_close=2`, `th_similarity=4%`, `σ=20`), torus dims, congestion bands |
| `torus.py` | torus hop-distance, neighbour pairs, adjacency (wrap-around) |
| `data_loader.py` | load a 60-s snapshot → per-node table (coords + directional PTs) |
| `synthetic.py` | Monet-style synthetic congestion generator (Route A ground truth) |
| `baselines.py` | K-means + faithful Monet region-growing |
| `logiknet_cluster.py` | LogiK-Net logic-guided clustering (LTNtorch, 4 axioms) |
| `metrics.py` | internal indices, synthetic overlap/precision/recall/ARI/NMI, constraint satisfaction |
| `cpe_finder.py` | scan counter telemetry for CPE-candidate days/timestamps |
| `cpe_crosscheck.py` | external validation vs congestion-protection-event log |
| `run_experiment.py` | orchestrator → result tables (mean ± std) + JSON |

## Running

Route A — synthetic ground-truth benchmark (no download needed):
```bash
cd P6_hpc
python run_experiment.py synthetic --config config.yml --out results/
```

Route B — real snapshots (after downloading), internal indices + constraint
satisfaction + optional CPE cross-check:
```bash
python run_experiment.py real --config config.yml \
    --snapshots /data/20170315:1489557600 /data/20170316:1489644000 \
    --cpe /data/cpe_log.csv --out results/
```

## Dependencies

`numpy scipy scikit-learn pandas pyyaml` for everything except LogiK-Net;
`torch` + `LTNtorch` (the repo's existing stack) to include LogiK-Net. Without
torch, the orchestrator runs K-means + Monet only and says so.

## What each evaluation answers

- **Route A (synthetic)** — the only way to report *accuracy* here, exactly as
  Monet does in its Appendix D. Reproduces their generator on the same 24³ torus
  and reports overlap score / precision / recall (their metrics) plus ARI / NMI.
- **Route B (internal indices)** — silhouette, Davies-Bouldin, Calinski-Harabasz
  on real snapshots; needs no labels. Paired with **constraint satisfaction** so
  a constraint-aware method is not judged only by a constraint-blind,
  K-means-favouring yardstick.
- **CPE cross-verification** — matches detected high-congestion clusters to
  logged congestion-protection events in time and torus location: an external,
  model-independent signal that congestion truly occurred.
