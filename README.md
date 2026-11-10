# Ultra-KMeans

Ultra-KMeans is a shared-memory parallel k-means system written in C++20. It is the implementation behind the paper *UltraKMeans: Fully Scalable and High-Quality $k$-Means* (PPoPP 2027).

## Features

- Five seeding algorithms:
  | Seeder name | `-seed` value |
  | --- | --- |
  | Doubling k-means | `PrefixDoubling` |
  | Ward's HAC | `Wards` |
  | Random | `UniformlyRandom` |
  | k-means++ | `SequentialPlusPlus` |
  | k-means‖ | `ParallelPlusPlus` |
- Lloyd's refinement with approximate nearest-neighbor assignment (`-dist
  ANNS`, a Vamana graph over the centers with warm-started searches) or exact
  assignment (`-dist Pairwise`).
- Float32, uint8, and int8 input vectors.
- Aggregate and per-iteration TSV output.

## Prerequisites

| Requirement | Version | Notes |
| --- | --- | --- |
| Operating system | 64-bit Linux, x86-64 | Tested on Ubuntu 24.04. |
| C++ compiler | **GCC 15** (default) | Another C++20 compiler can be selected with `CXX`. |
| Build system | CMake ≥ 3.16 **or** Bazel **8.x** | The two builds are equivalent and use the same compiler; see the notes below. |
| `numactl` | any | Recommended on multi-socket machines. |
| `git` | any | To fetch the dependency submodules. |

**Dependencies.** Ultra-KMeans uses four header-only libraries. They are not
copied into this repository: the CMake build takes them from git submodules
under `third_party/`, and the Bazel build downloads the same revisions
(`MODULE.bazel`).

| Library | Used for | Source | Revision |
| --- | --- | --- | --- |
| ParlayLib | fork-join parallelism | `github.com/ParAlg/parlaylib` | `6b4a4cd` (tag `bazel`) |
| ParlayANN | Vamana graph index and beam search | `github.com/Akatsukis/ParlayANN` | `ca36f78` |
| CPAM / PAM | ordered maps inside the Ward's HAC seeder | `github.com/ParAlg/CPAM` | `049ad57` |
| Eigen | matrix products for exact assignment and SSE evaluation | `github.com/eigen-mirror/eigen` | `8ad4344` |

Get the sources together with the repository:

```sh
git clone --recurse-submodules https://github.com/ParAlg/Ultra-KMeans.git
# or, in an existing clone:
git submodule update --init
```

If `third_party/` is already populated, as in an archive of this repository
that includes the dependencies, nothing needs to be fetched.

**Bazel version.** The Bazel build requires Bazel 8.x and is pinned to
**8.5.1** in `.bazelversion`. Newer Bazel releases fail to build this
repository, so `MODULE.bazel` rejects any Bazel outside `>=8.0.0, <9.0.0`
with a clear message instead of a compilation error. The simplest way to get
the right version is [bazelisk](https://github.com/bazelbuild/bazelisk),
which reads `.bazelversion` and downloads Bazel 8.5.1 automatically;
alternatively install Bazel 8.5.1 directly. The Bazel build needs network
access: it downloads the four dependencies and Bazel's own rules. If Bazel is
not an option, use CMake.

**Compiler: GCC 15 for both builds.** GCC 15 is the default compiler. CMake
and Bazel choose their C++
compiler independently (CMake from `CXX` or the `c++` on `PATH`, Bazel from
`CC` or the first `gcc` on `PATH`), so on a machine with several compilers
they could end up with different versions. CMake here defaults to GCC 15
(the `c++` on `PATH` if that is GCC 15, otherwise `g++-15`); pass the same
compiler to Bazel as shown below. GCC 15 is packaged as `g++-15` in the
Ubuntu toolchain PPA (`ppa:ubuntu-toolchain-r/test`) and as `gcc-toolset-15`
on RHEL-family systems.

On Ubuntu the prerequisites can be installed with

```sh
sudo add-apt-repository ppa:ubuntu-toolchain-r/test      # provides GCC 15 on Ubuntu 24.04
sudo apt install g++-15 cmake git numactl
```

## Build

The CMake and Bazel builds are equivalent: both compile the single translation
unit `lloyds/kmeans.cpp` against the same dependency revisions, as C++20,
with the same optimization flags (`-O3 -march=native -mcx16 -fno-plt
-DPARLAY_USE_STD_ALLOC`, assertions enabled), and both declare project version
0.1.0. Compilation takes two to three minutes.

CMake (picks GCC 15 by default); the binary is `build/ultra_kmeans`:

```sh
git submodule update --init          # once, unless third_party/ is already populated
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Bazel 8.5.1 with the same compiler; the binary is
`bazel-bin/lloyds/ultra_kmeans`:

```sh
bazel build --repo_env=CC="$(command -v g++-15)" //lloyds:ultra_kmeans
```

To use another compiler, pass `-DCMAKE_CXX_COMPILER=/path/to/g++` to CMake and
`--repo_env=CC=/path/to/g++` to Bazel.

`-march=native` tunes the binary for the build machine; without it the program
is about 2x slower. To build a portable binary with CMake, pass
`-DULTRA_KMEANS_NATIVE=OFF`.

## Quick start

```sh
PARLAY_NUM_THREADS=16 \
ULTRA_KMEANS_TSV_PATH=results.tsv \
./build/ultra_kmeans \
  -i data/example.fbin \
  -k 1000 \
  -seed PrefixDoubling \
  -dist ANNS \
  -iters 25
```

One untimed warm-up run precedes the measured runs. By default, three measured
runs are averaged. Result files are appended rather than overwritten.

## Input formats

The input extension selects the reader. `-data_type` must match the stored
element type.

| Extension | `-data_type` | Layout |
| --- | --- | --- |
| `.fbin` | `float` | `uint32 n`, `uint32 d`, then `n*d` row-major float32 values |
| `.u8bin` | `uint8` | `uint32 n`, `uint32 d`, then `n*d` row-major uint8 values |
| `.i8bin` | `int8` | `uint32 n`, `uint32 d`, then `n*d` row-major int8 values |
| `.pbbs` | `float` | PBBS text format beginning with `pbbs_sequencePoint<D>d` |

The binary formats use little-endian headers and data. Non-finite floating
point input is rejected. Distances are Euclidean.

## Command-line options

Run `./build/ultra_kmeans -h` for the short usage string.

### Core options

| Option | Default | Meaning |
| --- | --- | --- |
| `-i PATH` | required | Input vector file |
| `-k N` | `10` | Number of clusters; must be in `[1,n]` |
| `-data_type TYPE` | `float` | `float`, `uint8`, or `int8` |
| `-seed NAME` | `PrefixDoubling` | Seeding algorithm (see Features) |
| `-dist NAME` | `Pairwise` | Assignment oracle: `ANNS` or `Pairwise` |
| `-iters N` | `25` | Lloyd iterations per run |
| `-rounds N` | `3` | Number of measured runs after one warm-up |
| `-no-subsample` | off | Always train on the full input |

By default, when `n > 256*k`, seeding and refinement run on a uniform sample
of `256*k` points (the default training cap of faiss). The SSE is always
evaluated on the full input.

### Seeding options

| Option | Default | Meaning |
| --- | --- | --- |
| `-pd-alpha X` | `1.05` | Doubling k-means: geometric batch-growth factor |
| `-samw X` | `5` | Ward's HAC: number of sampled representatives, as a multiple of `k` |
| `-epsw X` | `0.8` | Ward's HAC: merge approximation parameter; `0` selects exact HAC |
| `-deltaw X` | `1.0` | Ward's HAC: cluster-weight grouping parameter |
| `-Rw N` | `16` | Ward's HAC: Vamana graph degree |
| `-Lw N` | `12` | Ward's HAC: Vamana beam width |
| `-aw X` | `1.2` | Ward's HAC: Vamana pruning factor |
| `-npw N` | `1` | Ward's HAC: Vamana construction passes |

### ANNS refinement options

| Option | Default | Meaning |
| --- | --- | --- |
| `-R N` | `16` | Graph degree |
| `-L N` | `32` | Beam width |
| `-a X` | `1.5` | Pruning factor |
| `-np N` | `1` | Graph-construction passes |

For very small datasets, choose `R <= k-1`.

## Output

Aggregate results are appended to `kmeans.tsv`, or to the path in
`ULTRA_KMEANS_TSV_PATH`. Each record has 11 tab-separated fields:

1. input path
2. seeding algorithm
3. assignment oracle
4. number of points (`n`)
5. dimensions (`d`)
6. clusters (`k`)
7. worker threads
8. mean seeding time in seconds
9. SSE after seeding, on the full input
10. mean refinement time in seconds (initial assignment excluded)
11. final SSE, on the full input

The total running time is field 8 plus field 10. SSE evaluation and input I/O
are not part of either time. When a run exceeds
the time limit, fields 8, 9 and 11 are `na` and field 10 is `timeout`.

Setting `ULTRA_KMEANS_PER_ROUND_TSV_PATH` enables a second file with one record
for the state after seeding (`iteration=0`) and after each Lloyd iteration.
Its 10 fields are: input path, seeding algorithm, oracle, `n`, `d`, `k`,
threads, iteration, cumulative seeding + refinement time, and full-input SSE.
When this mode is enabled and `-rounds` is omitted, one measured run is made.

## Environment variables

| Variable | Effect |
| --- | --- |
| `PARLAY_NUM_THREADS` | Number of worker threads (default: all hardware threads) |
| `ULTRA_KMEANS_TSV_PATH` | Aggregate TSV path |
| `ULTRA_KMEANS_PER_ROUND_TSV_PATH` | Enables per-iteration TSV output at this path |
| `ULTRA_KMEANS_TIMEOUT_SECONDS` | Per-run limit on seeding + refinement time, checked after seeding and after each iteration |

## Repository layout

```text
lloyds/                 Lloyd's refinement (ANNS and exact) and the executable
seeding/                Seeding algorithms
seeding/wards/          Ward's HAC and its dynamic Vamana index
third_party/            Dependencies as git submodules (ParlayLib, ParlayANN, CPAM, Eigen)
CMakeLists.txt          CMake build
MODULE.bazel, BUILD     Bazel build (Bazel 8.x)
```
