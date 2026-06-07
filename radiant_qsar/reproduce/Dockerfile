# Dockerfile for the RADIANT-QSAR pipeline.
#
# This image reproduces the entire pipeline from `chembl_36.db` to
# Phase G figures in a single command (`run_all.sh`). The image is
# CUDA-12.x-ready; on a CPU-only host it still runs Phase A and the
# CPU baselines.
#
# Build:
#   docker build -t radiant-qsar:0.1.0 \
#       -f radiant_qsar/reproduce/Dockerfile .
#
# Run end-to-end (mount the ChEMBL DB read-only, persist outputs):
#   docker run --gpus all --rm \
#       -v "$(pwd)/chembl_36.db:/work/chembl_36.db:ro" \
#       -v "$(pwd)/data:/work/data" \
#       -v "$(pwd)/runs:/work/runs" \
#       -v "$(pwd)/checkpoints:/work/checkpoints" \
#       radiant-qsar:0.1.0 \
#       bash radiant_qsar/reproduce/run_all.sh
#
# Drop --gpus all on a CPU-only host; the script auto-detects CUDA and
# skips the GPU-bound phases.

ARG CUDA_VERSION=12.6.3
ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:/opt/conda/condabin:$PATH

# System deps -- git for repo hashing, sqlite3 for sanity-pinging the
# ChEMBL DB, build-essential for any source builds pip falls back to.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl bzip2 git wget sqlite3 build-essential \
    && rm -rf /var/lib/apt/lists/*

# Miniforge -- same distribution as the reference WSL env so package
# versions resolve identically.
ARG MINIFORGE_VERSION=24.11.3-2
RUN curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniforge.sh \
    && conda clean -afy

# Build the qsar env. Pins mirror radiant_qsar/reproduce/manifest.yaml;
# update both files together when bumping a dep.
RUN conda create -n qsar -c conda-forge -y \
        python=3.11 \
        "pytorch=2.7.*" "pytorch-cuda=12.6" \
        "numpy=2.4.*" \
        "pandas=3.0.*" \
        "scipy=1.17.*" \
        "scikit-learn=1.8.*" \
        "matplotlib=3.10.*" \
        "seaborn=0.13.*" \
        "pyarrow=23.0.*" \
        "pyyaml=6.0.*" \
        "rdkit=2026.03.*" \
        "transformers=5.8.*" \
        "tokenizers=0.22.*" \
        "huggingface_hub=1.14.*" \
        "joblib=1.5.*" \
        "pytest=9.*" "pytest-xdist" \
    && conda clean -afy

# pip-only deps that aren't on conda-forge or that we want pinned by pip.
RUN /opt/conda/envs/qsar/bin/pip install --no-cache-dir \
        "chembl_structure_pipeline>=1.2.4" \
        "selfies>=2.1"

# Default to the qsar env in every RUN, ENTRYPOINT, CMD.
SHELL ["conda", "run", "--no-capture-output", "-n", "qsar", "/bin/bash", "-lc"]
ENV PATH=/opt/conda/envs/qsar/bin:$PATH

WORKDIR /work

# Copy source. Layer the copies so deps-only changes don't bust the
# heavy image-build cache.
COPY pyproject.toml ./pyproject.toml
COPY radiant ./radiant
COPY radiant_chem ./radiant_chem
COPY radiant_qsar ./radiant_qsar
COPY training ./training
COPY configs ./configs
COPY tests ./tests
COPY examples ./examples
COPY docs ./docs
COPY README.md LICENSE ./

# Editable install so on-disk hot-edits during a container session take
# effect without rebuilding. The `[all]` extra is a no-op for already-
# satisfied conda deps; it backstops anything we forgot to pin above.
RUN pip install --no-deps --no-cache-dir -e ".[all]"

# Sanity-check: every test that doesn't need a checkpoint must pass on
# image build. If this trips, the image is stale.
RUN PYTHONPATH=. pytest -q tests \
        --ignore=tests/qsar/test_pretrain_loop.py \
        --ignore=tests/qsar/test_finetune_single_task.py \
        -x

# Convenience: enter `bash` already inside the qsar env.
CMD ["bash", "-lc", "echo 'Image ready. Try:  bash radiant_qsar/reproduce/run_all.sh' && exec bash"]
