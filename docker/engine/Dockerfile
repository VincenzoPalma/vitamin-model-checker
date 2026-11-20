# VITAMIN engine with Spot from conda-forge, for linux/amd64 and linux/arm64.
# spottl (the pip Spot) only ships x86_64 binaries, so Spot comes from conda-forge
# on both architectures and VITAMIN is installed without it.

FROM mambaorg/micromamba:1.5.8

ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN micromamba install -y -n base -c conda-forge --override-channels \
        python=3.11 spot=2.16.3 pip \
    && micromamba clean -a -y

WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml README.md LICENSE ./
COPY --chown=$MAMBA_USER:$MAMBA_USER model_checker/ ./model_checker/

# Every runtime dependency from pyproject.toml except spottl, then VITAMIN itself.
RUN python -c "import tomllib; print('\n'.join(d for d in tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies'] if not d.startswith('spottl')))" > /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir --no-deps .

COPY --chown=$MAMBA_USER:$MAMBA_USER examples/ ./examples/
COPY --chown=$MAMBA_USER:$MAMBA_USER docker/engine/smoke_test.py ./smoke_test.py

CMD ["python", "smoke_test.py"]
