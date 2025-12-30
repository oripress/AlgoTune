# Use an official Python 3.11 base (compatible with many wheels)
FROM python:3.11-slim

# Set env vars
ENV PYTHONUNBUFFERED=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# Ubuntu tools (slim) + git for packaging + AWS CLI for S3 uploads
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
        awscli \
        libssl-dev \
        libblas-dev liblapack-dev \
        libffi-dev \
        libxml2-dev libxslt-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Workdir for project
WORKDIR /app

# Copy project files
COPY pyproject.toml /app/pyproject.toml
COPY algotune.py /app/algotune.py
COPY algotune.sh /app/algotune.sh
COPY config.env /opt/config.env

COPY AlgoTuner /app/AlgoTuner
COPY AlgoTuneTasks /app/AlgoTuneTasks
COPY scripts /app/scripts

# Create directories for logs and outputs FIRST
RUN mkdir -p /app/logs /app/aws/outputs /app/aws/errors /app/reports

# Copy generation config (contains n values for each task)
# Use wildcard to make optional - if file doesn't exist, pattern won't match
COPY reports/generation.jso[n] /app/reports/

# Make shell scripts executable
RUN chmod +x /app/algotune.sh /app/scripts/*.sh

# Install build tools
RUN python3 -m pip install --upgrade pip setuptools wheel

# **Core editable install** of your package via PEP 660
# Avoid pulling heavy optional deps during image build.
RUN python3 -m pip install --no-cache-dir --no-deps -e .

# Later you can install heavy deps explicitly if needed
# We *don’t install the entire huge dependency list blindly*
# because many heavy packages (torch, jax, faiss, cvxpy etc.)
# either require CUDA or system libs not present in slim images.

# But install the essentials analogous to your singularity:
RUN python3 -m pip install --no-cache-dir \
        numpy \
        scipy \
        pandas \
        pyyaml \
        ipython \
        joblib \
        matplotlib \
        seaborn \
        networkx \
        huggingface_hub

# Expose /app on PYTHONPATH so imports resolve
ENV PYTHONPATH=/app:$PYTHONPATH

# Default entrypoint
ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["echo 'AlgoTune container environment ready.'"]
