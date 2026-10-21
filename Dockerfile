# Veyra worker: PyTorch + Ultralytics + OpenCV con CUDA. Se ejecuta con acceso a la GPU (ver docker-compose.yml).
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    VEYRA_DATA_ROOT=/data VEYRA_PYTHON=/opt/conda/bin/python
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg tini \
    && rm -rf /var/lib/apt/lists/*
# Ultralytics 8.2 (mismo que usa Veyra hoy; no soporta YOLO11) y OpenCV sin interfaz gráfica
RUN pip install "ultralytics==8.2.0" opencv-python-headless "numpy<2"

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# no se ejecuta como root: el uid/gid coincide con el dueño de la carpeta compartida (1000:3000 en el NFS actual)
ARG UID=1000
ARG GID=3000
RUN groupadd -g ${GID} veyra && useradd -m -u ${UID} -g ${GID} veyra
USER veyra
ENTRYPOINT ["/usr/bin/tini", "--", "veyra-worker"]
CMD ["run"]
