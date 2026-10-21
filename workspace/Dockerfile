# Espacio de trabajo de un proyecto de Veyra: SSH (para VS Code Remote-SSH), git y el mismo entorno de Python/GPU del worker.
ARG BASE=docker.labs.hapticus.cl/veyra-worker:0.2.0
FROM ${BASE}
USER root
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server git sudo iproute2 less nano vim-tiny curl wget \
        build-essential pkg-config procps unzip rsync tmux htop \
    && rm -rf /var/lib/apt/lists/* && mkdir -p /run/sshd
# el usuario veyra (1000:3000, el mismo del NFS) vive en la carpeta de trabajo, que es un volumen persistente
RUN usermod -d /workspace/home -s /bin/bash -p '*' veyra \
    && echo 'veyra ALL=(root) NOPASSWD: /usr/bin/apt-get, /usr/bin/apt' > /etc/sudoers.d/veyra && chmod 440 /etc/sudoers.d/veyra \
    && for b in python python3 pip pip3 ipython jupyter; do [ -x /opt/conda/bin/$b ] && ln -sf /opt/conda/bin/$b /usr/local/bin/$b || true; done
COPY sshd_config /etc/ssh/sshd_config_veyra
COPY bin/ /usr/local/bin/
COPY entrypoint.sh /entrypoint.sh
COPY bashrc /etc/veyra-bashrc
RUN chmod +x /entrypoint.sh /usr/local/bin/veyra-*
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
EXPOSE 22
WORKDIR /workspace
ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
