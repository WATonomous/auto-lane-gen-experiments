# # ================= Dependencies ===================
# RUN apt-get update && apt-get install -y curl && \
#     rm -rf /var/lib/apt/lists/*

# # Add a docker user so that created files in the docker container are owned by a non-root user
# RUN addgroup --gid 1000 docker && \
#     adduser --uid 1000 --ingroup docker --home /home/docker --shell /bin/bash --disabled-password --gecos "" docker && \
#     echo "docker ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/nopasswd

# # Remap the docker user and group to be the same uid and group as the host user.
# # Any created files by the docker container will be owned by the host user.
# RUN USER=docker && \
#     GROUP=docker && \
#     curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.4/fixuid-0.4-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
#     chown root:root /usr/local/bin/fixuid && \
#     chmod 4755 /usr/local/bin/fixuid && \
#     mkdir -p /etc/fixuid && \
#     printf "user: $USER\ngroup: $GROUP\npaths:\n  - /home/docker/" > /etc/fixuid/config.yml

# USER docker:docker

# ENV DEBIAN_FRONTEND noninteractive
# RUN sudo chsh -s /bin/bash
# ENV SHELL=/bin/bash

# ================= Repositories ===================
FROM jupyter/scipy-notebook

COPY src/atrous_attn_vit/randaug.py randaug.py
COPY src/atrous_attn_vit/vit-imagenet.py vit-imagenet.py
COPY src/atrous_attn_vit/test_script.sh test_script.sh

RUN python3 test_script.sh

