FROM tensorflow/tensorflow:2.15.0-gpu

RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl libgoogle-perftools4 \
    file \
    sox \
    libsox-dev \
    libsox-fmt-all screen tmux

RUN groupadd -g 100000 users_deezer && useradd -r -m -s /bin/bash -u 100000 deezer

# Changed permissions to make /home/deezer fully accessible
RUN chmod -R 777 /usr/local && chown -R 1000:1000 /usr/local && chmod -R 777 /home/deezer
# This line was added to ensure the deezer user owns its home directory
RUN chown -R deezer:users_deezer /home/deezer

# BOILERPLATE STUFF
USER deezer

WORKDIR /workspace

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 8000 5678

ENV TERM xterm-256color

CMD ["tmux", "new-session", "-d", "-s", "whisper", "bash"]