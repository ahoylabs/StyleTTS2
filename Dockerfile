ARG UBUNTU_VERSION=22.04
# This needs to generally match the container host's environment.
# We use a slightly older version for greater compatibility
ARG CUDA_VERSION=12.3.2
# CUDA base image (excludes cublas)
ARG CUDA_BASE_CONTAINER=nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION}

# this is the ffmpeg build stage
# we build a cut down static ffmpeg, which saves about 500MB vs system install
# note only very limited functionality is available
FROM debian:bookworm-slim AS ffmpeg

RUN apt-get update && apt-get install -y \
    build-essential yasm libogg-dev libopus-dev libsndfile1-dev libmp3lame-dev wget

WORKDIR /ffmpeg
RUN wget https://ffmpeg.org/releases/ffmpeg-4.4.2.tar.gz && \
    tar -xzf ffmpeg-4.4.2.tar.gz && \
    cd ffmpeg-4.4.2 && \
    ./configure \
        --prefix=/usr/local \
        --extra-cflags="-I/usr/include" \
        --extra-ldflags="-L/usr/lib" \
        --enable-static \
        --disable-shared \
        --disable-debug \
        --disable-doc \
        --disable-ffplay \
        --disable-ffprobe \
        --enable-libmp3lame \
        --enable-libopus && \
    make -j $(nproc) && \
    make install && \
    make clean

# this is the final stage
FROM ${CUDA_BASE_CONTAINER} AS runtime
# copy early so that any code changes do not invalidate the cache
COPY --from=ffmpeg /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg

ENV DEBIAN_FRONTEND=noninteractive

RUN /bin/echo -e '#!/bin/bash\nDEBIAN_FRONTEND=noninteractive\napt-get update && apt-get install -y $@ && apt-get clean autoclean && apt-get autoremove --yes && rm -rf /var/lib/apt/lists/*' \
    > /usr/local/sbin/apt_install_clean.sh && \
    chmod a+x /usr/local/sbin/apt_install_clean.sh && \
    /bin/echo -e '#!/bin/bash\nDEBIAN_FRONTEND=noninteractive\napt-get update && apt-get remove -y $@ && apt-get clean autoclean && apt-get autoremove --yes && rm -rf /var/lib/apt/lists/*' \
    > /usr/local/sbin/apt_remove_clean.sh && \
    chmod a+x /usr/local/sbin/apt_remove_clean.sh

RUN /usr/local/sbin/apt_install_clean.sh iputils-ping net-tools curl wget nano git espeak-ng python3 python3-pkg-resources libopus0 libmp3lame0

COPY . /StyleTTS2
WORKDIR /StyleTTS2

RUN /usr/local/sbin/apt_install_clean.sh gcc python3-pip && \
    pip install -r freeze.txt && \
    rm -rf /root/.cache && \
    /usr/local/sbin/apt_remove_clean.sh gcc python3-pip

RUN mkdir -p additional_voices
CMD ["/usr/bin/python3", "api_v2.py"]
EXPOSE 5000
HEALTHCHECK CMD curl --fail http://localhost:5000 || exit 1
