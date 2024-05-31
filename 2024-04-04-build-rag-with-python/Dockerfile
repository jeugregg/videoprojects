FROM debian:bookworm

RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    libsqlite3-dev \
    zlib1g-dev \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libffi-dev \
    liblzma-dev \
    tk-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    git \
    curl \
    && apt-get clean

RUN wget https://www.sqlite.org/2024/sqlite-autoconf-3460000.tar.gz && \
    tar xvfz sqlite-autoconf-3460000.tar.gz && \
    cd sqlite-autoconf-3460000 && \
    ./configure --prefix=/usr/local && \
    make && \
    make install && \
    cd .. && \
    rm -rf sqlite-autoconf-3460000 sqlite-autoconf-3460000.tar.gz

RUN wget https://www.python.org/ftp/python/3.12.2/Python-3.12.2.tgz && \
    tar xvf Python-3.12.2.tgz && \
    cd Python-3.12.2 && \
    ./configure --enable-optimizations --with-ensurepip=install --with-sqlite3=/usr/local && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.12.2 Python-3.12.2.tgz

RUN python3.12 -m pip install --upgrade pip && \
    python3.12 -m pip install chromadb

WORKDIR /app

COPY . /app

EXPOSE 8000

ENTRYPOINT ["bash"]
CMD ["-c", "chroma run --host localhost --port 8000 --path vectordb-stores/chromadb"]

