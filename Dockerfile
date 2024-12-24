FROM ubuntu:20.04

WORKDIR /app

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        gcc-9 \
        g++-9 \
        gcc-9-multilib \
        g++-9-multilib \
        xutils-dev \
        patch \
        git \
        python3.9 \
        python3.9-dev \
        python3-pip \
        libpulse-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 20 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 20 \
    && python3 -m pip install --upgrade pip

COPY requirements.txt .
RUN pip3 install --default-timeout=500 torch==2.3.0+cu121 -f https://download.pytorch.org/whl/cu121/torch_stable.html
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]