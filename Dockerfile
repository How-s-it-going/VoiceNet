FROM nvidia/cuda:9.0-cudnn7-runtime

RUN apt-get update && apt-get install -y python3-pip \
					 swig \
					 mecab \
					 libmecab-dev \
					 mecab-ipadic-utf8 \
					 git \
					 curl \
					 sudo \
					 make \
					 libsndfile1-dev \
					 zip \
					 unzip

RUN pip3 install -U pip
RUN pip3 install tensorflow-gpu \
                 librosa \
                 matplotlib \
                 numpy \
                 scikit-learn \
                 scipy \
                 tqdm \
                 mecab-python3

RUN groupadd -g 1000 developer && \
    useradd -g developer -G sudo -m -s /bin/bash kodamanbou && \
    echo 'kodamanbou:password' | chpasswd

RUN echo 'Defaults visiblepw' >> /etc/sudoers
RUN echo 'kodamanbou ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER kodamanbou
WORKDIR /home/kodamanbou/

RUN git clone https://github.com/neologd/mecab-ipadic-neologd.git
RUN sudo ./mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -y
RUN sudo sed -i 's!/var/lib/mecab/dic/debian!/usr/lib/mecab/dic/mecab-ipadic-neologd!g' /etc/mecabrc
RUN git clone https://github.com/kodamanbou/VoiceNet.git

WORKDIR /home/kodamanbou/VoiceNet/
COPY dataset.zip .
RUN sudo unzip dataset.zip
RUN sudo rm dataset.zip
