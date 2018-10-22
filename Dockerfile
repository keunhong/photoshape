FROM nvidia/cudagl:9.0-devel-ubuntu16.04
MAINTAINER Keunhong Park (kpar@cs.washington.edu)

RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common python-software-properties
RUN add-apt-repository ppa:deadsnakes/ppa

ENV DEBIAN_FRONTEND noninteractive
ENV DISPLAY :1

RUN apt-get update && apt-get install -y --no-install-recommends \
         sudo \
         libsm6 \
         libxext6 \
         libglfw3 \
         fontconfig \
         xserver-xorg-video-dummy\
         x11-xserver-utils \
         xinit \
         libgl1-mesa-dri \
         python3.6 \
         python3.6-dev \
         libpython3.6-dev \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libbz2-dev \
         libreadline-dev \
         libssl-dev \
         libjpeg-dev \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*

ADD . /app

ENV DIR=/app
ENV PYTHONPATH=$DIR/src:$PYTHONPATH
ENV PYTHONPATH=$DIR/thirdparty/pyhog:$PYTHONPATH
ENV PYTHONPATH=$DIR/thirdparty/rendkit:$PYTHONPATH
ENV PYTHONPATH=$DIR/thirdparty/kitnn:$PYTHONPATH
ENV PYTHONPATH=$DIR/thirdparty/toolbox:$PYTHONPATH
ENV PYTHONPATH=$DIR/thirdparty/vispy:$PYTHONPATH
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ADD ./get-pip.py /app/get-pip.py
RUN python3.6 /app/get-pip.py
RUN pip3.6 install setuptools
RUN pip3.6 install -r /app/requirements.txt
RUN pip3.6 install numpy
RUN pip3.6 install requests
RUN pip3.6 install git+https://github.com/mcfletch/pyopengl.git
RUN pip3.6 install git+https://github.com/keunhong/vispy.git
RUN pip3.6 install git+https://github.com/scikit-image/scikit-image.git
RUN pip3.6 install git+https://github.com/lucasb-eyer/pydensecrf.git

RUN git clone git://git.blender.org/blender.git /app/blender
WORKDIR /app/blender
RUN git submodule update --init --recursive
RUN git submodule foreach --recursive git checkout master
RUN git submodule foreach --recursive git pull --rebase origin master
RUN ./build_files/build_environment/install_deps.sh --source ./ --threads=4 --with-all --skip-osd --skip-ffmpeg
RUN mkdir build

WORKDIR /app/blender/build
RUN cmake \
  -DPYTHON_VERSION=3.6.5 \
  -DPYTHON_ROOT_DIR=$(pyenv prefix) \
  -DCMAKE_INSTALL_PREFIX=$(pyenv prefix)/lib/python3.6/site-packages \
  -DWITH_CYCLES=ON \
  -DWITH_GAMENGINE=OFF \
  -DWITH_IMAGE_HDR=ON \
  -DWITH_IMAGE_OPENEXR=ON \
  -DWITH_IMAGE_TIFF=ON \
  -DWITH_OPENCOLLADA=OFF \
  -DWITH_OPENMP=ON \
  -DWITH_OPENIMAGEIO=ON \
  -DWITH_PYTHON_INSTALL=OFF \
  -DWITH_PYTHON_MODULE=ON \
  -DPYTHON_LIBRARY=$(pyenv virtualenv-prefix)/lib/libpython3.6m.so \
  -DPYTHON_LIBPATH=$(pyenv virtualenv-prefix)/lib \
  -DPYTHON_INCLUDE_DIR=$(pyenv virtualenv-prefix)/include/python3.6m \
  ..
RUN make -j4
RUN make install

ENTRYPOINT /bin/bash
