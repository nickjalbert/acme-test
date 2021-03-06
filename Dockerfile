# syntax=docker/dockerfile:1
FROM ubuntu:latest
FROM python:3.8.10
ARG CONTAINER_HOME
ARG HOST_HOME
ENV HOME=$CONTAINER_HOME
RUN mkdir -p $CONTAINER_HOME/acme-test
RUN apt-get update
RUN apt-get -y install git curl vim less
WORKDIR $CONTAINER_HOME

# House keeping
COPY ./usr-config/.bashrc $CONTAINER_HOME/.bashrc
COPY ./usr-config/.bash_profile $CONTAINER_HOME/.bash_profile
RUN bash $CONTAINER_HOME/.bashrc
RUN mkdir -p /usr/local/bin
RUN git clone https://github.com/so-fancy/diff-so-fancy.git
RUN ln -s $CONTAINER_HOME/diff-so-fancy/diff-so-fancy /usr/local/bin/diff-so-fancy

# Manually install reqs
RUN pip install --upgrade pip setuptools
RUN pip install imageio
RUN pip install dm-acme
RUN pip install tensorflow
RUN pip install tensorflow-probability
RUN pip install dm-acme[reverb]
RUN pip install dm-acme[envs]
RUN pip install dm-reverb[tensorflow]
RUN pip install dm-sonnet
RUN pip install jax
RUN pip install jaxlib
RUN pip install trfl
RUN pip install imageio
RUN pip install PILLOW
RUN pip install pyvirtualdisplay
RUN pip install launchpad
RUN pip install flake8
RUN pip install black
