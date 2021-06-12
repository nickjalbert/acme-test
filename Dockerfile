# syntax=docker/dockerfile:1
FROM ubuntu:latest
FROM python:3.8.10
RUN apt-get update
WORKDIR /code
# COPY requirements.txt /code/
# RUN pip install -r requirements.txt
COPY .git/ /code/
COPY .gitignore /code/
COPY LICENSE /code/
COPY README.md /code/
COPY cartpole.py /code/
COPY sonnet_test.py /code/
COPY verify.py /code/
