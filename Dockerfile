FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

ENV CODEOCEAN_BASE_DIR=/

WORKDIR /

COPY requirements.txt /code/
RUN pip install --no-cache-dir -r /code/requirements.txt

RUN mkdir -p /results