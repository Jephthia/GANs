FROM tensorflow/tensorflow:2.4.1-gpu

COPY . /app

WORKDIR /app

RUN pip install runipy

ENTRYPOINT ["runipy"]
