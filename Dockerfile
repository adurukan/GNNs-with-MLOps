# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

COPY requirements-pip.txt requirements.txt

RUN pip install -r requirements.txt
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
RUN pip install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
RUN pip install torch-geometric

COPY . .

RUN echo "CREATING DATA"
RUN python create_data.py

RUN echo "TRAINING THE MODEL"
RUN python train.py

RUN echo "EVALUATE THE MODEL"
RUN python evaluate.py

CMD echo "LOGGER"