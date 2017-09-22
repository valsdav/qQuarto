FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update && apt-get install -y python3-tk
RUN pip install keras pydot graphviz

RUN mkdir /quarto
WORKDIR /quarto
ADD ./Quarto ./Quarto
ADD ./QuartoTrain.py .

CMD ["python3", "QuartoTrain.py"]
