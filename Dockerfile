FROM tensorflow/tensorflow:latest-gpu-py3

COPY ./Quarto quarto/Quarto
ADD QuartoTrain.py ~/quarto/

WORKDIR quarto
RUN apt-get update && apt-get install -y python3-tk
RUN pip install keras pydot graphviz

CMD ["python3", "QuartoTrain.py"]
