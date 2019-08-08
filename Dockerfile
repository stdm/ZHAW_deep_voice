FROM tensorflow/tensorflow:1.13.1-gpu-py3
RUN pip install --upgrade pip
RUN pip install numpy==1.17.0
RUN pip install scipy==0.19.1
RUN pip install scikit-learn==0.19.1
RUN pip install matplotlib==3.0.3
RUN pip install keras==2.2.4
RUN pip install librosa==0.5.1
RUN pip install pandas==0.20.3
RUN pip install munkres==1.0.7
RUN pip install pyannote.metrics==2.1
RUN pip install SIDEKIT==1.3.2
RUN pip install mxnet==1.5.0
