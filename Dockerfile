FROM tensorflow/tensorflow:1.13.1-gpu-py3
RUN pip install --upgrade pip 
RUN pip install numpy==1.14.5
RUN pip install scipy==0.19.1
RUN pip install theano==1.0.4
RUN pip install scikit-learn==0.19.1
RUN pip install matplotlib==2.1.0
RUN pip install keras==2.2.4
RUN pip install librosa==0.5.1
RUN pip install --upgrade https://github.com/Lasagne/Lasagne/archive/a61b76fd991f84c50acdb7bea02118899b5fefe1.zip
RUN pip install https://github.com/dnouri/nolearn/archive/2ef346c869e80fc90247916e4aea5cfa7cf2edda.zip#egg=nolearn
RUN pip install pandas==0.20.3
RUN pip install munkres==1.0.7
RUN pip install pyannote.metrics==2.1
RUN pip install xarray==0.12.3
