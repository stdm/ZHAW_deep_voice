FROM tensorflow/tensorflow:1.3.0-py3
RUN apt-get update && apt-get install -y git
RUN pip install numpy==1.10.4
RUN pip install scipy==0.16.1
RUN pip install Theano==0.8.2
RUN pip install https://github.com/Lasagne/Lasagne/archive/master.zip
RUN pip install joblib==0.9.3
RUN pip install scikit-learn==0.17
RUN pip install tabulate==0.7.5
RUN pip install git+https://github.com/dnouri/nolearn.git@master#egg=nolearn==0.7.git