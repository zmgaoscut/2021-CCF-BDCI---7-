FROM paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7

RUN cd work/PaddleVideo

RUN pip install -r requirements.txt


