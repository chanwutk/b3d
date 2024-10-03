# FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y build-essential git ffmpeg libsm6 libxext6 fonts-freefont-ttf
# RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'
# RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
# RUN pip install jupyterlab opencv-python scikit-image filterpy

COPY ./requirements.txt .
RUN pip install -r requirements.txt

# RUN cd /workspace && git clone https://github.com/facebookresearch/detectron2.git
RUN mkdir /root/.jupyter
RUN echo "get_config().ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py
