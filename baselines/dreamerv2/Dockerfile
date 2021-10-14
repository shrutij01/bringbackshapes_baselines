FROM tensorflow/tensorflow:2.6.0-gpu

RUN apt-get update && apt-get install -y \
  python3-pip \
  && apt-get clean

RUN pip3 install --no-cache-dir \
  dreamerv2>=2.0.7 \
  gym

RUN pip install -e .

COPY . /app
WORKDIR /app
CMD ["python3", "main.py", "--steps", "5e5"]