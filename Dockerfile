FROM python:3.9-slim

WORKDIR /app/HkAIHub

COPY . ./

RUN pip install -r requirements.txt

RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx
RUN apt-get -y install libglib2.0-0

CMD ["python", "main_AI_
hub.py"]
