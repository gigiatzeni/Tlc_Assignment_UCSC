FROM ubuntu:latest

LABEL maintainer="Vincenzo Nardelli <vincenzo.nardelli01@icatt.it>"
USER root

WORKDIR /home/
COPY ./app/ app/

# Python Setup
RUN apt-get update && \
    apt-get install python3 python3-pip -y

RUN pip3 install -r app/pip_requirements.txt

EXPOSE 5000
WORKDIR /home/app/
CMD ["python3", "app.py"]