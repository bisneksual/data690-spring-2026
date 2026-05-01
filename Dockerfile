FROM python:3.12-slim

# Install pip requirements
COPY requirements.txt .

RUN pip install -r requirements.txt

# Copy source files
COPY . .

#RUN apt-get update -y  && apt-get install feh nano -y && mkdir explore

EXPOSE 3000