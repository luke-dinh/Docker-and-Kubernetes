FROM python:3.6
WORKDIR /home/lukedinh/Desktop/Docker-and-Kubernetes
COPY . .
RUN pip install pipenv
RUN pipenv install
EXPOSE 5000
CMD ["pipenv", "run". "python", "api.py"]