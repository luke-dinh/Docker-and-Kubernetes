FROM python:3.9.2
WORKDIR /home/lukedinh/Desktop/Docker-and-Kubernetes
COPY . .
RUN pip install pipenv
RUN pipenv install
EXPOSE 5000
CMD ["pipenv", "run". "python", "api.py"]