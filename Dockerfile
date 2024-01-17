FROM python:3.10-slim-buster

RUN pip install explainerdashboard

COPY /app/dashboard.py ./
COPY /app/abalone_cl.csv ./
COPY /app/app.py ./

RUN python dashboard.py

EXPOSE 9050
CMD ["python", "./app.py"]