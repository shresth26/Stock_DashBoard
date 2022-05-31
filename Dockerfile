FROM python:3.8

COPY . /app
WORKDIR /app

RUN pip3 install -r requirements.txt
EXPOSE 8501

WORKDIR /src

ENTRYPOINT ["streamlit", "run"]
CMD ["stocks.py"]
