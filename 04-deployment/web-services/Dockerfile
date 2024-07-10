FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install -U pip & pip install pipenv

RUN pip install pandas

RUN pip install pyarrow

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "homework.py", "homework.py", "./" ]

EXPOSE 9696

ENTRYPOINT [ "python", "homework.py" ]