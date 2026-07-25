FROM tensorflow/tensorflow:2.13.0

WORKDIR /app

COPY pyproject.toml requirements.txt ./
COPY model.py data_handler.py utils.py main.py ./

RUN pip install --no-cache-dir -e "."

CMD ["python", "main.py"]