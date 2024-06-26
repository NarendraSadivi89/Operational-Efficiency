FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

COPY . .

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

RUN flask --app api/api.py run

ENTRYPOINT ["streamlit", "run", "ChatPage.py", "--server.port=8501", "--server.address=0.0.0.0"]