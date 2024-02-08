# Knowledge Management AI Chat

State purpose here

### Requirements

```commandline
pip install -r requirements.txt
```
### Run
```commandline
streamlit run Snow_Chat.py
```
### Docker
Once again, you will need to provision your api keys in a .env file at the root of the project directory.
```commandline
docker build -t streamlit .
```
```commandline
docker run -p 8501:8501 streamlit
```