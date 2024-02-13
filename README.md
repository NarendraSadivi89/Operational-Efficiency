# Knowledge Management AI Chat

AI leveraging LLMs to chat with Confluence, ServiceNow, and CMDB data.

![TechStackDiagram.png](..%2F..%2FDownloads%2FTechStackDiagram.png)

### Requirements

```commandline
pip install -r requirements.txt
```
### Run
```commandline
streamlit run SnowChat.py
```
### Docker
Once again, you will need to provision your api keys in a .env file at the root of the project directory.
```commandline
docker build -t streamlit .
```
```commandline
docker run -p 8501:8501 streamlit
```