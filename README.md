# Knowledge Management AI Chat

AI leveraging LLMs to chat with Confluence, ServiceNow, and CMDB data.

![TechStackDiagram.png](assets%2FTechStackDiagram.png)

### Requirements

```commandline
pip install -r requirements.txt
```
```commandline
python -m spacy download en_core_web_sm
```
### Run
```commandline
streamlit run ChatPage.py
```
### Docker
Once again, you will need to provision your api keys in a .env file at the root of the project directory.
```commandline
docker build -t streamlit .
```
```commandline
docker run -p 8501:8501 streamlit
```