import os
from langchain_community.document_loaders import ConfluenceLoader


def provision_confluence():
    loader = ConfluenceLoader(
        url="https://cgi-prasannapuli.atlassian.net/wiki", username="Michael.Culpepper", api_key=os.getenv('conf_api')
    )
    documents = loader.load(space_key="SPACE", include_attachments=True, limit=50)

    print(documents)


provision_confluence()
