from ChatPage import ChatPage


def main():
    ChatPage(
        page_title="Snow Chat",
        page_icon="❄",
        options=[
            'incident',
            'change_request',
            'change_task'
        ],
        header="Snow Chat ❄",
        glider_based=True
    )


if __name__ == '__main__':
    main()
