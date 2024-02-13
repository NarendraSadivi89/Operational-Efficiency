from ChatPage import ChatPage


def main():
    ChatPage(
        page_title="CMDB Chat",
        page_icon="🤖",
        options=[
            # 'cmdb_baseline',
            'cmdb_ci',
            'cmdb_rel_ci'
        ],
        header="CMDB Chat 🤖",
        glider_based=True
    )


if __name__ == '__main__':
    main()
