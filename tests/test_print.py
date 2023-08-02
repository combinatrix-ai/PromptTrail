from src.prompttrail.core import Session, TextMessage

print(TextMessage(content="Hello", sender="User"))
print(TextMessage(content="He\nllo", sender="User"))

print(
    Session(
        messages=[
            TextMessage(content="Hello", sender="User"),
            TextMessage(content="He\nllo", sender="User"),
        ]
    )
)
