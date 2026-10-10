# demo-app

A tiny in-memory notes API used to try out NotByDefault's hooks.

    py -3.12 app.py
    curl -X POST localhost:8000/notes -d '{"text": "hi"}'
    curl localhost:8000/notes
