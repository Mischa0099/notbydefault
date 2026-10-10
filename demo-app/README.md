# demo-app

A tiny MongoDB-backed notes API used to try out NotByDefault's hooks.

Requires a running MongoDB (default `mongodb://localhost:27017`, override with
`MONGODB_URI`; database name via `MONGODB_DB`, default `demo_app`).

    py -3.12 -m pip install -r requirements.txt
    py -3.12 app.py
    curl -X POST localhost:8000/notes -d '{"text": "hi"}'
    curl localhost:8000/notes
    curl -X POST localhost:8000/summarize   # needs OPENAI_API_KEY; summarizes all notes with gpt-4o
