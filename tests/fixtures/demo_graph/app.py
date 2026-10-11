"""Tiny notes API. Notes are persisted in MongoDB (set MONGODB_URI to override the default)."""
import json
import os
from datetime import date
from http.server import BaseHTTPRequestHandler, HTTPServer

from openai import OpenAI
from pymongo import MongoClient, ReturnDocument

from actions import extract_action_items

client = MongoClient(os.environ.get("MONGODB_URI", "mongodb://localhost:27017"))
db = client[os.environ.get("MONGODB_DB", "demo_app")]
notes = db["notes"]
counters = db["counters"]

# Summaries run on Groq (OpenAI-compatible API) with a large open-weight model by default.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
llm_client = None  # created on first /summarize so the app starts without GROQ_API_KEY


def summarize_notes(texts: list[str]) -> str:
    global llm_client
    if llm_client is None:
        llm_client = OpenAI(base_url=GROQ_BASE_URL, api_key=os.environ["GROQ_API_KEY"])
    response = llm_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "Summarize the user's notes in a few concise sentences."},
            {"role": "user", "content": "\n".join(f"- {t}" for t in texts)},
        ],
    )
    return response.choices[0].message.content


def next_note_id() -> int:
    counter = counters.find_one_and_update(
        {"_id": "notes"}, {"$inc": {"seq": 1}}, upsert=True, return_document=ReturnDocument.AFTER
    )
    return counter["seq"]


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/notes":
            self._send(200, list(notes.find({}, {"_id": 0}).sort("id", 1)))
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/action-items":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or "{}")
            try:
                meeting = date.fromisoformat(body.get("meeting_date") or date.today().isoformat())
            except ValueError:
                return self._send(400, {"error": "meeting_date must be YYYY-MM-DD"})
            try:
                items = extract_action_items(body.get("notes", ""), meeting)
            except Exception as e:
                return self._send(502, {"error": f"extraction failed: {e}"})
            return self._send(200, {"action_items": items})
        if self.path == "/summarize":
            texts = [n["text"] for n in notes.find({}, {"_id": 0, "text": 1}).sort("id", 1)]
            if not texts:
                return self._send(200, {"summary": ""})
            try:
                return self._send(200, {"summary": summarize_notes(texts)})
            except Exception as e:
                return self._send(502, {"error": f"summarization failed: {e}"})
        if self.path != "/notes":
            return self._send(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length", 0))
        note = {"id": next_note_id(), "text": json.loads(self.rfile.read(length) or "{}").get("text", "")}
        notes.insert_one(dict(note))  # copy: insert_one adds an ObjectId _id in place
        self._send(201, note)


if __name__ == "__main__":
    print("Serving on http://localhost:8000")
    HTTPServer(("localhost", 8000), Handler).serve_forever()
