from flask import Flask, render_template, request, jsonify
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
import os
import base64

from casper_prompt import casper_prompt
from memory import save_memory, get_memory

load_dotenv()

AI_PROVIDER = os.getenv("AI_PROVIDER", "openai").strip().lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")

openai_client = None
gemini_client = None

def fallback_reply(user_message: str) -> str:
    return (
        "Casper is still online and ready to chat. "
        "Even if the AI service hiccuped, I’ll keep the convo going with energy and attitude."
    )

if AI_PROVIDER == "openai":
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI provider.")
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
elif AI_PROVIDER == "gemini":
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is required for Gemini provider.")
    try:
        import google.generativeai as generativeai
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Gemini support requires the google-generativeai package. Install it with `pip install google-generativeai`."
        ) from exc

    generativeai.configure(api_key=GEMINI_API_KEY)
    gemini_client = generativeai
else:
    raise RuntimeError("Unsupported AI_PROVIDER value. Use 'openai' or 'gemini'.")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():

    data = request.json

    user_message = data["message"]
    username = "Karl"

    memories = get_memory(username)

    memory_context = "\n".join(memories)

    save_memory(username, user_message)

    try:
        if AI_PROVIDER == "openai":
            response = openai_client.responses.create(
                model=OPENAI_MODEL,
                input=[
                    {
                        "role": "system",
                        "content": f"{casper_prompt}\n\nMemories:\n{memory_context}"
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ]
            )
            reply = response.output_text
        else:
            model = gemini_client.GenerativeModel(
                GEMINI_MODEL,
                system_instruction=f"{casper_prompt}\n\nMemories:\n{memory_context}"
            )
            chat_session = model.start_chat()
            response = chat_session.send_message(user_message)
            reply = getattr(response, "text", None)
            if reply is None:
                reply = str(response)

        audio_base64 = None
        if not reply:
            reply = fallback_reply(user_message)

        audio_base64 = None
        if AI_PROVIDER == "openai" and request.json.get("voice"):
            tts_response = openai_client.audio.speech.create(
                model=OPENAI_TTS_MODEL,
                voice="alloy",
                input=reply,
                response_format="mp3"
            )
            audio_base64 = base64.b64encode(tts_response.content).decode("ascii")

        return jsonify({
            "reply": reply,
            "audio": audio_base64
        })
    except OpenAIError as e:
        app.logger.error("OpenAI error: %s", e)
        return jsonify({
            "reply": fallback_reply(user_message),
            "audio": None
        }), 200
    except Exception as e:
        app.logger.error("Chat error: %s", e)
        return jsonify({
            "reply": fallback_reply(user_message),
            "audio": None
        }), 200

if __name__ == "__main__":
    app.run(debug=True)