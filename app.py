import json
import re
import time

import streamlit as st
from google import genai
from google.genai import types


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Generate Lyrics",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ Gemini Audio Recognition")
st.caption("Upload an audio file and get a transcript, language detection, summary, keywords, and speaker information.")


# ============================================================
# API KEY
# ============================================================

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = None

if not api_key:
    st.error("GEMINI_API_KEY is missing from Streamlit secrets.")
    st.code(
        'GEMINI_API_KEY = "your-gemini-api-key"',
        language="toml",
    )
    st.stop()

client = genai.Client(api_key=api_key)


# ============================================================
# MODEL DISCOVERY
# ============================================================

FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
]


def discover_models():
    """Return Gemini models that appear usable for generate_content."""
    models = []

    try:
        for model in client.models.list():
            name = getattr(model, "name", "") or ""

            if name.startswith("models/"):
                name = name.split("/", 1)[1]

            if not name:
                continue

            # Skip obvious embedding / non-generative models.
            lower = name.lower()
            if "embedding" in lower:
                continue

            if name not in models:
                models.append(name)

    except Exception:
        pass

    return models


def model_candidates():
    discovered = discover_models()

    # Prefer Flash models for speed/cost.
    discovered_sorted = sorted(
        discovered,
        key=lambda x: (
            0 if "flash" in x.lower() else 1,
            x.lower(),
        ),
    )

    candidates = []

    for model in discovered_sorted + FALLBACK_MODELS:
        model = model.strip()

        if model and model not in candidates:
            candidates.append(model)

    return candidates


# ============================================================
# AUDIO MIME TYPES
# ============================================================

AUDIO_MIME_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "m4a": "audio/mp4",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "aac": "audio/aac",
    "webm": "audio/webm",
}


def get_mime_type(filename):
    extension = filename.lower().rsplit(".", 1)[-1]

    return AUDIO_MIME_TYPES.get(
        extension,
        "application/octet-stream",
    )


# ============================================================
# GEMINI AUDIO CALL
# ============================================================

def is_retryable_error(error):
    message = str(error).upper()

    retry_words = [
        "503",
        "UNAVAILABLE",
        "500",
        "INTERNAL",
        "DEADLINE_EXCEEDED",
        "TIMEOUT",
    ]

    return any(word in message for word in retry_words)


def is_quota_error(error):
    message = str(error).upper()

    quota_words = [
        "429",
        "RESOURCE_EXHAUSTED",
        "QUOTA",
        "RATE LIMIT",
    ]

    return any(word in message for word in quota_words)

#Create a Part object from these raw bytes.
#That Part object can then be passed to a model/API as an audio input.
#audio_part() takes raw audio + its format and packages them into the Part format expected by the API.
def audio_part(audio_bytes, mime_type):
    return types.Part.from_bytes(
        data=audio_bytes,
        mime_type=mime_type,
    )

#Send this audio and prompt to Gemini. 
#If Gemini successfully returns text, give me that text. 
#If something goes wrong, check whether it's a quota error or a non-retryable error; if so, stop immediately. 
#Otherwise, wait a little and try again, up to the specified number of retries.
def call_gemini_audio(
    audio_bytes,
    mime_type,
    prompt,
    model_name,
    retries=3,
):
    last_error = None

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    audio_part(audio_bytes, mime_type),
                    prompt,
                ],
            )

            text = getattr(response, "text", None)

            if not text:
                raise ValueError("Gemini returned an empty response.")

            return text

        except Exception as error:
            last_error = error

            # Do not waste retries on quota exhaustion.
            if is_quota_error(error):
                raise

            if not is_retryable_error(error):
                raise

            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    raise RuntimeError(str(last_error))


# ============================================================
# JSON PARSER
# ============================================================

def parse_json(text):
    text = text.strip()

    # Remove Markdown JSON fences if Gemini adds them.
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    data = json.loads(text)

    # Protect against Gemini returning a JSON list.
    if isinstance(data, list):
        if len(data) == 1 and isinstance(data[0], dict):
            data = data[0]
        else:
            raise ValueError(
                "Gemini returned a JSON list instead of the expected analysis object."
            )

    if not isinstance(data, dict):
        raise ValueError(
            "Gemini returned JSON, but the root value was not an object."
        )

    return data


# ============================================================
# AUDIO PROMPT
# ============================================================

AUDIO_PROMPT = """
Analyze the uploaded audio recording.

Your job is to perform speech/audio recognition.

Requirements:

1. Detect the primary spoken language.
2. Produce the complete spoken transcript as accurately as possible.
3. Preserve the speaker's actual words.
4. Do not invent missing words.
5. If a section is unclear, mark it as [inaudible] rather than guessing.
6. Give a concise summary.
7. Extract the most important key points.
8. Extract useful keywords.
9. Identify speakers when possible.
10. If there are multiple speakers, describe them as Speaker 1, Speaker 2, etc.
11. Do not claim a person's real identity unless the audio itself provides it.
12. If the recording contains no understandable speech, say so clearly.

Return ONLY valid JSON.

Use exactly this structure:

{
  "language": "detected language",
  "transcript": "complete transcript",
  "summary": "short summary",
  "key_points": [
    "point 1",
    "point 2"
  ],
  "keywords": [
    "keyword 1",
    "keyword 2"
  ],
  "speakers": [
    {
      "speaker": "Speaker 1",
      "description": "description if available"
    }
  ]
}
"""


# ============================================================
# SESSION STATE
# ============================================================

if "audio_result" not in st.session_state:
    st.session_state.audio_result = None

if "audio_model" not in st.session_state:
    st.session_state.audio_model = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Settings")

    available_models = model_candidates()

    if available_models:
        selected_model = st.selectbox(
            "Gemini model",
            available_models,
            index=0,
        )
    else:
        selected_model = st.text_input(
            "Gemini model",
            value="gemini-2.5-flash",
        )

    st.divider()

    st.markdown(
        """
        **Supported audio**

        - MP3
        - WAV
        - M4A
        - OGG
        - FLAC
        - AAC
        - WebM
        """
    )


# ============================================================
# AUDIO UPLOAD
# ============================================================

uploaded_audio = st.file_uploader(
    "Upload an audio file",
    type=[
        "mp3",
        "wav",
        "m4a",
        "ogg",
        "flac",
        "aac",
        "webm",
    ],
)

if uploaded_audio:
    audio_bytes = uploaded_audio.getvalue()
    mime_type = get_mime_type(uploaded_audio.name)

    st.audio(
        audio_bytes,
        format=mime_type,
    )

    file_size_mb = len(audio_bytes) / (1024 * 1024)

    st.caption(
        f"File: {uploaded_audio.name} · "
        f"Size: {file_size_mb:.2f} MB · "
        f"Type: {mime_type}"
    )

    if st.button(
        "🎙️ Transcribe & Analyze Audio",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.audio_result = None
        st.session_state.audio_model = None

        candidates = []

        # Try the selected model first.
        if selected_model:
            candidates.append(selected_model)

        # Then try discovered/fallback models.
        for model in model_candidates():
            if model not in candidates:
                candidates.append(model)

        errors = []

        with st.spinner("🎙️ Analyzing audio..."):
            for model in candidates:
                try:
                    raw_response = call_gemini_audio(
                        audio_bytes=audio_bytes,
                        mime_type=mime_type,
                        prompt=AUDIO_PROMPT,
                        model_name=model,
                    )

                    result = parse_json(raw_response)

                    # Ensure expected fields exist.
                    result.setdefault("language", "Unknown")
                    result.setdefault("transcript", "")
                    result.setdefault("summary", "")
                    result.setdefault("key_points", [])
                    result.setdefault("keywords", [])
                    result.setdefault("speakers", [])

                    st.session_state.audio_result = result
                    st.session_state.audio_model = model

                    break

                except Exception as error:
                    errors.append(
                        f"{model}: {error}"
                    )

            else:
                st.error("Audio analysis failed.")

                if errors:
                    with st.expander("Technical error details"):
                        for error in errors:
                            st.write(error)


# ============================================================
# RESULTS
# ============================================================

result = st.session_state.audio_result

if result:
    st.success(
        f"Audio analysis completed using `{st.session_state.audio_model}`."
    )

    tab_transcript, tab_summary, tab_points, tab_keywords, tab_speakers, tab_download = st.tabs(
        [
            "📝 Transcript",
            "📋 Summary",
            "🔑 Key Points",
            "🏷️ Keywords",
            "👥 Speakers",
            "⬇️ Downloads",
        ]
    )

    with tab_transcript:
        st.subheader("Transcript")

        language = result.get("language", "Unknown")

        st.info(f"Detected language: **{language}**")

        transcript = result.get(
            "transcript",
            "",
        )

        if transcript:
            st.text_area(
                "Full transcript",
                transcript,
                height=400,
            )
        else:
            st.warning("No transcript was returned.")

    with tab_summary:
        st.subheader("Summary")

        summary = result.get(
            "summary",
            "",
        )

        if summary:
            st.write(summary)
        else:
            st.info("No summary was returned.")

    with tab_points:
        st.subheader("Key Points")

        points = result.get(
            "key_points",
            [],
        )

        if points:
            for point in points:
                st.markdown(f"- {point}")
        else:
            st.info("No key points were returned.")

    with tab_keywords:
        st.subheader("Keywords")

        keywords = result.get(
            "keywords",
            [],
        )

        if keywords:
            st.write(" · ".join(map(str, keywords)))
        else:
            st.info("No keywords were returned.")

    with tab_speakers:
        st.subheader("Speakers")

        speakers = result.get(
            "speakers",
            [],
        )

        if speakers:
            for speaker in speakers:
                if isinstance(speaker, dict):
                    name = speaker.get(
                        "speaker",
                        "Speaker",
                    )

                    description = speaker.get(
                        "description",
                        "",
                    )

                    st.markdown(
                        f"**{name}**"
                    )

                    if description:
                        st.write(description)

                    st.divider()
                else:
                    st.write(str(speaker))
        else:
            st.info(
                "No speaker information was returned."
            )

    with tab_download:
        st.subheader("Download Results")

        transcript = result.get(
            "transcript",
            "",
        )

        transcript_text = (
            f"Detected language: "
            f"{result.get('language', 'Unknown')}\n\n"
            f"{transcript}"
        )

        st.download_button(
            "⬇️ Download Transcript",
            data=transcript_text,
            file_name="audio_transcript.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.download_button(
            "⬇️ Download JSON Analysis",
            data=json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            ),
            file_name="audio_analysis.json",
            mime="application/json",
            use_container_width=True,
        )
