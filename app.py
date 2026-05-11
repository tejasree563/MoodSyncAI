import os
from functools import lru_cache

import gradio as gr
import pandas as pd
import torch
import numpy as np

from PIL import (
    Image,
    ImageEnhance,
)

from transformers import pipeline
from deepface import DeepFace


# APP CONFIG

APP_TITLE = "MoodSyncAI"

TEXT_MODEL_ID = os.getenv(
    "TEXT_MODEL_ID",
    "j-hartmann/emotion-english-distilroberta-base"
)

ASR_MODEL_ID = os.getenv(
    "ASR_MODEL_ID",
    "openai/whisper-tiny.en"
)

# EMOTIONS

EMOTIONS = [
    "happy",
    "sad",
    "angry",
    "fearful",
    "surprised",
    "disgust",
    "neutral",
    "confused",
    "calm",
    "frustrated",
]

EMOTION_STATES = {

    "happy": "Positive Engagement",

    "sad": "Distress",

    "angry": "High Tension",

    "fearful": "Anxiety",

    "surprised": "Heightened Reaction",

    "disgust": "Strong Aversion",

    "neutral": "Stable State",

    "confused": "Cognitive Uncertainty",

    "calm": "Relaxed Stability",

    "frustrated": "Emotional Stress",
}


# DEVICE

def _device():
    return 0 if torch.cuda.is_available() else -1


# CLEAN LABELS

def _clean_label(label):

    label = str(label).lower().strip()

    aliases = {

        "anger": "angry",
        "sadness": "sad",
        "joy": "happy",
        "fear": "fearful",
        "surprise": "surprised",
        "frustration": "frustrated",
        "confusion": "confused",
    }

    return aliases.get(label, label)


# ENHANCED EMOTION

def enhance_emotion(label, confidence):
    if label == "neutral" and confidence > 0.75:
        return "calm"
    
    if label == "angry" and confidence > 0.70:
        return "frustrated"

    return label


# BLANK SCORES

def _blank_scores():
    """_summary_

    Returns:
        _type_: _description_
    """

    return {emotion: 0.0 for emotion in EMOTIONS}
# TOP EMOTION

def _top(scores):

    if not scores:
        return "unknown", 0.0

    if sum(scores.values()) == 0:
        return "unknown", 0.0

    label = max(scores, key=scores.get)

    confidence = scores[label]

    enhanced = enhance_emotion(
        label,
        confidence
    )

    return enhanced, confidence


# DATAFRAME

def _scores_frame(scores):

    df = pd.DataFrame(
        [
            {
                "emotion": k.title(),
                "confidence": round(v * 100, 1)
            }
            for k, v in scores.items()
            if v > 0.5 / 100
        ]
    )

    if len(df) == 0:

        df = pd.DataFrame(
            [
                {
                    "emotion": "Neutral",
                    "confidence": 0
                }
            ]
        )

    return df.sort_values(
        "confidence",
        ascending=False
    )

# TEXT MODEL
@lru_cache(maxsize=1)
def text_classifier():
    return pipeline(
        "text-classification",
        model=TEXT_MODEL_ID,
        top_k=None,
        device=_device()
    )

# AUDIO MODEL
@lru_cache(maxsize=1)
def speech_recogniser():

    return pipeline(
        "automatic-speech-recognition",
        model=ASR_MODEL_ID,
        chunk_length_s=20,
        device=_device(),
    )

# IMAGE CLASSIFICATION
def classify_image(image):
    if image is None:
        return _blank_scores()

    try:

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB")

        image = ImageEnhance.Contrast(
            image
        ).enhance(1.2)

        image = ImageEnhance.Sharpness(
            image
        ).enhance(1.1)

        image_np = np.array(image)

        result = DeepFace.analyze(
            img_path=image_np,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True,
        )

        if isinstance(result, list):
            result = result[0]

        emotions = result.get("emotion", {})

        scores = _blank_scores()

        mapping = {

            "angry": "angry",
            "disgust": "disgust",
            "fear": "fearful",
            "happy": "happy",
            "sad": "sad",
            "surprise": "surprised",
            "neutral": "neutral",
        }

        for k, v in emotions.items():

            clean = mapping.get(
                k.lower()
            )

            if clean in scores:

                scores[clean] = (
                    float(v) / 100.0
                )

        if scores["neutral"] > 0.45:
            scores["neutral"] += 0.20

        if scores["surprised"] < 0.35:
            scores["surprised"] *= 0.35

        if scores["fearful"] < 0.35:
            scores["fearful"] *= 0.35

        total = sum(scores.values())

        if total > 0:

            scores = {
                k: v / total
                for k, v in scores.items()
            }

        return scores

    except Exception as e:

        print("IMAGE ERROR:", e)

        return _blank_scores()


# TEXT CLASSIFICATION

def classify_text(text):

    if not text or not text.strip():
        return _blank_scores()

    try:

        result = text_classifier()(text.strip())

        # transformers may return nested list
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list):
                result = result[0]

        scores = _blank_scores()

        for row in result:

            raw_label = str(
                row.get("label", "")
            ).lower().strip()

            score = float(
                row.get("score", 0)
            )

            aliases = {

                "joy": "happy",
                "sadness": "sad",
                "anger": "angry",
                "fear": "fearful",
                "surprise": "surprised",

                "happy": "happy",
                "sad": "sad",
                "angry": "angry",
                "fearful": "fearful",
                "surprised": "surprised",

                "neutral": "neutral",
                "disgust": "disgust",
            }

            label = aliases.get(raw_label)

            if label:
                scores[label] += score

        total = sum(scores.values())

        if total > 0:

            scores = {
                k: v / total
                for k, v in scores.items()
            }

        return scores

    except Exception as e:

        print("TEXT ERROR:", e)

        return _blank_scores()

# AUDIO TRANSCRIPTION

def transcribe_audio(audio_path):

    if not audio_path:
        return ""

    try:

        result = speech_recogniser()(audio_path)

        return result.get("text", "").strip()

    except Exception as e:

        print("AUDIO ERROR:", e)

        return ""


# FUSION

def fuse_scores(
    visual_scores,
    text_scores,
    visual_weight,
):

    visual_strength = max(
        visual_scores.values()
    )

    text_strength = max(
        text_scores.values()
    )

    if visual_strength == 0 and text_strength == 0:
        return _blank_scores()

    if visual_strength > 0.75:
        visual_weight = 0.80

    elif visual_strength > 0.60:
        visual_weight = 0.70

    elif text_strength > 0.85:
        visual_weight = 0.40

    text_weight = 1.0 - visual_weight

    fused = {}

    for emotion in EMOTIONS:

        visual_part = (
            visual_scores.get(emotion, 0.0)
            * visual_weight
        )

        text_part = (
            text_scores.get(emotion, 0.0)
            * text_weight
        )

        fused[emotion] = (
            visual_part + text_part
        )

    total = sum(fused.values())

    if total > 0:

        fused = {
            k: v / total
            for k, v in fused.items()
        }

    return fused


# ALIGNMENT

def detect_alignment(
    visual_top,
    text_top,
    visual_conf,
    text_conf,
):

    emotion_similarity = {

        "angry": {"frustrated", "disgust"},
        "frustrated": {"angry", "disgust"},

        "sad": {"neutral", "calm"},
        "neutral": {"sad", "calm"},
        "calm": {"neutral"},

        "fearful": {"confused"},
        "confused": {"fearful"},

        "happy": {"surprised"},
        "surprised": {"happy"},
    }

    if visual_top == text_top:

        return (
            "ALIGNED",
            "success",
            "Both modalities strongly agree."
        )

    if text_top in emotion_similarity.get(
        visual_top,
        set()
    ):

        return (
            "PARTIALLY ALIGNED",
            "info",
            "Emotionally related states detected."
        )

    if (
        visual_conf >= 0.45
        and text_conf >= 0.45
    ):

        return (
            "MISMATCH DETECTED",
            "warning",
            "Face and language conflict emotionally."
        )

    return (
        "MIXED EMOTION",
        "secondary",
        "Emotional signals are blended."
    )


# SUMMARY

def generate_summary(
    visual_top,
    text_top,
    fused_top,
    badge,
):

    if badge == "MISMATCH DETECTED":

        return (
            f"The visual model reads the face as "
            f"{visual_top}, while the text/audio "
            f"signal is closer to {text_top}. "
            f"The fused emotional state leans "
            f"{fused_top}, so the system flags "
            "a mismatch between facial "
            "expression and language."
        )

    if badge == "PARTIALLY ALIGNED":

        return (
            f"The emotions are contextually related. "
            f"The face appears {visual_top}, while "
            f"the text reflects {text_top}."
        )

    if badge == "ALIGNED":

        return (
            f"Both visual and text/audio signals "
            f"strongly indicate {fused_top}."
        )

    return (
        f"The fused emotional state is "
        f"{fused_top}."
    )

# MAIN ANALYSIS

def analyse(
    image,
    text,
    audio,
    visual_weight,
):

    try:

        audio_text = ""

        if audio:
            audio_text = transcribe_audio(audio)

        final_text = " ".join(
            [
                text.strip() if text else "",
                audio_text,
            ]
        ).strip()

        visual_scores = classify_image(image)

        text_scores = classify_text(final_text)

        fused_scores = fuse_scores(
            visual_scores,
            text_scores,
            visual_weight,
        )

        visual_top, visual_conf = _top(
            visual_scores
        )

        text_top, text_conf = _top(
            text_scores
        )

        fused_top, fused_conf = _top(
            fused_scores
        )

        badge, tone, note = detect_alignment(
            visual_top,
            text_top,
            visual_conf,
            text_conf,
        )

        summary = generate_summary(
            visual_top,
            text_top,
            fused_top,
            badge,
        )

        metrics = pd.DataFrame(
            [
                ["Visual Emotion",
                 f"{visual_top.title()} ({visual_conf:.0%})"],

                ["Text Emotion",
                 f"{text_top.title()} ({text_conf:.0%})"],

                ["Fusion Result",
                 f"{fused_top.title()} ({fused_conf:.0%})"],

                ["Emotional State",
                 EMOTION_STATES.get(fused_top)],

                ["Alignment",
                 badge],

                ["Audio Transcript",
                 audio_text or "No audio transcript"],
            ],
            columns=["Metric", "Value"]
        )

        badge_html = f"""
        <div class="status {tone}">
          <span>{badge}</span>
          <small>{note}</small>
        </div>
        """

        return (
            _scores_frame(visual_scores),
            _scores_frame(text_scores),
            _scores_frame(fused_scores),
            metrics,
            badge_html,
            summary,
            final_text,
        )

    except Exception as e:

        print("MAIN ERROR:", e)

        empty_df = _scores_frame(
            _blank_scores()
        )

        error_metrics = pd.DataFrame(
            [
                ["Error", str(e)]
            ],
            columns=["Metric", "Value"]
        )

        return (
            empty_df,
            empty_df,
            empty_df,
            error_metrics,
            "<div class='status warning'>ERROR</div>",
            str(e),
            "",
        )


# LOAD CSS

with open("styles.css", "r") as f:
    CSS = f.read()


# THEME

APP_THEME = gr.themes.Soft(
    primary_hue="cyan",
    neutral_hue="slate",
)


# UI

with gr.Blocks(
    title=APP_TITLE,
    css=CSS,
    theme=APP_THEME,
) as demo:

    gr.HTML(
        """
        <section class="hero">

          <div>

            <h1>MoodSyncAI</h1>

            <p>
              Multi-modal emotional state inference
              using transformer-based fusion across
              visual, linguistic, and speech modalities.
            </p>

          </div>

          <div class="hero-metrics">

            <div class="metric-pill">
              <strong>DeepFace</strong>
              <span>visual emotion</span>
            </div>

            <div class="metric-pill">
              <strong>RoBERTa</strong>
              <span>text emotion</span>
            </div>

            <div class="metric-pill">
              <strong>Whisper</strong>
              <span>audio transcript</span>
            </div>

          </div>

        </section>
        """
    )

    with gr.Row():

        with gr.Column(scale=5):

            with gr.Row():

                image = gr.Image(
                    label="Upload Photo",
                    type="pil",
                    height=320,
                )

                audio = gr.Audio(
                    label="Optional Audio",
                    sources=["upload", "microphone"],
                    type="filepath",
                )

            text = gr.Textbox(
                label="Text Statement or Transcript",
                placeholder="Type text here...",
                lines=4,
            )

            visual_weight = gr.Slider(
                0.0,
                1.0,
                value=0.55,
                step=0.05,
                label="Visual Fusion Weight",
            )

            run = gr.Button(
                "Analyse Emotion",
                variant="primary",
            )

        with gr.Column(scale=4):

            badge = gr.HTML()

            metrics = gr.Dataframe(
                headers=["Metric", "Value"],
                datatype=["str", "str"],
                interactive=False,
                wrap=True,
                label="Key Outputs",
            )

            summary = gr.Textbox(
                label="Generated Summary",
                lines=6,
            )

            transcript = gr.Textbox(
                label="Final Text Analysed",
                lines=3,
            )

    with gr.Row():

        visual_plot = gr.BarPlot(
            label="Visual Emotion Distribution",
            x="emotion",
            y="confidence",
            y_lim=[0, 100],
            height=320,
        )

        text_plot = gr.BarPlot(
            label="Text Emotion Distribution",
            x="emotion",
            y="confidence",
            y_lim=[0, 100],
            height=320,
        )

        fused_plot = gr.BarPlot(
            label="Fusion Emotion Distribution",
            x="emotion",
            y="confidence",
            y_lim=[0, 100],
            height=320,
        )
        
        
    run.click(
        fn=analyse,
        inputs=[
            image,
            text,
            audio,
            visual_weight,
        ],
        outputs=[
            visual_plot,
            text_plot,
            fused_plot,
            metrics,
            badge,
            summary,
            transcript,
        ],
    )
visual_plot = gr.BarPlot(
    label="Visual Emotion Distribution",
    x="emotion",
    y="confidence",
    y_lim=[0, 100],
    height=350,
    tooltip=["emotion", "confidence"],
)

text_plot = gr.BarPlot(
    label="Text Emotion Distribution",
    x="emotion",
    y="confidence",
    y_lim=[0, 100],
    height=350,
    tooltip=["emotion", "confidence"],
)

fused_plot = gr.BarPlot(
    label="Fusion Emotion Distribution",
    x="emotion",
    y="confidence",
    y_lim=[0, 100],
    height=350,
    tooltip=["emotion", "confidence"],
)

# LAUNCH

if __name__ == "__main__":

    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=8502,
        share=True,
    )