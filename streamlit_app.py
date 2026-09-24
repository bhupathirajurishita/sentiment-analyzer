from transformers import pipeline
import streamlit as st
import random

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="How's The Vibe?",
    page_icon="🧠",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.stApp {
    background: #0b0d0f;
    color: #f5f5f5;
}

.block-container {
    max-width: 900px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Title */

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 8px;
}

.subtitle {
    text-align: center;
    color: #9aa3ad;
    font-size: 16px;
    margin-bottom: 35px;
}

/* Text area */

textarea {
    background: #0e1114 !important;
    color: white !important;
    border: 1px solid #303840 !important;
    border-radius: 12px !important;
}

/* Labels */

.stTextArea label {
    color: #f5f5f5 !important;
    font-weight: 600 !important;
}

/* Buttons */

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 50px;
    font-weight: 700;
    border: 1px solid #293139;
}

/* Button hover */

.stButton > button:hover {
    border-color: #21c78a;
}

/* Example selector */

div[data-baseweb="select"] > div {
    background: #12161a !important;
    border-color: #303840 !important;
}

/* Result */

.result-card {
    background: #12161a;
    border: 1px solid #293139;
    border-radius: 16px;
    padding: 25px;
    margin-top: 25px;
}

.reaction {
    font-size: 20px;
    font-weight: 600;
    padding: 18px;
    background: #0d1114;
    border-radius: 12px;
    border-left: 3px solid #21c78a;
    margin-bottom: 22px;
}

.result-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 30px;
}

.result-label {
    color: #7f8993;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
}

.result-value {
    margin-top: 6px;
    font-size: 21px;
    font-weight: 700;
}

.footer {
    text-align: center;
    color: #68727c;
    font-size: 12px;
    margin-top: 40px;
}

/* Mobile */

@media (max-width: 600px) {

    .title {
        font-size: 32px;
    }

    .result-row {
        grid-template-columns: 1fr;
        gap: 18px;
    }

}

</style>
""", unsafe_allow_html=True)


# ---------------- AI MODEL ----------------

@st.cache_resource
def load_model():

    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )


sentiment_model = load_model()


# ---------------- FUNNY REACTIONS ----------------

REACTIONS = {

    "POSITIVE": [
        "🚀 WHO GAVE YOU THIS MUCH HAPPINESS?",
        "😎 Okayyy, main character energy.",
        "✨ The vibes are immaculate.",
        "🥳 Someone's having a VERY good day."
    ],

    "NEGATIVE": [
        "🫠 Yeah... that hurt a little.",
        "😭 We might need snacks.",
        "🫂 This sentence needs emotional support.",
        "💀 Okay. Who upset you?"
    ]

}


# ---------------- TITLE ----------------

st.markdown(
    '<div class="title">🧠 HOW\'S THE VIBE? 👀</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Let\'s see what your text is feeling.</div>',
    unsafe_allow_html=True
)


# ---------------- INPUT ----------------

text = st.text_area(
    "Tell me what's on your mind...",
    placeholder="Type something like: I had an amazing day today!",
    height=140
)


# ---------------- EXAMPLES ----------------

st.markdown(
    "##### 💡 Try an example"
)

example = st.selectbox(
    "Choose one",
    [
        "Select an example...",
        "I finally finished my project and I'm so happy!",
        "Everything went wrong today.",
        "This is the best day ever!",
        "I don't really know how I feel about this."
    ]
)

if example != "Select an example...":

    if st.button("Use Example"):

        st.session_state["selected_example"] = example
        st.rerun()


# ---------------- BUTTONS ----------------

col1, col2 = st.columns(2)

with col1:

    analyze = st.button(
        "✨ Analyze Vibe",
        use_container_width=True
    )

with col2:

    clear = st.button(
        "🧹 Clear",
        use_container_width=True
    )


# ---------------- CLEAR ----------------

if clear:

    st.session_state.clear()
    st.rerun()


# ---------------- ANALYSIS ----------------

if analyze:

    if not text.strip():

        st.warning(
            "👀 Give me something to analyze first!"
        )

    else:

        # Run AI model

        result = sentiment_model(text)[0]

        sentiment = result["label"]

        confidence = result["score"] * 100

        # Random personality reaction

        reaction = random.choice(
            REACTIONS[sentiment]
        )

        # Sentiment emoji

        emoji = (
            "😊"
            if sentiment == "POSITIVE"
            else "😶‍🌫️"
        )

        # ---------------- RESULT CARD ----------------

        st.html(
            f"""
            <div class="result-card">

                <div class="reaction">
                    {reaction}
                </div>

                <div class="result-row">

                    <div>
                        <div class="result-label">
                            SENTIMENT
                        </div>

                        <div class="result-value">
                            {emoji} {sentiment.capitalize()}
                        </div>
                    </div>

                    <div>
                        <div class="result-label">
                            CONFIDENCE
                        </div>

                        <div class="result-value">
                            {confidence:.2f}%
                        </div>
                    </div>

                </div>

            </div>
            """
        )


# ---------------- FOOTER ----------------

st.markdown(
    """
    <div class="footer">
        Built with Python · Hugging Face Transformers · Streamlit
    </div>
    """,
    unsafe_allow_html=True
)