import streamlit as st
from transformers import pipeline
import pandas as pd
from annotated_text import annotated_text
import plotly.express as px

st.title("Emotion Tagging for Reviews")

text = 0
l = []

st.write("**Enter review and click outside the text area to run the model**")
text = st.text_area("")
print("text")
print(type(text))
if text != 0 and bool(text.strip()) is True:
    emotion = pipeline('sentiment-analysis',
                       model='arpanghoshal/EmoRoBERTa', return_all_scores=True)
    emotion_labels = emotion(text)

    for i in range(len(emotion_labels[0])):
        l.append(list(emotion_labels[0][i].values()))

    l = pd.DataFrame(l, columns=['emotion', 'score'])
    z = l.sort_values(by=['score'], ascending=False)

    t = z.head(3)

    st.subheader('Tags')
    annotated_text(
        "  ",
        (z.iloc[0][0], str(round(z.iloc[0][1], 5)), '#63B475'),
        "  ",
        (z.iloc[1][0], str(round(z.iloc[1][1], 5)), '#ADD576'),
        "   ",
        (z.iloc[2][0], str(round(z.iloc[2][1], 5)), '#CDEF9D')
    )

    st.subheader("")
    st.subheader('Emotion with top 3 scores')
    fig = px.bar(t, x="score", y="emotion", color='score', height=300, color_continuous_scale='emrld')
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig)

    st.subheader("Other emotions detected")
    z = z.drop(z[['score']].idxmax())
    fig = px.bar(z, x="score", y="emotion", color='score', height=800, color_continuous_scale='emrld')
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig)
