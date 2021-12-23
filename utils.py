import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import json

# Reference: https://huggingface.co/spaces/team-zero-shot-nli/zero-shot-nli/blob/main/utils.py
def plot_result(top_topics, scores):
    top_topics = np.array(top_topics)
    scores = np.array(scores)
    scores *= 100
    fig = px.bar(x=np.around(scores,2), y=top_topics, orientation='h', 
                 labels={'x': 'Confidence Score', 'y': 'Label'},
                 text=scores,
                 range_x=(0,115),
                 title='Predictions',
                 color=np.linspace(0,1,len(scores)),
                 color_continuous_scale='GnBu')
    fig.update(layout_coloraxis_showscale=False)
    fig.update_traces(texttemplate='%{text:0.1f}%', textposition='outside')
    st.plotly_chart(fig)


def plot_dual_bar_chart(topics_summary, scores_summary, topics_text, scores_text):
    data1 = pd.DataFrame({'label': topics_summary, 'scores on summary': scores_summary})
    data2 = pd.DataFrame({'label': topics_text, 'scores on full text': scores_text})
    data = pd.merge(data1, data2, on = ['label'])
    data.sort_values('scores on summary', ascending = True, inplace = True)

    fig = make_subplots(rows=1, cols=2, 
        subplot_titles=("Predictions on Summary", "Predictions on Full Text"),
        )

    fig1 = px.bar(x=round(data['scores on summary']*100, 2), y=data['label'], orientation='h', 
                 text=round(data['scores on summary']*100, 2),
                 )

    fig2 = px.bar(x=round(data['scores on full text']*100,2), y=data['label'], orientation='h', 
                 text=round(data['scores on full text']*100,2),
                 )

    fig.add_trace(fig1['data'][0], row=1, col=1)
    fig.add_trace(fig2['data'][0], row=1, col=2)

    fig.update_traces(texttemplate='%{text:0.1f}%', textposition='outside')
    fig.update_layout(height=600, width=700) #, title_text="Predictions for")
    fig.update_xaxes(range=[0,115])
    fig.update_xaxes(matches='x')
    fig.update_yaxes(showticklabels=False) # hide all the xticks
    fig.update_yaxes(showticklabels=True, row=1, col=1)

    st.plotly_chart(fig)

# def plot_dual_bar_chart(topics_summary, scores_summary, topics_text, scores_text):
#     data1 = pd.DataFrame({'label': topics_summary, 'scores': scores_summary})
#     data1['classification_on'] = 'summary'
#     data2 = pd.DataFrame({'label': topics_text, 'scores': scores_text})
#     data2['classification_on'] = 'full text'
#     data = pd.concat([data1, data2])
#     data['scores'] = round(data['scores']*100,2)

#     fig = px.bar(
#         data, x="scores", y="label", #orientation = 'h',
#                  labels={'x': 'Confidence Score', 'y': 'Label'},
#                  text=data['scores'],
#                  range_x=(0,115),
#                  color="label", barmode="group", 
#                  facet_col="classification_on",
#                  category_orders={"classification_on": ["summary", "full text"]}
#        )
#     fig.update_traces(texttemplate='%{text:0.1f}%', textposition='outside')

#     st.plotly_chart(fig)


def examples_load():
    with open("examples.json") as f:
        data=json.load(f)
    return data['text'], data['long_text_license'], data['labels'], data['ground_labels']

def example_long_text_load():
    with open("example_long_text.txt", "r") as f:
        text_data = f.read()
    return text_data
