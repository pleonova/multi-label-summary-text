---
title: Multi Label Summary Text
emoji: 📚
colorFrom: indigo
colorTo: gray
sdk: streamlit
python_version: 3.9.13
app_file: app.py
pinned: false
---

#### Interactive version
This app is hosted on HuggingFace spaces: https://huggingface.co/spaces/pleonova/multi-label-summary-text

#### Objective
The goal of this app is to identify multiple relevant labels for long text.

#### Model
facebook/bart-large-mnli zero-shot transfer-learning summarizer and classifier

#### Approach
Updating the head of the neural network, we can use the same pretrained bart model to first summarize our long text by first splitting out our long text into chunks of 1024 tokens and then generating a summary for each of the text chunks. Next, all the summaries are concanenated and the bart model is used classify the summarized text. Alternatively, one can also classify the whole text as is.
