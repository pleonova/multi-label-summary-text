---
title: Multi Label Long Text
emoji: 📚
colorFrom: indigo
colorTo: gray
sdk: streamlit
app_file: app.py
pinned: false
---

**Interactive version**: This app is hosted on https://huggingface.co/spaces/pleonova/multi-label-long-text

**Objectvie**: As the name may suggest, the goal of this app is to identify multiple relevant labels for long text.

**Model**: zero-shot learning - facebook/bart-large-mnli summarizer and classifier

**Approach**: Updating the head of the neural network, we can use the same pretrained bart model to first summarize our long text by first splitting out our long text into chunks of 1024 tokens and then generating a summary for each of the text chunks. Next, all the summaries are concanenated and the bart model is used classify the summarized text. Alternatively, one can also classify the whole text as is.
