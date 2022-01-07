import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import streamlit as st
from keybert import KeyBERT


import spacy
nlp = spacy.load('en_core_web_sm')

# Reference: https://discuss.huggingface.co/t/summarization-on-long-documents/920/7
def create_nest_sentences(document:str, token_max_length = 1024):
  nested = []
  sent = []
  length = 0
  tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
  tokens = nlp(document)

  for sentence in tokens.sents:
    tokens_in_sentence = tokenizer(str(sentence), truncation=False, padding=False)[0] # hugging face transformer tokenizer
    length += len(tokens_in_sentence)

    if length < token_max_length:
      sent.append(sentence)
    else:
      nested.append(sent)
      sent = []
      length = 0

  if sent:
    nested.append(sent)
  return nested

# Reference: https://github.com/MaartenGr/KeyBERT
@st.cache(allow_output_mutation=True)
def load_keyword_model():
  kw_model = KeyBERT()
  return kw_model

def keyword_gen(kw_model, sequence:str):
  keywords = kw_model.extract_keywords(sequence, 
    keyphrase_ngram_range=(1, 1),
    stop_words='english', 
    use_mmr=True, 
    diversity=0.5,
    top_n=10)
  return keywords



# Reference: https://huggingface.co/facebook/bart-large-mnli
@st.cache(allow_output_mutation=True)
def load_summary_model():
    model_name = "facebook/bart-large-mnli"
    summarizer = pipeline(task='summarization', model=model_name)
    return summarizer

# def load_summary_model():
#     model_name = "facebook/bart-large-mnli"
#     tokenizer = BartTokenizer.from_pretrained(model_name)
#     model = BartForConditionalGeneration.from_pretrained(model_name)
#     summarizer = pipeline(task='summarization', model=model, tokenizer=tokenizer, framework='pt')
#     return summarizer

def summarizer_gen(summarizer, sequence:str, maximum_tokens:int, minimum_tokens:int):
	output = summarizer(sequence, 
    num_beams=4, 
    length_penalty=2.0,
    max_length=maximum_tokens, 
    min_length=minimum_tokens, 
    do_sample=False, 
    early_stopping = True,
    no_repeat_ngram_size=3)
	return output[0].get('summary_text')


# # Reference: https://www.datatrigger.org/post/nlp_hugging_face/
# # Custom summarization pipeline (to handle long articles)
# def summarize(text, minimum_length_of_summary = 100):
#     # Tokenize and truncate
#     inputs = tokenizer_bart([text], truncation=True, max_length=1024, return_tensors='pt').to('cuda')
#     # Generate summary 
#     summary_ids = model_bart.generate(inputs['input_ids'], num_beams=4, min_length = minimum_length_of_summary, max_length=400, early_stopping=True)
#     # Untokenize
#     return([tokenizer_bart.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0])


# Reference: https://huggingface.co/spaces/team-zero-shot-nli/zero-shot-nli/blob/main/utils.py
@st.cache(allow_output_mutation=True)
def load_model():
    model_name = "facebook/bart-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline(task='zero-shot-classification', model=model, tokenizer=tokenizer, framework='pt')
    return classifier

def classifier_zero(classifier, sequence:str, labels:list, multi_class:bool):
    outputs = classifier(sequence, labels, multi_label=multi_class)
    return outputs['labels'], outputs['scores']

