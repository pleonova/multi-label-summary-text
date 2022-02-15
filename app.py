
from os import write
import time
import pandas as pd
import base64
from typing import Sequence
import streamlit as st
from sklearn.metrics import classification_report


# from models import create_nest_sentences, load_summary_model, summarizer_gen, load_model, classifier_zero
import models as md
from utils import plot_result, plot_dual_bar_chart, examples_load, example_long_text_load
import json

ex_text, ex_license, ex_labels, ex_glabels = examples_load()
ex_long_text = example_long_text_load()


# if __name__ == '__main__':
st.markdown("### Long Text Summarization & Multi-Label Classification")
st.write("This app summarizes and then classifies your long text with multiple labels using [BART Large MNLI](https://huggingface.co/facebook/bart-large-mnli). The keywords are generated using [KeyBERT](https://github.com/MaartenGr/KeyBERT).")
st.write("__Inputs__: User enters their own custom text and labels.")
st.write("__Outputs__: A summary of the text, likelihood percentages for each label and a downloadable csv of the results. \
    Includes additional options to generate a list of keywords and/or evaluate results against a list of ground truth labels, if available.")

example_button = st.button(label='See Example')
if example_button:
    example_text = ex_long_text #ex_text
    display_text = 'Excerpt from Frankenstein:' + example_text + '"\n\n' + "[This is an excerpt from Project Gutenberg's Frankenstein. " + ex_license + "]"
    input_labels = ex_labels
    input_glabels = ex_glabels
else:
    display_text = ''
    input_labels = ''
    input_glabels = ''


with st.form(key='my_form'):
    st.markdown("##### Step 1: Upload Text")
    text_input = st.text_area("Input any text you want to summarize & classify here (keep in mind very long text will take a while to process):", display_text)

    text_csv_expander = st.expander(label=f'Want to upload multiple texts at once? Expand to upload your text files below.', expanded=False)
    with text_csv_expander:
        st.write("Option A:")
        uploaded_text_files = st.file_uploader(label="Upload file(s) that end with the .txt suffix",
                                              accept_multiple_files=True, key = 'text_uploader',
                                              type = 'txt')
        st.write("Option B:")
        uploaded_csv_text_files = st.file_uploader(label='Upload a CSV file with columns: "title" and "text"',
                                                   accept_multiple_files=False, key = 'csv_text_uploader',
                                                   type = 'csv')

    if text_input == display_text and display_text != '':
        text_input = example_text


    st.text("\n\n\n")
    st.markdown("##### Step 2: Enter Labels")
    labels = st.text_input('Enter possible topic labels, which can be either keywords and/or general themes (comma-separated):',input_labels, max_chars=1000)
    labels = list(set([x.strip() for x in labels.strip().split(',') if len(x.strip()) > 0]))

    labels_csv_expander = st.expander(label=f'Prefer to upload a list of labels instead? Click here to upload your CSV file.',expanded=False)
    with labels_csv_expander:
        uploaded_labels_file = st.file_uploader("Or Choose a CSV file with one column and no header, where each cell is a separate label",
                                                key='labels_uploader')

    gen_keywords = st.radio(
        "Generate keywords from text (independent from the above labels)?",
        ('Yes', 'No')
        )

    st.text("\n\n\n")
    st.markdown("##### Step 3: Provide Ground Truth Labels (_Optional_)")
    glabels = st.text_input('If available, enter ground truth topic labels to evaluate results, otherwise leave blank (comma-separated):',input_glabels, max_chars=1000)
    glabels = list(set([x.strip() for x in glabels.strip().split(',') if len(x.strip()) > 0]))


    glabels_csv_expander = st.expander(label=f'Have a file with labels for the text? Click here to upload your CSV file.', expanded=False)
    with glabels_csv_expander:
        st.write("Option A:")
        uploaded_onetext_glabels_file = st.file_uploader("Choose a CSV file with one column and no header, where each cell is a separate label",
                                                         key = 'onetext_glabels_uploader')
        st.write("Option B:")
        uploaded_multitext_glabels_file = st.file_uploader('Or Choose a CSV file with two columns "title" and "label", with the cells in the title column matching the name of the files uploaded in step #1.',
                                                           key = 'multitext_glabels_uploader')



    threshold_value = st.slider(
         'Select a threshold cutoff for matching percentage (used for ground truth label evaluation)',
         0.0, 1.0, (0.5))

    submit_button = st.form_submit_button(label='Submit')

st.write("_For improvments/suggestions, please file an issue here: https://github.com/pleonova/multi-label-summary-text_")

with st.spinner('Loading pretrained models...'):
    start = time.time()
    summarizer = md.load_summary_model()   
    s_time = round(time.time() - start,4)

    start = time.time()
    classifier = md.load_model()  
    c_time = round(time.time() - start,4)

    start = time.time()
    kw_model = md.load_keyword_model()
    k_time = round(time.time() - start,4)

    st.success(f'Time taken to load various models: {k_time}s for KeyBERT model & {s_time}s for BART summarizer mnli model & {c_time}s for BART classifier mnli model.')


if submit_button or example_button:
    if len(text_input) == 0 and uploaded_text_files is None and uploaded_csv_text_files is None:
        st.error("Enter some text to generate a summary")
    else:

        if uploaded_text_files is not None:
            st.markdown("### Text Inputs")
            file_names = []
            raw_texts = []
            for uploaded_file in uploaded_text_files:
                text = str(uploaded_file.read(), "utf-8")
                raw_texts.append(text)
                title_file_name = uploaded_file.name.replace('.txt','')
                file_names.append(title_file_name)
            text_data = pd.DataFrame({'title': file_names,
                                      'text': raw_texts})
            st.dataframe(text_data.head())
            st.download_button(
                label="Download data as CSV",
                data=text_data.to_csv().encode('utf-8'),
                file_name='title_text_data.csv',
                mime='title_text/csv',
            )


        with st.spinner('Breaking up text into more reasonable chunks (transformers cannot exceed a 1024 token max)...'):
            # For each body of text, create text chunks of a certain token size required for the transformer
            nested_sentences = md.create_nest_sentences(document = text_input, token_max_length = 1024)
                    # For each chunk of sentences (within the token max)
        text_chunks = []
        for n in range(0, len(nested_sentences)):
            tc = " ".join(map(str, nested_sentences[n]))
            text_chunks.append(tc)

        if gen_keywords == 'Yes':
            st.markdown("### Top Keywords")
            with st.spinner("Generating keywords from text..."):

                kw_df = pd.DataFrame()
                for text_chunk in text_chunks:
                    keywords_list = md.keyword_gen(kw_model, text_chunk)
                    kw_df = kw_df.append(pd.DataFrame(keywords_list))
                kw_df.columns = ['keyword', 'score']
                top_kw_df = kw_df.groupby('keyword')['score'].max().reset_index()

                top_kw_df = top_kw_df.sort_values('score', ascending = False).reset_index().drop(['index'], axis=1)
                st.dataframe(top_kw_df.head(10))
 
        st.markdown("### Summary")
        with st.spinner(f'Generating summaries for {len(text_chunks)} text chunks (this may take a minute)...'):

            my_expander = st.expander(label=f'Expand to see intermediate summary generation details for {len(text_chunks)} text chunks')
            with my_expander:
                summary = []
                
                st.markdown("_Once the original text is broken into smaller chunks (totaling no more than 1024 tokens, \
                    with complete sentences), each block of text is then summarized separately using BART NLI \
                    and then combined at the very end to generate the final summary._")

                for num_chunk, text_chunk in enumerate(text_chunks):
                    st.markdown(f"###### Original Text Chunk {num_chunk+1}/{len(text_chunks)}" )
                    st.markdown(text_chunk)

                    chunk_summary = md.summarizer_gen(summarizer, sequence=text_chunk, maximum_tokens = 300, minimum_tokens = 20)
                    summary.append(chunk_summary) 
                    st.markdown(f"###### Partial Summary {num_chunk+1}/{len(text_chunks)}")
                    st.markdown(chunk_summary)
                    # Combine all the summaries into a list and compress into one document, again
                    final_summary = " \n\n".join(list(summary))

            st.markdown(final_summary)

    if len(text_input) == 0 or len(labels) == 0:
        st.error('Enter some text and at least one possible topic to see label predictions.')
    else:
        st.markdown("### Top Label Predictions on Summary vs Full Text")
        with st.spinner('Matching labels...'):
            topics, scores = md.classifier_zero(classifier, sequence=final_summary, labels=labels, multi_class=True)
            # st.markdown("### Top Label Predictions: Combined Summary")
            # plot_result(topics[::-1][:], scores[::-1][:])
            # st.markdown("### Download Data")
            data = pd.DataFrame({'label': topics, 'scores_from_summary': scores})
            # st.dataframe(data)
            # coded_data = base64.b64encode(data.to_csv(index = False). encode ()).decode()
            # st.markdown(
            #     f'<a href="data:file/csv;base64, {coded_data}" download = "data.csv">Download Data</a>',
            #     unsafe_allow_html = True
            #     )

            topics_ex_text, scores_ex_text = md.classifier_zero(classifier, sequence=text_input, labels=labels, multi_class=True)
            plot_dual_bar_chart(topics, scores, topics_ex_text, scores_ex_text)

            data_ex_text = pd.DataFrame({'label': topics_ex_text, 'scores_from_full_text': scores_ex_text})
            
            data2 = pd.merge(data, data_ex_text, on = ['label'])

            if len(glabels) > 0:
                gdata = pd.DataFrame({'label': glabels})
                gdata['is_true_label'] = int(1)           
            
                data2 = pd.merge(data2, gdata, how = 'left', on = ['label'])
                data2['is_true_label'].fillna(0, inplace = True)

            st.markdown("### Data Table")
            with st.spinner('Generating a table of results and a download link...'):
                st.dataframe(data2)

                @st.cache
                def convert_df(df):
                     # IMPORTANT: Cache the conversion to prevent computation on every rerun
                     return df.to_csv().encode('utf-8')
                csv = convert_df(data2)
                st.download_button(
                     label="Download data as CSV",
                     data=csv,
                     file_name='text_labels.csv',
                     mime='text/csv',
                 )
                # coded_data = base64.b64encode(data2.to_csv(index = False). encode ()).decode()
                # st.markdown(
                #     f'<a href="data:file/csv;base64, {coded_data}" download = "data.csv">Click here to download the data</a>',
                #     unsafe_allow_html = True
                #     )

            if len(glabels) > 0:
                st.markdown("### Evaluation Metrics")
                with st.spinner('Evaluating output against ground truth...'):

                    section_header_description = ['Summary Label Performance', 'Original Full Text Label Performance']
                    data_headers = ['scores_from_summary', 'scores_from_full_text']
                    for i in range(0,2):
                        st.markdown(f"###### {section_header_description[i]}")
                        report = classification_report(y_true = data2[['is_true_label']], 
                            y_pred = (data2[[data_headers[i]]] >= threshold_value) * 1.0,
                            output_dict=True)
                        df_report = pd.DataFrame(report).transpose()
                        st.markdown(f"Threshold set for: {threshold_value}")
                        st.dataframe(df_report)

        st.success('All done!')
        st.balloons()
