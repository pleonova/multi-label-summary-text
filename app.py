
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
st.write("This app summarizes and then classifies your long text(s) with multiple labels using [BART Large MNLI](https://huggingface.co/facebook/bart-large-mnli). The keywords are generated using [KeyBERT](https://github.com/MaartenGr/KeyBERT).")
st.write("__Inputs__: User enters their own custom text(s) and labels.")
st.write("__Outputs__: A summary of the text, likelihood match score for each label and a downloadable csv of the results. \
    Includes additional options to generate a list of keywords and/or evaluate results against a list of ground truth labels, if available.")

example_button = st.button(label='See Example')
if example_button:
    example_text = ex_long_text #ex_text
    display_text = 'Excerpt from Frankenstein:' + example_text + '"\n\n' + "[This is an excerpt from Project Gutenberg's Frankenstein. " + ex_license + "]"
    input_labels = ex_labels
    input_glabels = ex_glabels
    title_name = 'Frankenstein, Chapter 3'
else:
    display_text = ''
    input_labels = ''
    input_glabels = ''
    title_name = 'Submitted Text'


with st.form(key='my_form'):
    st.markdown("##### Step 1: Upload Text")
    text_input = st.text_area("Input any text you want to summarize & classify here (keep in mind very long text will take a while to process):", display_text)

    text_csv_expander = st.expander(label=f'Want to upload multiple texts at once? Expand to upload your text files below.', expanded=False)
    with text_csv_expander:
        st.markdown('##### Choose one of the options below:')
        st.write("__Option A:__")
        uploaded_text_files = st.file_uploader(label="Upload file(s) that end with the .txt suffix",
                                              accept_multiple_files=True, key = 'text_uploader',
                                              type='txt')
        st.write("__Option B:__")
        uploaded_csv_text_files = st.file_uploader(label='Upload a CSV file with two columns: "title" and "text"',
                                                   accept_multiple_files=False, key = 'csv_text_uploader',
                                                   type='csv')

    if text_input == display_text and display_text != '':
        text_input = example_text

    gen_keywords = st.radio(
        "Generate keywords from text? (independent from the input labels below)",
        ('Yes', 'No')
        )

    gen_summary = st.radio(
        "Generate summary from text? (recommended for label matching below, but will take longer)",
        ('Yes', 'No')
        )

    st.text("\n\n\n")
    st.markdown("##### Step 2: Enter Labels")
    labels = st.text_input('Enter possible topic labels, which can be either keywords and/or general themes (comma-separated):',input_labels, max_chars=2000)
    labels = list(set([x.strip() for x in labels.strip().split(',') if len(x.strip()) > 0]))

    labels_csv_expander = st.expander(label=f'Prefer to upload a list of labels instead? Click here to upload your CSV file.',expanded=False)
    with labels_csv_expander:
        uploaded_labels_file = st.file_uploader("Choose a CSV file with one column and no header, where each cell is a separate label",
                                                key='labels_uploader')

    st.text("\n\n\n")
    st.markdown("##### Step 3: Provide Ground Truth Labels (_Optional_)")
    glabels = st.text_input('If available, enter ground truth topic labels to evaluate results, otherwise leave blank (comma-separated):',input_glabels, max_chars=2000)
    glabels = list(set([x.strip() for x in glabels.strip().split(',') if len(x.strip()) > 0]))


    glabels_csv_expander = st.expander(label=f'Have a file with labels for the text? Click here to upload your CSV file.', expanded=False)
    with glabels_csv_expander:
        st.markdown('##### Choose one of the options below:')
        st.write("__Option A:__")
        uploaded_onetext_glabels_file = st.file_uploader("Single Text: Choose a CSV file with one column and no header, where each cell is a separate label",
                                                         key = 'onetext_glabels_uploader')
        st.write("__Option B:__")
        uploaded_multitext_glabels_file = st.file_uploader('Multiple Text: Choose a CSV file with two columns "title" and "label", with the cells in the title column matching the name of the files uploaded in step #1.',
                                                           key = 'multitext_glabels_uploader')



    # threshold_value = st.slider(
    #      'Select a threshold cutoff for matching percentage (used for ground truth label evaluation)',
    #      0.0, 1.0, (0.5))

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

    st.spinner(f'Time taken to load various models: {k_time}s for KeyBERT model & {s_time}s for BART summarizer mnli model & {c_time}s for BART classifier mnli model.')
    # st.success(None)

if submit_button or example_button:
    if len(text_input) == 0 and uploaded_text_files is None and uploaded_csv_text_files is None:
        st.error("Enter some text to generate a summary")
    else:

        if len(text_input) != 0:
            text_df = pd.DataFrame.from_dict({'title': [title_name], 'text': [text_input]})

        # OPTION A
        elif uploaded_text_files is not None:
            st.markdown("### Text Inputs")
            st.write('Files concatenated into a dataframe:')
            file_names = []
            raw_texts = []
            for uploaded_file in uploaded_text_files:
                text = str(uploaded_file.read(), "utf-8")
                raw_texts.append(text)
                title_file_name = uploaded_file.name.replace('.txt','')
                file_names.append(title_file_name)
            text_df = pd.DataFrame({'title': file_names,
                                      'text': raw_texts})
            st.dataframe(text_df.head())
            st.download_button(
                label="Download data as CSV",
                data=text_df.to_csv().encode('utf-8'),
                file_name='title_text.csv',
                mime='title_text/csv',
            )
        # OPTION B
        elif uploaded_csv_text_files is not None:
            text_df = pd.read_csv(uploaded_csv_text_files)

        # Which input was used? If text area was used, ignore the 'title'
        if len(text_input) != 0:
            title_element = []
        else:
            title_element = ['title']

        with st.spinner('Breaking up text into more reasonable chunks (transformers cannot exceed a 1024 token max)...'):
            # For each body of text, create text chunks of a certain token size required for the transformer

            text_chunks_lib = dict()
            for i in range(0, len(text_df)):
                nested_sentences = md.create_nest_sentences(document=text_df['text'][i], token_max_length=1024)

                # For each chunk of sentences (within the token max)
                text_chunks = []
                for n in range(0, len(nested_sentences)):
                    tc = " ".join(map(str, nested_sentences[n]))
                    text_chunks.append(tc)
                title_entry = text_df['title'][i]
                text_chunks_lib[title_entry] = text_chunks

    if gen_keywords == 'Yes':
        st.markdown("### Top Keywords")
        with st.spinner("Generating keywords from text..."):

            kw_dict = dict()
            text_chunk_counter = 0
            for key in text_chunks_lib:
                keywords_list = []
                for text_chunk in text_chunks_lib[key]:
                    text_chunk_counter += 1
                    keywords_list += md.keyword_gen(kw_model, text_chunk)
                    kw_dict[key] = dict(keywords_list)
            # Display as a dataframe
            kw_df0 = pd.DataFrame.from_dict(kw_dict).reset_index()
            kw_df0.rename(columns={'index': 'keyword'}, inplace=True)
            kw_df = pd.melt(kw_df0, id_vars=['keyword'], var_name='title', value_name='score').dropna()

            kw_column_list = ['keyword', 'score']
            kw_df = kw_df[kw_df['score'] > 0.25][title_element + kw_column_list].sort_values(title_element + ['score'], ascending=False).reset_index().drop(columns='index')

            st.dataframe(kw_df)
            st.download_button(
                label="Download data as CSV",
                data=kw_df.to_csv().encode('utf-8'),
                file_name='title_keywords.csv',
                mime='title_keywords/csv',
            )

 

    if gen_summary == 'Yes':
        st.markdown("### Summary")
        with st.spinner(f'Generating summaries for {len(text_df)} texts consisting of a total of {text_chunk_counter} chunks (this may take a minute)...'):
            sum_dict = dict()
            for i, key in enumerate(text_chunks_lib):
                with st.expander(label=f'({i+1}/{len(text_df)}) Expand to see intermediate summary generation details for: {key}', expanded=False):
                    # for key in text_chunks_lib:
                    summary = []
                    for num_chunk, text_chunk in enumerate(text_chunks_lib[key]):
                        chunk_summary = md.summarizer_gen(summarizer, sequence=text_chunk, maximum_tokens=300, minimum_tokens=20)
                        summary.append(chunk_summary)

                        st.markdown(f"###### Original Text Chunk {num_chunk+1}/{len(text_chunks)}" )
                        st.markdown(text_chunk)
                        st.markdown(f"###### Partial Summary {num_chunk+1}/{len(text_chunks)}")
                        st.markdown(chunk_summary)

                        # Combine all the summaries into a list and compress into one document, again
                        final_summary = "\n\n".join(list(summary))
                        sum_dict[key] = [final_summary]

            sum_df = pd.DataFrame.from_dict(sum_dict).T.reset_index()
            sum_df.columns = ['title', 'summary_text']
            # TO DO: Make sure summary_text does not exceed the token length

        st.dataframe(sum_df)
        st.download_button(
            label="Download data as CSV",
            data=sum_df.to_csv().encode('utf-8'),
            file_name='title_summary.csv',
            mime='title_summary/csv',
    )

    if ((len(text_input) == 0 and uploaded_text_files is None and uploaded_csv_text_files is None)
            or (len(labels) == 0 and uploaded_labels_file is None)):
        st.error('Enter some text and at least one possible topic to see label predictions.')
    else:
        if gen_summary == 'Yes':
            st.markdown("### Top Label Predictions on Summary vs Full Text")
        else:
            st.markdown("### Top Label Predictions on Full Text")

        if uploaded_labels_file is not None:
            labels_df = pd.read_csv(uploaded_labels_file, header=None)
            label_list = labels_df.iloc[:, 0]
        else:
            label_list = labels

        with st.spinner('Matching labels...(may take some time)'):
            if gen_summary == 'Yes':
                labels_sum_col_list = ['title', 'label', 'scores_from_summary']
                labels_sum_df = pd.DataFrame(columns=labels_sum_col_list)

            labels_full_col_list = ['title', 'label', 'scores_from_full_text']
            labels_full_df = pd.DataFrame(columns=labels_full_col_list)

            for i in range(0, len(text_df)):
                if gen_summary == 'Yes':
                    s_topics, s_scores = md.classifier_zero(classifier, sequence=sum_df['summary_text'][i], labels=label_list, multi_class=True)
                    ls_df = pd.DataFrame({'label': s_topics, 'scores_from_summary': s_scores})
                    ls_df['title'] = text_df['title'][i]
                    labels_sum_df = pd.concat([labels_sum_df, ls_df[labels_sum_col_list]])

                f_topics, f_scores = md.classifier_zero(classifier, sequence=text_df['text'][i], labels=label_list, multi_class=True)
                lf_df = pd.DataFrame({'label': f_topics, 'scores_from_full_text': f_scores})
                lf_df['title'] = text_df['title'][i]
                labels_full_df = pd.concat([labels_full_df, lf_df[labels_full_col_list]])

                with st.expander(f'({i+1}/{len(text_df)}) See intermediate label matching results for: {text_df["title"][i]}'):
                    if gen_summary == 'Yes':
                        st.dataframe(pd.merge(ls_df, lf_df, on=['title','label']))
                    else:
                        st.dataframe(lf_df)

            if gen_summary == 'Yes':
                label_match_df = pd.merge(labels_sum_df, labels_full_df, on=['title', 'label'])
            else:
                label_match_df = labels_full_df.copy()

            if len(glabels) > 0:
                gdata = pd.DataFrame({'label': glabels})
                join_list = ['label']
            elif uploaded_onetext_glabels_file is not None:
                gdata = pd.read_csv(uploaded_onetext_glabels_file, header=None)
                join_list = ['label']
                gdata.columns = join_list
            elif uploaded_multitext_glabels_file is not None:
                gdata = pd.read_csv(uploaded_multitext_glabels_file)
                join_list = ['title', 'label']
                gdata.columns = join_list

            if len(glabels) > 0 or uploaded_onetext_glabels_file is not None or uploaded_multitext_glabels_file is not None:
                gdata['correct_match'] = True
                label_match_df = pd.merge(label_match_df, gdata, how='left', on=join_list)
                label_match_df['correct_match'].fillna(False, inplace=True)

            st.dataframe(label_match_df) #.sort_values(['title', 'label'], ascending=[False, False]))
            st.download_button(
                label="Download data as CSV",
                data=label_match_df.to_csv().encode('utf-8'),
                file_name='title_label_sum_full.csv',
                mime='title_label_sum_full/csv',
            )

            # if len(glabels) > 0:
            #     st.markdown("### Evaluation Metrics")
            #     with st.spinner('Evaluating output against ground truth...'):
            #
            #         section_header_description = ['Summary Label Performance', 'Original Full Text Label Performance']
            #         data_headers = ['scores_from_summary', 'scores_from_full_text']
            #         for i in range(0,2):
            #             st.markdown(f"###### {section_header_description[i]}")
            #             report = classification_report(y_true = data2[['is_true_label']],
            #                 y_pred = (data2[[data_headers[i]]] >= threshold_value) * 1.0,
            #                 output_dict=True)
            #             df_report = pd.DataFrame(report).transpose()
            #             st.markdown(f"Threshold set for: {threshold_value}")
            #             st.dataframe(df_report)

        st.success('All done!')
        st.balloons()
