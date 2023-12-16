# Streamlit
import streamlit as st 
import os
import pandas as pd
import pickle
import json
# Preprocessing
import re
import py_vncorenlp
from VietnameseTextNormalizer.ReleasePython3 import VietnameseTextNormalizer

# Feature Extraction
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec

# Visualize
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
import plotly.express as px
from sklearn.decomposition import PCA
from PIL import Image

# Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


matplotlib.use('Agg')

# Define variables
ORIGINAL_DATA = "./original_data"
STOPWORD = "./vietnamese-stopwords-dash.txt"
with open(STOPWORD, 'r', encoding='utf-8') as stop_word_file:
    stop_words = stop_word_file.read().splitlines()
PREPROCESSED_DATA = "./preprocessed_data/preprocessed_data.json"
WORD2VEC_MODEL = './models/word2vec_model_100dim_10min.save'
DOC2VEC_MODEL = './models/doc2vec_model_100dim_10min.save'
# Word2Vec
LSTM_WORD2VEC_MODEL = "./models/lstm_word2vec.h5"
BiLSTM_WORD2VEC_MODEL = "./models/bilstm_word2vec.h5"
GRU_WORD2VEC_MODEL = "./models/gru_word2vec.h5"
# Doc2Vec
LSTM_DOC2VEC_MODEL = "./models/lstm_doc2vec.h5"
BiLSTM_DOC2VEC_MODEL = "./models/bilstm_doc2vec.h5"
GRU_DOC2VEC_MODEL = "./models/gru_doc2vec.h5"

# Load dữ liệu đã split train (60), test (20), val(20)
with open('data.pkl', 'rb') as f:
    loaded_data = pickle.load(f)
X_train,y_train, X_test, y_test,X_val, y_val, word_index,tokenizer = loaded_data.values()

@st.cache(allow_output_mutation=True)
def initialize_annotate():
    return py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos", "ner", "parse"], save_dir='./')

# Gọi annotate
annotate = initialize_annotate()

def load_json(file_path):
    
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            df = pd.json_normalize(data)
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# Xem tổng quan dữ liệu ban đầu
def plot_category_count(data_folder):
    # Tạo các biến để lưu số lượng hạng mục và số lượng bài báo
    categories_count = {}
    articles_count = 0

    # Duyệt qua tất cả các tệp tin trong thư mục
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.txt'):
                articles_count += 1  # Tăng số lượng bài báo

                # Lấy đường dẫn đầy đủ đến tệp tin
                file_path = os.path.join(root, file)

                # Lấy tên hạng mục từ đường dẫn
                category = os.path.basename(os.path.dirname(file_path))

                # Tăng số lượng bài báo cho hạng mục tương ứng
                categories_count[category] = categories_count.get(category, 0) + 1

    # Tạo DataFrame từ dữ liệu
    df = pd.DataFrame({'Category': list(categories_count.keys()), 'Count': list(categories_count.values())})

    # Tạo DataFrame mới với tổng số lượng bài báo và thể loại
    summary_df = pd.DataFrame({
        'Total Articles': [articles_count],
        'Total Categories': [len(categories_count)]
    })

    # Vẽ biểu đồ cột dọc với màu viridis
    fig = px.bar(df, x='Category', y='Count', color='Count', color_continuous_scale='Viridis',
                 title='Number of articles by category',
                 labels={'Count': 'Number of articles', 'Category': 'Categories'},
                 text='Count')

    # Hiển thị giá trị trên cột
    fig.update_traces(texttemplate='%{text}', textposition='outside')

    # Tinh chỉnh hiển thị trục x
    fig.update_xaxes(tickmode='array', tickvals=list(range(len(df['Category']))), ticktext=df['Category'])

    # Chỉnh sửa background
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )

    # Hiển thị biểu đồ trong Streamlit
    st.plotly_chart(fig)
    st.dataframe(summary_df)


# Tiền xử lý dữ liệu
def preprocess_text(text):
    # Chuẩn hóa tiếng việt (unicode,...)
    text= VietnameseTextNormalizer.Normalize(text)
    # Loại bỏ liên kết
    text = re.sub(r'http\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\b(www\.[^\s]+|(?!https?://)[^\s]+\.[a-z]{2,})\b', '', text)
    # Xóa số điện thoại
    text = re.sub(r'\b\d{10,}\b', '', text)
    # Xóa emails
    text = re.sub(r'\S+@\S+', '', text)
    # Loại bỏ stopwords và các từ không quan trọng
    annotations = annotate.annotate_text(text)
    all_tokens = []
    for sentence_index, sentence_annotations in annotations.items():
        all_tokens.extend(sentence_annotations)
    word_list  = [
        token['wordForm'] for token in all_tokens
        if isinstance(token, dict) and 'wordForm' in token and isinstance(token['wordForm'], str)
        and 'posTag' in token and isinstance(token['posTag'], str) and token['posTag'] in ['N', 'V', 'A']
        and 'nerLabel' in token and token['nerLabel'] not in ['B-PER']
    ]
    clean_words = [word.strip(',').strip().lower() for word in word_list  if word not in stop_words]
    clean_words = [re.sub(r'([^\s\w]|)+', '', sentence) for sentence in clean_words if sentence!='']
    return clean_words


# Mô hình trích xuất đặc trưng
def display_features_model_info(model):
    # Lấy thông tin từ mô hình 
    vector_size = model.vector_size
    vocab_size = len(model.wv.vocab)

    # Tạo DataFrame
    data = {'Measurements': ['Embedding Dimension', 'Vocabulary Size'],
            'Values': [vector_size, vocab_size]}

    df = pd.DataFrame(data)

    # In ra DataFrame
    st.dataframe(df.head())

def load_and_visualize_similar_words(word, model, num_words, title):
    # Kiểm tra xem từ có trong từ điển của mô hình không
    if word not in model.wv.vocab:
        st.error(f"'{word}' does not exist in the model's dictionary!")
        return None

    # Lấy danh sách từ tương tự
    similar_words = [item[0] for item in model.wv.most_similar(word, topn=num_words)]

    # Tạo DataFrame từ danh sách từ tương tự
    df = pd.DataFrame({'Word': similar_words, 'Similarity': [model.wv.similarity(word, w) for w in similar_words]})

    vectors = [model.wv[word] for word in similar_words]
    # Giảm chiều dữ liệu xuống còn 2 chiều để vẽ biểu đồ
    pca = PCA(n_components=2)
    result = pca.fit_transform(vectors)

    # Tạo một danh sách màu sắc đẹp
    colors = plt.cm.viridis(np.linspace(0, 1, len(similar_words)))

    # Vẽ biểu đồ và thêm chú thích (annotate)
    plt.figure(figsize=(10, 8))
    for i, (word, color) in enumerate(zip(similar_words, colors)):
        plt.scatter(result[i, 0], result[i, 1], color=color, s=100)
        plt.annotate(word, xy=(result[i, 0], result[i, 1]), xytext=(5, -5), textcoords='offset points', fontsize=8)

    # Hiển thị title
    plt.title(title)

    st.pyplot(plt)
    st.subheader("List of words similar")
    st.dataframe(df.head(10))


# Mô hình deep learning

def predict_labels(model, input_padded):
    # Thực hiện dự đoán
    predictions = model.predict(input_padded)

    # Mapping giữa chỉ số và tên nhãn
    label_mapping = {0: 'Chinh tri Xa hoi', 1: 'Cong nghe', 2: 'Doi song', 3: 'Giai tri', 4: 'Giao duc', 5: 'Khoa hoc',
                 6: 'Kinh doanh', 7: 'Phap luat', 8: 'Suc khoe', 9: 'The gioi', 10: 'The thao', 11: 'Van hoa'}
    # Tạo DataFrame để lưu trữ kết quả
    result_df = pd.DataFrame(columns=['Label', 'Confidence'])

    # In ra độ tin cậy của tất cả các nhãn và lưu vào DataFrame
    for i, prediction in enumerate(predictions):
        
        # In ra tên nhãn và độ tin cậy tương ứng
        for index, label_name in label_mapping.items():
            
            confidence_percent = prediction[index] * 100
            result_df = result_df.append({'Label': label_name, 'Confidence': confidence_percent}, ignore_index=True)
    # Lấy chỉ số của dòng có giá trị lớn nhất trong cột 'Phần trăm Độ tin cậy'
    max_confidence_index = result_df['Confidence'].idxmax()

    # Lấy thông tin của dòng có độ tin cậy cao nhất
    max_confidence_row = result_df.loc[max_confidence_index]

    # In ra nhãn có độ tin cậy cao nhất và độ tin cậy tương ứng
    max_confidence_label = max_confidence_row['Label']
    # Định dạng cột 'Phần trăm Độ tin cậy' để thêm đuôi %
    result_df['Confidence'] = result_df['Confidence'].apply(lambda x: f'{x:.2f}%')

    return max_confidence_label, result_df

def evaluate_classification(y_test, y_pred):

    # Chuyển nhãn dự đoán về one-hot-encoding
    y_pred_onehot = pd.get_dummies(y_pred)

    # accuracy
    accuracy = accuracy_score(y_test, y_pred_onehot)

    # precision, recall, and f1-score (weighted average)
    precision = precision_score(y_test, y_pred_onehot, average='weighted')
    recall = recall_score(y_test, y_pred_onehot, average='weighted')
    f1 = f1_score(y_test, y_pred_onehot, average='weighted')

    # Tạo dataframe
    evaluation_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Score': [accuracy, precision, recall, f1]
    })

    return evaluation_df

def evaluate_per_label(y_true, y_pred):
    
    # One-hot encode predicted labels
    y_pred_onehot = pd.get_dummies(y_pred)

    # Calculate precision, recall, and f1-score for each label
    precision_per_label = precision_score(y_true, y_pred_onehot, average=None)
    recall_per_label = recall_score(y_true, y_pred_onehot, average=None)
    f1_per_label = f1_score(y_true, y_pred_onehot, average=None)
    # Create a DataFrame to store the results
    evaluation_df = pd.DataFrame({
        'Label': y_true.columns,
        'Precision': precision_per_label,
        'Recall': recall_per_label,
        'F1-score': f1_per_label
    })

    return evaluation_df

def get_keys(val,my_dict):
	for key,value in my_dict.items():
		if val == value:
			return key



def main():
    
    st.title("News Classifer Deep Learning App")
    # Side bar
    activities = ["Text Preprocessing","Feature Extraction","Train and Evaluate Model","Prediction"]
    choice = st.sidebar.selectbox("Choose Activity",activities)

    # Tiền xử lý dữ liệu
    if choice == "Text Preprocessing":
        st.info("Text Preprocessing")
        preprocessing_task = ["No Options", "Data Overview", "Process Text Demo", "Load Preprocessed Data"]
        task_choice = st.selectbox("Choose Task", preprocessing_task)
        if task_choice == "Data Overview":
            plot_category_count(ORIGINAL_DATA)
        elif task_choice == "Process Text Demo":
            st.markdown("**Preprocessing Steps:**")
            st.markdown("- Use the `VietnameseTextNormalizer` library to standardize Vietnamese words")
            st.markdown("- Utilize techniques such as regular expressions to remove unwanted elements: html, links, emails,...")
            st.markdown("- Use the provided Vietnamese stopwords dictionary (`vietnamese-stopwords-dash.txt`) to remove common stopwords")
            st.markdown("- Employ a POS tagging tool to determine the grammatical category of each word in the sentence and filter out important components")
            st.markdown("- Split the text into a list of individual words")
            
            news_text = st.text_area("Enter Text","Type Here")
            if st.button("Analyze"):
                st.subheader("Original Text")
                st.info(news_text)
                preprocessed_news = preprocess_text(news_text)
                st.subheader("Preprocessed Text")
                st.success(preprocessed_news)
        elif task_choice == "Load Preprocessed Data":
            df = pd.read_json(PREPROCESSED_DATA, encoding='utf-8', lines=True)
            st.dataframe(df.head(20))
    # Trích xuất đặc trưng
    if choice == "Feature Extraction":
        st.info("Feature Extraction")
        feature_extraction_task = ["No Options", "Word2Vec","Doc2Vec"]
        task_choice = st.selectbox("Choose Model",feature_extraction_task)
        if task_choice == "Word2Vec":
            word2vec_model = Word2Vec.load(WORD2VEC_MODEL)
            word2vec_task = ["Overview", "Word Similar"]
            task_choice = st.selectbox("Choose Task",word2vec_task )
            if task_choice == "Overview":
                display_features_model_info(word2vec_model)
            elif task_choice == "Word Similar":
                word = st.text_area("Enter Word","Type Here")
                if st.button("Execute"):
                    load_and_visualize_similar_words(word, word2vec_model, num_words=50, title="Word2Vec Word Similar")
                    st.subheader(f"Vector Dimension of '{word}'")
                    st.table(pd.DataFrame(word2vec_model.wv.get_vector(word), columns=["Values"]))
        if task_choice == "Doc2Vec":
            doc2vec_model = Doc2Vec.load(DOC2VEC_MODEL)
            doc2vec_task = ["Overview", "Word Similar"]
            task_choice = st.selectbox("Choose Task",doc2vec_task )
            if task_choice == "Overview":
                display_features_model_info(doc2vec_model)
            elif task_choice == "Word Similar":
                word = st.text_area("Enter Word","Type Here")
                if st.button("Execute"):
                    load_and_visualize_similar_words(word, doc2vec_model, num_words=50, title="Doc2Vec Word Similar")
                    st.subheader(f"Vector Dimension of '{word}'")
                    st.table(pd.DataFrame(doc2vec_model.wv.get_vector(word), columns=["Values"]))

    if choice == "Prediction":
         st.info("Prediction with Deep Learning")
         news_text = st.text_area("Enter Text", "Type Here")
         all_dl_models = ["No Options","BiLSTM", "LSTM", "GRU"]
         model_choice = st.selectbox("Choose DL Model", all_dl_models)
         feature_extraction_task = ["No Options", "Word2Vec","Doc2Vec"]
         task_choice = st.selectbox("Choose Features Model",feature_extraction_task)
         if st.button("Classify"):
              st.subheader("Original Text")
              st.info(news_text)
              processed_news = preprocess_text(news_text)
              st.subheader("Step 1: Data preprocessing")
              st.success(processed_news)
              # Chuyển đổi văn bản thành chuỗi số
              news_token = tokenizer.texts_to_sequences([processed_news])
              st.subheader("Step 2: Represent each sentence by sequences of numbers")
              st.success(news_token)
              # Padding chuỗi chỉ số 
              news_pad = pad_sequences(news_token, padding='post', truncating='post', maxlen=200)
              st.subheader("Step 3: Padding the sequence")
              st.success(news_pad)
              st.subheader("Step 4: Predict")
              if model_choice == "LSTM":
                if task_choice == "Word2Vec":
                    LSTM_Word2Vec_model = load_model(LSTM_WORD2VEC_MODEL)
                    # Lấy nhãn có xác suất cao nhất
                    predicted_label, confidence = predict_labels(LSTM_Word2Vec_model, news_pad)
                    st.subheader("Confidence Per Label")
                    st.dataframe(confidence)
                    st.subheader("Predicted Label")
                    st.success(predicted_label)
                if task_choice == "Doc2Vec":
                    LSTM_Doc2Vec_model = load_model(LSTM_DOC2VEC_MODEL)
                    # Lấy nhãn có xác suất cao nhất
                    predicted_label, confidence = predict_labels(LSTM_Doc2Vec_model, news_pad)
                    st.subheader("Confidence Per Label")
                    st.dataframe(confidence)
                    st.subheader("Predicted Label")
                    st.success(predicted_label)
              if model_choice == "BiLSTM":
                if task_choice == "Word2Vec":
                    BiLSTM_Word2Vec_model = load_model(BiLSTM_WORD2VEC_MODEL)
                    # Lấy nhãn có xác suất cao nhất
                    predicted_label, confidence = predict_labels(BiLSTM_Word2Vec_model, news_pad)
                    st.subheader("Confidence Per Label")
                    st.dataframe(confidence)
                    st.subheader("Predicted Label")
                    st.success(predicted_label)
                if task_choice == "Doc2Vec":
                    BiLSTM_Doc2Vec_model = load_model(BiLSTM_DOC2VEC_MODEL)
                    # Lấy nhãn có xác suất cao nhất
                    predicted_label, confidence = predict_labels(BiLSTM_Doc2Vec_model, news_pad)
                    st.subheader("Confidence Per Label")
                    st.dataframe(confidence)
                    st.subheader("Predicted Label")
                    st.success(predicted_label)
              if model_choice == "GRU":
                if task_choice == "Word2Vec":
                    GRU_Word2Vec_model = load_model(GRU_WORD2VEC_MODEL)
                    # Lấy nhãn có xác suất cao nhất
                    predicted_label, confidence = predict_labels(GRU_Word2Vec_model, news_pad)
                    st.subheader("Confidence Per Label")
                    st.dataframe(confidence)
                    st.subheader("Predicted Label")
                    st.success(predicted_label)
                if task_choice == "Doc2Vec":
                    GRU_Doc2Vec_model = load_model(GRU_DOC2VEC_MODEL)
                    # Lấy nhãn có xác suất cao nhất
                    predicted_label, confidence = predict_labels(GRU_Doc2Vec_model, news_pad)
                    st.subheader("Confidence Per Label")
                    st.dataframe(confidence)
                    st.subheader("Predicted Label")
                    st.success(predicted_label)
    if choice == "Train and Evaluate Model":
         st.info("Train and Evaluate Model")
         training_task = ["No Options", "Overview", "Tuning Hyperparameters with Optuna", "Evaluate Model"]
         training_choice = st.selectbox("Choose Task", training_task)
         if training_choice == "Overview":
             st.subheader("Model Training Results")
             model_results = pd.read_csv("./trials_result/model_results.csv")
             st.dataframe(model_results)
             
         if training_choice == "Tuning Hyperparameters with Optuna":
             dl_model = ["No Options", "LSTM", "BiLSTM", "GRU"]
             model_choice = st.selectbox("Choose DL Model", dl_model)
             feature_extraction_task = ["No Options", "Word2Vec","Doc2Vec"]
             task_choice = st.selectbox("Choose Features Model",feature_extraction_task)
             if st.button("Show Result"):
                if model_choice == "LSTM":
                    if task_choice == "Word2Vec":
                        lstm_word2vec_trials = pd.read_csv("./trials_result/study_lstm_word2vec_trials.csv")
                        lstm_word2vec_best_param = load_json("./hyperparameters/LSTM_Word2Vec.json")
                        st.subheader("Number of Completed Trials of 50 trials")
                        st.dataframe(lstm_word2vec_trials)
                        st.subheader("Best Hyperparamters")
                        st.dataframe(lstm_word2vec_best_param)
                        lstm_word2vec_optimize_history = Image.open("./images/study_lstm_word2vec_optimize_history.png")
                        st.image(lstm_word2vec_optimize_history, caption="Optimize History Plot", use_column_width=True)

                    if task_choice == "Doc2Vec":
                        lstm_doc2vec_trials = pd.read_csv("./trials_result/study_lstm_doc2vec_trials.csv")
                        lstm_doc2vec_best_param = load_json("./hyperparameters/LSTM_Doc2Vec.json")
                        st.subheader("Number of Completed Trials of 50 trials")
                        st.dataframe(lstm_doc2vec_trials)
                        st.subheader("Best Hyperparamters")
                        st.dataframe(lstm_doc2vec_best_param)
                        
                        
                        study_lstm_doc2vec_optimize_history = open('./images/study_lstm_doc2vec_optimize_history.html','r')

                        study_lstm_doc2vec_optimize_history = study_lstm_doc2vec_optimize_history.read()

                        st.components.v1.html(study_lstm_doc2vec_optimize_history,height=600, scrolling=True)
                if model_choice == "BiLSTM":
                    if task_choice == "Word2Vec":
                        bilstm_word2vec_trials = pd.read_csv("./trials_result/study_bilstm_word2vec_trials.csv")
                        bilstm_word2vec_best_param = load_json("./hyperparameters/BiLSTM_Word2Vec.json")
                        st.subheader("Number of Completed Trials of 50 trials")
                        st.dataframe(bilstm_word2vec_trials)
                        st.subheader("Best Hyperparamters")
                        st.dataframe(bilstm_word2vec_best_param)
                        
                        bilstm_word2vec_optimize_history = Image.open("./images/study_bilstm_word2vec_optimize_history.png")
                        st.image(bilstm_word2vec_optimize_history, caption="Optimize History Plot", use_column_width=True)

                    if task_choice == "Doc2Vec":
                        bilstm_doc2vec_trials = pd.read_csv("./trials_result/study_bilstm_doc2vec_trials.csv")
                        bilstm_doc2vec_best_param = load_json("./hyperparameters/BiLSTM_Doc2Vec.json")
                        st.subheader("Number of Completed Trials of 50 trials")
                        st.dataframe(bilstm_doc2vec_trials)
                        st.subheader("Best Hyperparamters")
                        st.dataframe(bilstm_doc2vec_best_param)
                        
                        study_bilstm_doc2vec_optimize_history = open('./images/study_bilstm_doc2vec_optimize_history.html','r')

                        study_bilstm_doc2vec_optimize_history = study_bilstm_doc2vec_optimize_history.read()

                        st.components.v1.html(study_bilstm_doc2vec_optimize_history,height=600, scrolling=True)
                if model_choice == "GRU":
                    if task_choice == "Word2Vec":
                        gru_word2vec_trials = pd.read_csv("./trials_result/study_gru_word2vec_trials.csv")
                        gru_word2vec_best_param = load_json("./hyperparameters/GRU_Word2Vec.json")
                        st.subheader("Number of Completed Trials of 50 trials")
                        st.dataframe(gru_word2vec_trials)
                        st.subheader("Best Hyperparamters")
                        st.dataframe(gru_word2vec_best_param)
                        
                        study_gru_word2vec_optimize_history = open('./images/study_gru_word2vec_optimize_history.html','r')

                        study_gru_word2vec_optimize_history = study_gru_word2vec_optimize_history.read()

                        st.components.v1.html(study_gru_word2vec_optimize_history,height=600, scrolling=True)

                    if task_choice == "Doc2Vec":
                        gru_doc2vec_trials = pd.read_csv("./trials_result/study_gru_doc2vec_trials.csv")
                        gru_doc2vec_best_param = load_json("./hyperparameters/GRU_Doc2Vec.json")
                        st.subheader("Number of Completed Trials of 50 trials")
                        st.dataframe(gru_doc2vec_trials)
                        st.subheader("Best Hyperparamters")
                        st.dataframe(gru_doc2vec_best_param)
                        
                        
                        study_gru_doc2vec_optimize_history = open('./images/study_gru_doc2vec_optimize_history.html','r')

                        study_gru_doc2vec_optimize_history = study_gru_doc2vec_optimize_history.read()

                        st.components.v1.html(study_gru_doc2vec_optimize_history,height=600, scrolling=True)
         if training_choice == "Evaluate Model":
             dl_model = ["No Options", "LSTM", "BiLSTM", "GRU"]
             model_choice = st.selectbox("Choose DL Model", dl_model)
             feature_extraction_task = ["No Options","Word2Vec","Doc2Vec"]
             task_choice = st.selectbox("Choose Features Model",feature_extraction_task)
             if st.button("Show Result"):
                if model_choice == "LSTM":
                    if task_choice == "Word2Vec":
                        
                        model = load_model(LSTM_WORD2VEC_MODEL)
                        y_pred=model.predict_classes(X_test)
                        st.subheader("Overall")
                        evaluation_result = evaluate_classification(y_test, y_pred)
                        st.dataframe(evaluation_result)
                        st.subheader("Evaluate per label")
                        per_label = evaluate_per_label(y_test, y_pred)
                        st.dataframe(per_label)
                        
                        
                    if task_choice == "Doc2Vec":
                        
                        model = load_model(LSTM_DOC2VEC_MODEL)
                        y_pred=model.predict_classes(X_test)
                        st.subheader("Overall")
                        evaluation_result = evaluate_classification(y_test, y_pred)
                        st.dataframe(evaluation_result)
                        st.subheader("Evaluate per label")
                        per_label = evaluate_per_label(y_test, y_pred)
                        st.dataframe(per_label)
                if model_choice == "BiLSTM":
                    if task_choice == "Word2Vec":
                        
                        model = load_model(BiLSTM_WORD2VEC_MODEL)
                        y_pred=model.predict_classes(X_test)
                        st.subheader("Overall")
                        evaluation_result = evaluate_classification(y_test, y_pred)
                        st.dataframe(evaluation_result)
                        st.subheader("Evaluate per label")
                        per_label = evaluate_per_label(y_test, y_pred)
                        st.dataframe(per_label)

                    if task_choice == "Doc2Vec":
                        
                        model = load_model(BiLSTM_DOC2VEC_MODEL)
                        y_pred=model.predict_classes(X_test)
                        st.subheader("Overall")
                        evaluation_result = evaluate_classification(y_test, y_pred)
                        st.dataframe(evaluation_result)
                        st.subheader("Evaluate per label")
                        per_label = evaluate_per_label(y_test, y_pred)
                        st.dataframe(per_label)
                if model_choice == "GRU":
                    if task_choice == "Word2Vec":
                        
                        model = load_model(GRU_WORD2VEC_MODEL)
                        y_pred=model.predict_classes(X_test)
                        st.subheader("Overall")
                        evaluation_result = evaluate_classification(y_test, y_pred)
                        st.dataframe(evaluation_result)
                        st.subheader("Evaluate per label")
                        per_label = evaluate_per_label(y_test, y_pred)
                        st.dataframe(per_label)

                    if task_choice == "Doc2Vec":
                        
                        model = load_model(GRU_DOC2VEC_MODEL)
                        y_pred=model.predict_classes(X_test)
                        st.subheader("Overall")
                        evaluation_result = evaluate_classification(y_test, y_pred)
                        st.dataframe(evaluation_result)
                        st.subheader("Evaluate per label")
                        per_label = evaluate_per_label(y_test, y_pred)
                        st.dataframe(per_label)

if __name__ == '__main__':
	main()