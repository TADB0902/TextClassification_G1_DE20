import regex as re
from underthesea import word_tokenize
import streamlit as st
import os
from keras.preprocessing.text import Tokenizer
from os import listdir
import pandas as pd
import pickle
from keras.preprocessing.sequence import pad_sequences

sep = os.sep
 
uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"
 
def loaddicchar():
    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic
dicchar = loaddicchar()

# Hàm chuyển Unicode dựng sẵn về Unicde tổ hợp (phổ biến hơn)
def convert_unicode(txt):
    return re.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

bang_nguyen_am = [['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
                  ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
                  ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
                  ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
                  ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
                  ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
                  ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
                  ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
                  ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
                  ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
                  ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
                  ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']]
bang_ky_tu_dau = ['', 'f', 's', 'r', 'x', 'j']

nguyen_am_to_ids = {}

for i in range(len(bang_nguyen_am)):
    for j in range(len(bang_nguyen_am[i]) - 1):
        nguyen_am_to_ids[bang_nguyen_am[i][j]] = (i, j)

def chuan_hoa_dau_tu_tieng_viet(word):
    if not is_valid_vietnam_word(word):
        return word

    chars = list(word)
    dau_cau = 0
    nguyen_am_index = []
    qu_or_gi = False
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x == -1:
            continue
        elif x == 9:  # check qu
            if index != 0 and chars[index - 1] == 'q':
                chars[index] = 'u'
                qu_or_gi = True
        elif x == 5:  # check gi
            if index != 0 and chars[index - 1] == 'g':
                chars[index] = 'i'
                qu_or_gi = True
        if y != 0:
            dau_cau = y
            chars[index] = bang_nguyen_am[x][0]
        if not qu_or_gi or index != 1:
            nguyen_am_index.append(index)

    if len(nguyen_am_index) < 2:
        if qu_or_gi:
            if len(chars) == 2:
                x, y = nguyen_am_to_ids.get(chars[1])
                chars[1] = bang_nguyen_am[x][dau_cau]
            else:
                x, y = nguyen_am_to_ids.get(chars[2], (-1, -1))
                if x != -1:
                    chars[2] = bang_nguyen_am[x][dau_cau]
                else:
                    chars[1] = bang_nguyen_am[5][dau_cau] if chars[1] == 'i' else bang_nguyen_am[9][dau_cau]
            return ''.join(chars)
        return word

    for index in nguyen_am_index:
        x, y = nguyen_am_to_ids[chars[index]]
        if x == 4 or x == 8:  # ê, ơ
            chars[index] = bang_nguyen_am[x][dau_cau]
            return ''.join(chars)

    if len(nguyen_am_index) == 2:
        if nguyen_am_index[-1] == len(chars) - 1:
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            chars[nguyen_am_index[0]] = bang_nguyen_am[x][dau_cau]
        else:
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
    else:
        x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
        chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
    return ''.join(chars)


def is_valid_vietnam_word(word):
    chars = list(word)
    nguyen_am_index = -1
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x != -1:
            if nguyen_am_index == -1:
                nguyen_am_index = index
            else:
                if index - nguyen_am_index != 1:
                    return False
                nguyen_am_index = index
    return True


def chuan_hoa_dau_cau_tieng_viet(sentence):
    """
        Chuyển câu tiếng việt về chuẩn gõ dấu kiểu cũ.
        :param sentence:
        :return:
        """
    sentence = sentence.lower()
    words = sentence.split()
    for index, word in enumerate(words):
        cw = re.sub(r'(^\p{P}*)([p{L}.]*\p{L}+)(\p{P}*$)', r'\1/\2/\3', word).split('/')
        # print(cw)
        if len(cw) == 3:
            cw[1] = chuan_hoa_dau_tu_tieng_viet(cw[1])
        words[index] = ''.join(cw)
    return ' '.join(words)

def remove_html(txt):
    return re.sub(r'<[^>]*>', '', txt)



def text_preprocess(document):
    # xóa html code
    document = remove_html(document)
    # chuẩn hóa unicode
    document = convert_unicode(document)
    # chuẩn hóa cách gõ dấu tiếng Việt
    document = chuan_hoa_dau_cau_tieng_viet(document)
    # tách từ
    document = word_tokenize(document, format="text")
    # đưa về lower
    document = document.lower()
    # xóa các ký tự không cần thiết
    document = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',document)
    # tách thành danh sách các từ
    word_list = document.split()
    #xóa khoảng trắng thừa
    word_list = [word.strip() for word in word_list]
    return word_list

def tokenizer(texts):
    tokenizer = Tokenizer()
    # fit the tokenizer on our text
    tokenizer.fit_on_texts(texts)

    # get all words that the tokenizer knows
    word_index = tokenizer.word_index
    return tokenizer, word_index

 # Hàm để đọc và hiển thị nội dung của tệp tin văn bản
def show_text_file_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content  

# Hàm load data
def load_data(data_folder):

    texts = []
    labels = []
    #
    for folder in listdir(data_folder):
        #
        if folder != ".DS_Store":
            print("Load cat: ",folder)
            for file in listdir(data_folder + sep + folder):
                #
                if file!=".DS_Store":
                    print("Load file: ", file)
                    with open(data_folder + sep + folder + sep +  file, 'r', encoding="utf-8") as f:
                        all_of_it = f.read()
                        sentences  = all_of_it.split('.')

                        # Remove garbage
                        sentences = [text_preprocess(sentence) for sentence in sentences ]

                        texts = texts + sentences
                        label = [folder for _ in sentences]
                        labels = labels + label
                        del all_of_it, sentences


    return texts, labels



def preprocess_page():
    input_file = "./data"
    st.header("Tiền Xử Lý Và Load Dữ Liệu")
    # Danh sách các bước tiền xử lý
    preprocessing_steps = [
        "Xóa HTML code (nếu có)",
        "Chuẩn hóa bảng mã Unicode",
        "Chuẩn hóa kiểu gõ dấu tiếng Việt",
        "Thực hiện tách từ tiếng Việt (sử dụng thư viện tách từ underthesea)",
        "Đưa về văn bản lower (viết thường)",
        "Xóa các ký tự đặc biệt: \".\", \",\", \";\", \")\", ...",
        "Tokenizer",
    ]

    # Hiển thị danh sách các bước tiền xử lý
    st.text("Các bước tiền xử lý:")
    st.markdown("\n".join([f"- {step}" for step in preprocessing_steps]))
    st.header("1. Tiền xử lý cho 1 tệp TXT")

    # Option 1: Load một tệp TXT từ bên ngoài và xuất ra kết quả
    external_content = st.file_uploader("Chọn một tệp TXT từ bên ngoài:", type=["txt"])
    execute_button_external = st.button("Thực thi xử lý")

    if external_content and execute_button_external:
        st.subheader("Nội dung của tệp trước tiền xử lý:")
        st.text(external_content.getvalue().decode("utf-8"))

        sentences = external_content.getvalue().decode("utf-8").split(".")
        # Tiền xử lý dữ liệu
        processed_external_content = [text_preprocess(sentence) for sentence in sentences]

        st.subheader("Nội dung của tệp sau tiền xử lý:")
        st.text(processed_external_content)

    # Option 2: Tiền xử lý cho cả tập dữ liệu
    st.header("2. Tiền xử lý cho cả tập dữ liệu")

    # Kiểm tra xem thư mục nguồn có tồn tại không
    if not os.path.exists(input_file):
        st.warning(f"Thư mục nguồn '{input_file}' không tồn tại. Hãy chia dữ liệu trước khi tiền xử lý.")
        return

    # Nút "Thực thi tiền xử lý cho cả tập dữ liệu"
    execute_button_all_files = st.button("Thực thi tiền xử lý cho cả tập dữ liệu")

    if execute_button_all_files:
        if not os.path.exists(input_file + sep + "data.pkl"):
            st.warning("Chưa có dữ liệu. Tiến hành xây dựng dữ liệu...")

            texts, labels = load_data(input_file)
            my_tokenizer, word_index = tokenizer(texts)

            # put the tokens in a matrix
            X = my_tokenizer.texts_to_sequences(texts)
            X = pad_sequences(X)

            # prepare the labels
            y = pd.get_dummies(labels)
            with open(input_file + sep + "data.pkl", 'wb') as f:
                pickle.dump([X,y, texts], f)
                

            # Hiển thị thông báo khi xây dựng dữ liệu
            st.success("Xây dựng dữ liệu thành công!")
            #sys.exit()
        else:
            st.info("Dữ liệu đã được xây dựng. Tiến hành load...")
            #
            with open(input_file + sep + "data.pkl", 'rb') as f:
                X,y,texts = pickle.load(f)

            # Hiển thị thông báo khi load dữ liệu
            st.success("Load dữ liệu thành công!")

        
    st.header("3. Xem Dữ Liệu")

    # Kiểm tra xem file data.pkl đã tồn tại hay chưa
    if not os.path.exists(input_file + sep + "data.pkl"):
        st.warning("Chưa có dữ liệu. Load dữ liệu trước khi xem.")
    else:
        st.text("Thông tin sau khi tải dữ liệu raw")

        # Hiển thị kích thước của X
        st.write("Kích thước của X:", X.shape)

        # Hiển thị các phần tử của X từ index 10 đến 29
        st.write("Các phần tử của X từ index 10 đến 29:", X[10:30])

        # Hiển thị các giá trị của y từ index 10 đến 29
        st.write("Các giá trị của y từ index 10 đến 29:", y[10:30])

        # Hiển thị các đoạn văn bản từ index 10 đến 29
        st.write("Các đoạn văn bản từ index 10 đến 29:", texts[10:30])