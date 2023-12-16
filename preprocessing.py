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
 
def load_dictionary_characters():
    """
    Tạo và trả về một từ điển ánh xạ giữa kí tự tiếng Việt có dấu trong bảng mã Latin-1 và Unicode.

    Returns:
    dic (dict): Từ điển ánh xạ giữa kí tự tiếng Việt có dấu (Latin-1) và Unicode.
    """
    dic = {}
    char_1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split('|')
    char_utf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split('|')
    for i in range(len(char_1252)):
        dic[char_1252[i]] = char_utf8[i]
    return dic

dic_char = load_dictionary_characters()

# Hàm chuyển Unicode dựng sẵn về Unicde tổ hợp (phổ biến hơn)
def convert_unicode(txt):
    """
    Chuyển đổi các kí tự tiếng Việt có dấu từ Unicode đã được dựng sẵn về Unicode tổ hợp.
    
    Args:
    txt (str): Chuỗi cần chuyển đổi.
    
    Returns:
    str: Chuỗi đã chuyển đổi.
    """
    # Sử dụng re.sub để thực hiện thay thế các chuỗi kí tự theo từ điển ánh xạ dic_char
    return re.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dic_char[x.group()], txt)

# Tạo một từ điển ánh xạ giữa kí tự nguyên âm và mã nguyên âm trong bảng chữ cái tiếng Việt
vowel_mapping = [['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
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
# Tạo một từ điển ánh xạ giữa kí tự nguyên âm và chỉ số trong bảng nguyên âm
vowel_diacritic_mapping = {}

for i in range(len(vowel_mapping)):
    for j in range(len(vowel_mapping[i]) - 1):
        vowel_diacritic_mapping[vowel_mapping[i][j]] = (i, j)

def normalize_vietnamese_diacritics(word):
    """
    Chuẩn hóa dấu thanh (dấu mũ) của từ tiếng Việt.

    Args:
    word (str): Từ cần chuẩn hóa.

    Returns:
    str: Từ đã được chuẩn hóa.
    """
    if not is_valid_vietnamese_word(word):
        return word

    chars = list(word)
    diacritic_mark = 0
    vowel_index = []
    qu_or_gi = False
    for index, char in enumerate(chars):
        x, y = vowel_diacritic_mapping.get(char, (-1, -1))
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
            diacritic_mark = y
            chars[index] = vowel_mapping[x][0]
        if not qu_or_gi or index != 1:
            vowel_index.append(index)

    # Xử lý các trường hợp đặc biệt với "qu" và "gi"
    if len(vowel_index) < 2:
        if qu_or_gi:
            if len(chars) == 2:
                x, y = vowel_diacritic_mapping.get(chars[1])
                chars[1] = vowel_mapping[x][diacritic_mark]
            else:
                x, y = vowel_diacritic_mapping.get(chars[2], (-1, -1))
                if x != -1:
                    chars[2] = vowel_mapping[x][diacritic_mark]
                else:
                    chars[1] = vowel_mapping[5][diacritic_mark] if chars[1] == 'i' else vowel_mapping[9][diacritic_mark]
            return ''.join(chars)
        return word
     # Xử lý trường hợp có nhiều hơn một nguyên âm
    for index in vowel_index:
        x, y = vowel_diacritic_mapping[chars[index]]
        if x == 4 or x == 8:  # ê, ơ
            chars[index] = vowel_mapping[x][diacritic_mark]
            return ''.join(chars)

    if len(vowel_index) == 2:
        if vowel_index[-1] == len(chars) - 1:
            x, y = vowel_diacritic_mapping[chars[vowel_index[0]]]
            chars[vowel_index[0]] = vowel_mapping[x][diacritic_mark]
        else:
            x, y = vowel_diacritic_mapping[chars[vowel_index[1]]]
            chars[vowel_index[1]] = vowel_mapping[x][diacritic_mark]
    else:
        x, y = vowel_diacritic_mapping[chars[vowel_index[1]]]
        chars[vowel_index[1]] = vowel_mapping[x][diacritic_mark]
    return ''.join(chars)


def is_valid_vietnamese_word(word):
    """
    Kiểm tra xem một từ tiếng Việt có hợp lệ không.

    Args:
    word (str): Từ cần kiểm tra.

    Returns:
    bool: True nếu từ hợp lệ, False nếu không hợp lệ.
    """
    chars = list(word)
    vowel_index = -1

    # Duyệt qua các ký tự trong từ
    for index, char in enumerate(chars):
        x, y = vowel_diacritic_mapping.get(char, (-1, -1))
        if x != -1:
            if vowel_index == -1:
                vowel_index = index
            else:
                # Kiểm tra nguyên âm liền kề có cách nhau 1 vị trí không
                if index - vowel_index != 1:
                    return False
                vowel_index = index
    return True

def normalize_diacritics_and_case(sentence):
    """
    Chuẩn hóa viết thường (lowercase) và loại bỏ dấu tiếng Việt của một câu.

    Args:
    sentence (str): Câu cần chuẩn hóa.

    Returns:
    str: Câu sau khi được chuẩn hóa.
    """
    sentence = sentence.lower()  # Chuyển câu về lowercase
    words = sentence.split()

    # Duyệt qua từng từ trong câu
    for index, word in enumerate(words):
        # Sử dụng biểu thức chính quy để tách từ và loại bỏ dấu câu
        cw = re.sub(r'(^\p{P}*)([p{L}.]*\p{L}+)(\p{P}*$)', r'\1/\2/\3', word).split('/')

        # Nếu từ được tách thành 3 phần và phần giữ nguyên âm cần được chuẩn hóa
        if len(cw) == 3:
            cw[1] = normalize_vietnamese_diacritics(cw[1])

        # Gắn từ đã xử lý lại vào danh sách từ
        words[index] = ''.join(cw)

    return ' '.join(words)

def remove_html_tags(txt):
    return re.sub(r'<[^>]*>', '', txt)


def text_preprocess(document):
    """
    Tiền xử lý văn bản bao gồm:
    1. Xóa mã HTML.
    2. Chuẩn hóa Unicode.
    3. Chuẩn hóa cách gõ dấu tiếng Việt.
    4. Tách từ.
    5. Đưa về lowercase.
    6. Xóa các ký tự không cần thiết.
    7. Tách thành danh sách các từ.

    Args:
    document (str): Văn bản cần tiền xử lý.

    Returns:
    list: Danh sách các từ sau khi tiền xử lý.
    """
    # Bước 1: Xóa mã HTML
    document = remove_html_tags(document)
    
    # Bước 2: Chuẩn hóa Unicode
    document = convert_unicode(document)
    
    # Bước 3: Chuẩn hóa cách gõ dấu tiếng Việt
    document = normalize_diacritics_and_case(document)
    
    # Bước 4: Tách từ
    document = word_tokenize(document, format="text")
    
    # Bước 5: Đưa về lowercase
    document = document.lower()
    
    # Bước 6: Xóa các ký tự không cần thiết
    document = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]', ' ', document)
    
    # Bước 7: Tách thành danh sách các từ
    word_list = document.split()
    
    # Xóa khoảng trắng thừa
    word_list = [word.strip() for word in word_list]
    
    return word_list


def tokenizer(texts):
    """
    Tokenize danh sách văn bản.

    Args:
    texts (list): Danh sách các văn bản cần tokenize.

    Returns:
    tuple: Một tuple chứa tokenizer và word_index.
           tokenizer: Đối tượng tokenizer đã được fit trên văn bản đầu vào.
           word_index: Một từ điển chứa các từ và chỉ số tương ứng của chúng trong tokenizer.
     Ex: word_index: {'sentence': 1, 'example': 2, 'an': 3, 'is': 4, 'this': 5, 'another': 6}
    """
    # Khởi tạo một đối tượng tokenizer
    tokenizer = Tokenizer()
    
    # Fit tokenizer trên văn bản đầu vào
    tokenizer.fit_on_texts(texts)

    # Lấy word_index, là một từ điển chứa các từ và chỉ số tương ứng của chúng trong tokenizer
    word_index = tokenizer.word_index
    
    return tokenizer, word_index


 # Hàm để đọc và hiển thị nội dung của tệp tin văn bản
def show_text_file_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content  

# Hàm load data
def load_data(data_folder):
    # Danh sách lưu trữ các câu và nhãn tương ứng
    texts = []
    labels = []

    # Lặp qua từng thư mục con trong thư mục chứa dữ liệu
    for folder in listdir(data_folder):
        # Bỏ qua tệp tin ẩn .DS_Store nếu có
        if folder != ".DS_Store":
            print("Load cat: ", folder)
            
            # Lặp qua từng tệp tin trong thư mục con
            for file in listdir(data_folder + sep + folder):
                # Bỏ qua tệp tin ẩn .DS_Store nếu có
                if file != ".DS_Store":
                    print("Load file: ", file)
                    
                    # Đọc nội dung từ tệp tin văn bản
                    with open(data_folder + sep + folder + sep + file, 'r', encoding="utf-8") as f:
                        all_of_it = f.read()
                        sentences = all_of_it.split('.')
                        
                        # Xử lý và làm sạch văn bản cho từng câu
                        sentences = [text_preprocess(sentence) for sentence in sentences]

                        # Thêm câu đã xử lý vào danh sách texts
                        texts = texts + sentences
                        
                        # Tạo danh sách nhãn tương ứng
                        label = [folder for _ in sentences]
                        
                        # Thêm nhãn vào danh sách labels
                        labels = labels + label
                        
                        # Giải phóng bộ nhớ
                        del all_of_it, sentences

    # Trả về danh sách texts và labels sau khi xử lý
    return texts, labels



def preprocess_page():
    CATEGORIES = "./data"
    DATA_FILENAME = "data.pkl"
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
    if not os.path.exists(CATEGORIES):
        st.warning(f"Thư mục nguồn '{CATEGORIES}' không tồn tại. Hãy chia dữ liệu trước khi tiền xử lý.")
        return

    # Nút "Thực thi tiền xử lý cho cả tập dữ liệu"
    execute_button_all_files = st.button("Thực thi tiền xử lý cho cả tập dữ liệu")

    
    if execute_button_all_files:
        
        if not os.path.exists(os.path.join(CATEGORIES, DATA_FILENAME)):
            st.warning("Chưa có dữ liệu. Tiến hành xây dựng dữ liệu...")
            texts, labels = load_data(CATEGORIES)
            my_tokenizer, word_index = tokenizer(texts)

            # put the tokens in a matrix
            X = my_tokenizer.texts_to_sequences(texts)
            X = pad_sequences(X)

            # prepare the labels
            y = pd.get_dummies(labels)
            with open(CATEGORIES + sep + DATA_FILENAME, 'wb') as f:
                pickle.dump([X, y, texts], f)

            # Hiển thị thông báo khi xây dựng dữ liệu
            st.success("Xây dựng dữ liệu thành công!")
        else:
            st.info("Dữ liệu đã tồn tại!")
            
        
    st.header("3. Xem Dữ Liệu")

    # Kiểm tra xem file data.pkl đã tồn tại hay chưa
    if not os.path.exists(CATEGORIES + sep + DATA_FILENAME):
        st.warning("Chưa có dữ liệu. Load dữ liệu trước khi xem.")
    else:
        # Đọc dữ liệu từ tệp tin
        try:
            with open(CATEGORIES + sep + DATA_FILENAME, 'rb') as f:
                X, y, texts = pickle.load(f)
                # Hiển thị thông báo khi load dữ liệu
                st.success("Load dữ liệu thành công!")
        except Exception as e:
                st.error(f"Lỗi khi load dữ liệu: {e}")

        st.text("Thông tin sau khi tải dữ liệu raw")
        # Hiển thị kích thước của X
        st.write("Kích thước của X:", X.shape)

        # Hiển thị các phần tử của X từ index 10 đến 29
        st.write("Các phần tử của X từ index 10 đến 29:", X[10:30])

        # Hiển thị các giá trị của y từ index 10 đến 29
        st.write("Các giá trị của y từ index 10 đến 29:", y[10:30])

        # Hiển thị các đoạn văn bản từ index 10 đến 29
        st.write("Các đoạn văn bản từ index 10 đến 29:", texts[10:30])