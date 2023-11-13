import streamlit as st
from preprocessing import preprocess_page

def main():
    st.title("Text Classification App")

    # Sidebar với mục lục
    st.sidebar.title("Mục lục")
    page = st.sidebar.radio("Chọn trang", ["Tiền Xử Lý Dữ Liệu"])

    if page == "Tiền Xử Lý Dữ Liệu":
        preprocess_page()


if __name__ == "__main__":
    main()