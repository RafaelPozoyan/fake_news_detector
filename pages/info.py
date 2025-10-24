import streamlit as st
from PIL import Image
from gensim.models import KeyedVectors


st.set_page_config(page_title="Обзор подходов", page_icon="🔍", layout="wide")

def set_styles():
    st.markdown("""
    <style>
                
    .stApp { 
        background: rgb(20, 19, 28);
    }
                
    h1 { 
        color: #FFFFE0; font-weight: 700; text-align: center; font-size: 3rem; margin-bottom: 0.5rem; 
    }
                
    h2, h3 { 
        color: #FFFFE0; font-weight: 600; 
    }
                
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #FFFFE0; border-radius: 7px; padding: 0.75rem 3rem; font-weight: 600;
        font-size: 1.1rem; border: none; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease; width: 100%;
    }
                
    .stButton>button:hover {
        transform: translateY(-2px); box-shadow: 0 8px 20px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
                
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 7px; border: 2px solid #e5e7eb; font-size: 1rem; padding: 0.75rem; transition: all 0.3s ease;
    }
                
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #667eea; box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
                
    .stTextInput>label, .stTextArea>label { 
        font-weight: 1000; color: #FFFFE0; font-size: 1.1rem; 
    }
                
    [data-testid="stSidebar"] { 
        background: #0e1117; 
    }
                
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { 
        color: white;
    }
                
    .metric-container { 
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border-radius: 12px; padding: 0.1rem; text-align: center; margin: 0.1rem; 
    }
    
    .metric-value {
        font-size: 2.5rem; font-weight: 700; color: #0369a1; margin-top: 0.1rem;
    }
    
    .metric-label {
        font-size: 1rem; color: #64748b;
    }
    
    .stImage.round-logo img {
        border-radius: 50%;
    }
    
    .stImage.default-img img {
        border-radius: 0;
    }
    
    .streamlit-expanderHeader {
        color: #FFFFE0; font-weight: 600;
    }
    
    </style>
    """, unsafe_allow_html=True)

if st.button("←"):
    st.switch_page("app.py")

st.title("Обзор подходов", anchor=None)

st.info("Векторизация через TF-IDF")

st.markdown("Использовал векторизацию через ***TF-IDF*** для оценки важности слов в датасете. Score примерно ***0.955***," \
" но не удавалось проверить согласованность заголовка и основного текста новости. При указании правильного текста инфоповода и изменении" \
" заголовка, модель все равно воспринимала новость, как правдивую, а все предсказания были с маленькой уверенностью.")

st.markdown("---")
st.markdown("При написании кода сравнил ***3 модели***:")
st.markdown("* Logistic Regression")
st.markdown("* Naive Bayes")
st.markdown("* Random Forest")
st.markdown("Наибольший скор и на валидационном, и на тренировочном датасетах выдала ***Логистическая регрессия***, как видно по графикам.")
st.image("assets/models_scores.png")

st.markdown("---")
st.image("assets/text_lenght.png")

st.markdown("---")
st.image("assets/wordcloud_vectorized.png")

st.info("Векториазция через Word2Vec")

st.markdown("При использовании Word2Vec учитывается семантическая связь между заголовком новости и основным текстом." \
"Дополнительно каждый запрос проходит проверку на соответсвие бизнес-правилам. То есть, при изменении заголовка новости на ложный," \
"модель это понимает и выводится объяснение, почему при высокой вероятность правдивости новости, она все равно фейковая.")

st.markdown("---")
st.markdown("При написании кода сравнил ***2 модели***:")
st.markdown("* Logistic Regression")
st.markdown("* Random Forest")
st.markdown("Наибольший скор и на валидационном, и на тренировочном датасетах выдала ***Логистическая регрессия***.")

st.image("assets/models_scores_w2v.png")

st.markdown("---")
if st.button("Вернуться на главную страницу", use_container_width=True):
    st.switch_page("app.py")
