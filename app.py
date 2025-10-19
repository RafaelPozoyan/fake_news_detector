import streamlit as st
import pickle
import re
from PIL import Image
import os
import json
from pathlib import Path

# ============================================
# НАСТРОЙКА СТРАНИЦЫ
# ============================================
st.set_page_config(
    page_title="Детектор фейковых новостей",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CSS СТИЛИ
# ============================================
def set_background_and_styles():
    st.markdown(
        """
        <style>
        /* Основной фон приложения */
        .stApp {
            background: rgb(20, 19, 28);
        }
        
        /* Заголовки */
        h1 {
            color: #FFFFE0;
            font-weight: 700;
            text-align: center;
            font-size: 3rem;
            margin-bottom: 0.5rem;
        }
        
        h2 {
            color: #FFFFE0;
            font-weight: 600;
        }
        
        h3 {
            color: #FFFFE0;
            font-weight: 600;
        }
        
        /* Кнопки */
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #FFFFE0;
            border-radius: 7px;
            padding: 0.75rem 3rem;
            font-weight: 600;
            font-size: 1.1rem;
            border: none;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.6);
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
        
        .stButton>button:active {
            transform: translateY(0);
        }
        
        /* Текстовые поля */
        .stTextInput>div>div>input {
            border-radius: 7px;
            border: 2px solid #e5e7eb;
            font-size: 1rem;
            padding: 0.75rem;
            transition: all 0.3s ease;
        }
        
        .stTextInput>div>div>input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        /* Текстовая область */
        .stTextArea>div>div>textarea {
            border-radius: 7px;
            border: 2px solid #e5e7eb;
            font-size: 1rem;
            padding: 0.75rem;
            transition: all 0.3s ease;
        }
        
        .stTextArea>div>div>textarea:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        /* Метки полей ввода */
        .stTextInput>label, .stTextArea>label {
            font-weight: 1000;
            color: #FFFFE0;
            font-size: 1.1rem;
        }
        
        /* Success сообщения */
        .stSuccess {
            background-color: #d1fae5;
            border-left: 5px solid #10b981;
            border-radius: 10px;
            padding: 1rem;
        }
        
        /* Error сообщения */
        .stError {
            background-color: #fee2e2;
            border-left: 5px solid #ef4444;
            border-radius: 10px;
            padding: 1rem;
        }
        
        /* Warning сообщения */
        .stWarning {
            background-color: #fef3c7;
            border-left: 5px solid #f59e0b;
            border-radius: 10px;
            padding: 1rem;
        }
        
        /* Info сообщения */
        .stInfo {
            background-color: #dbeafe;
            border-left: 5px solid #3b82f6;
            border-radius: 10px;
            padding: 1rem;
        }
        
        /* Сайдбар */
        [data-testid="stSidebar"] {
            background: #0e1117;
        }
        
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            color: white;
        }
        
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            color: white;
        }
        
        /* Разделители */
        hr {
            margin-top: 2rem;
            margin-bottom: 2rem;
            border: none;
            border-top: 2px solid #e5e7eb;
        }
        
        /* Кастомные классы для карточек */
        .info-card {
            background-color: #f8fafc;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        
        .metric-container {
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            margin: 1rem 0;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #0369a1;
        }

        .stImage.round-logo img {
            border-radius: 50%;
        }
        .stImage.default-img img {
            border-radius: 0 !important;
        }

        .metric-label {
            font-size: 1rem;
            color: #64748b;
            margin-top: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Применяем стили
set_background_and_styles()

# ============================================
# ФУНКЦИИ ПРЕДОБРАБОТКИ
# ============================================
def load_metrics():
    path = Path("results/metrics/metrics.json")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

metrics = load_metrics()
if metrics:
    best_model_name = metrics["best_model_name"]
    best_acc = metrics["val_accuracy"]
    best_f1 = metrics["val_f1"]
else:
    best_model_name, best_acc, best_f1 = None, None, None

import nltk
from nltk.corpus import stopwords
import streamlit as st

@st.cache_data(show_spinner=False)
def ensure_nltk_stopwords():
    try:
        # Проба доступа без скачивания
        _ = stopwords.words('russian')
    except LookupError:
        nltk.download('stopwords')
    return stopwords.words('russian')

@st.cache_data(show_spinner=False)
def load_stopwords():
    return ensure_nltk_stopwords()

def preprocess_text(text, stopwords_list):
    """Предобработка текста"""
    if not isinstance(text, str) or len(text) == 0:
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^а-яё\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = text.split()
    words = [word for word in words if word not in stopwords_list and len(word) > 2]
    
    return ' '.join(words)

# ============================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================
@st.cache_resource
def load_model():
    """Загрузка обученной модели и векторизатора"""
    try:
        model = pickle.load(open('models/fake_news_detector.pkl', 'rb'))
        vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
        return model, vectorizer, True
    except FileNotFoundError:
        return None, None, False

# Загружаем модель
model, vectorizer, model_loaded = load_model()
stopwords_list = load_stopwords()

# ============================================
# САЙДБАР
# ============================================
with st.sidebar:

    # Логотип
    logo = Image.open('assets/logo.png')
    st.image(logo, use_column_width=True, output_format="round-logo")
    
    st.markdown("---")
    
    st.markdown(
        """
        <div style='color: #FFFFE0;'>
            <h3 style='color: #FFFFE0;'>О проекте</h3>
            <p>Система автоматической детекции фейковых новостей. Вы можете проверить любую новость на достоверность, 
            просто вставив заголовок  и текст инфоповода в соответствующие поля</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    if model_loaded:
        st.markdown(
            """
            <div style='color: #FFFFE0;'>
                <h4 style='color: #FFFFE0;'>✅ Статус модели</h4>
                <p>Модель загружена и готова к работе</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style='color: #FFFFE0;'>
                <h4 style='color: #FFFFE0;'>⚠️ Статус модели</h4>
                <p>Модель не найдена.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    st.markdown(
        """
        <div style='color: #FFFFE0;'>
            <h4 style='color: #FFFFE0;'>📊 Информация</h4>
            <ul>
                <li><strong>Модель:</strong> {model}</li>
                <li><strong>Векторизация:</strong> TF-IDF</li>
                <li><strong>Score на датасете (accuracy):</strong> {accuracy_score:.3f}</li>
            </ul>
        </div>
        """.format(model=best_model_name, accuracy_score=best_acc),
        unsafe_allow_html=True
    )

# ============================================
# ГЛАВНЫЙ КОНТЕНТ
# ============================================

# Заголовок
st.title('🔍 Детектор фейковых новостей')

st.markdown(
    """
    <div style='text-align: center; color: #FFFFE0; font-size: 1.2rem; margin-bottom: 2rem;'>
        Проверьте достоверность новости с помощью искусственного интеллекта
    </div>
    """,
    unsafe_allow_html=True
)

# Инструкция
with st.expander("📖 Как пользоваться?", expanded=False):
    st.markdown("""
    1. Введите заголовок новости в первое поле
    2. Вставьте текст новости во второе поле
    3. Нажмите кнопку **Проверить новость**
    4. Получите результат с процентом уверенности
    """)

st.markdown("<br>", unsafe_allow_html=True)

# Форма ввода
with st.container():
    headline = st.text_input(
        '📰 Заголовок новости:',
        placeholder='Например: Банк России снизил ключевую ставку до 17%',
        help='Введите заголовок новости, которую хотите проверить'
    )
    
    body = st.text_area(
        '📄 Текст новости:',
        height=250,
        placeholder='Вставьте полный текст новости или её основную часть...',
        help='Вставьте текст статьи для анализа'
    )
    
    # Кнопка проверки
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        check_button = st.button('🔍 Проверить новость', use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================
# ОБРАБОТКА И РЕЗУЛЬТАТЫ
# ============================================
if check_button:
    if not headline or not body:
        st.warning('⚠️ Пожалуйста, заполните оба поля: заголовок и текст новости')
    else:
        with st.spinner('🔄 Анализирую новость...'):
            try:
                # Предобработка
                headline_clean = preprocess_text(headline, stopwords_list)
                body_clean = preprocess_text(body, stopwords_list)
                
                # Комбинируем
                body_words = body_clean.split()
                combined_text = f"{headline_clean} {' '.join(body_words)}"
                
                if len(combined_text) < 10:
                    st.error('❌ Недостаточно текста для анализа. Пожалуйста, введите больше информации.')
                else:
                    # Векторизация
                    text_vec = vectorizer.transform([combined_text])
                    
                    # Предсказание
                    prediction = model.predict(text_vec)[0]
                    probabilities = model.predict_proba(text_vec)[0]
                    
                    # Вывод результата
                    st.markdown("---")
                    st.markdown("### 📊 Результат анализа:")
                    
                    if prediction == 1:
                        # Реальная новость
                        confidence = probabilities[1] * 100
                        st.success(f'✅ **РЕАЛЬНАЯ НОВОСТЬ**')
                        
                        st.markdown(
                            f"""
                            <div class='metric-container'>
                                <div class='metric-value'>{confidence:.1f}%</div>
                                <div class='metric-label'>Уверенность</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        st.info("""
                        **Интерпретация:** Заголовок соответствует содержанию статьи. 
                        Новость скорее всего является достоверной.
                        """)
                        
                    else:
                        # Фейковая новость
                        confidence = probabilities[0] * 100
                        st.error(f'❌ **ФЕЙКОВАЯ НОВОСТЬ**')
                        
                        st.markdown(
                            f"""
                            <div class='metric-container' style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);'>
                                <div class='metric-value' style='color: #b91c1c;'>{confidence:.1f}%</div>
                                <div class='metric-label'>Уверенность</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        st.warning("""
                        **Интерпретация:** Заголовок не соответствует или противоречит содержанию статьи. 
                        Новость может быть недостоверной или вводить в заблуждение.
                        """)
                    
                    # Дополнительная информация
                    with st.expander("📈 Подробная информация"):
                        st.write("**Распределение вероятностей:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Фейковая новость", f"{probabilities[0]:.2f}")
                        with col2:
                            st.metric("Реальная новость", f"{probabilities[1]:.2f}")
                        
                        st.write("**Длина обработанного текста:**")
                        st.write(f"- Слов в заголовке: {len(headline_clean.split())}")
                        st.write(f"- Слов в тексте (использовано): {len(body_words)}")
                        st.write(f"- Всего слов для анализа: {len(combined_text.split())}")

  
            except Exception as e:
                st.error(f'❌ Произошла ошибка при анализе: {str(e)}')
                # Графики
with st.expander("🤖 Работа моделей"):
    st.image("assets/fake_news_analysis.png", caption="Сравнение моделей", use_column_width=True, output_format="default-img")
    st.image("assets/wordcloud_vectorized.png", caption="Облако наиболее важных слов", use_column_width=True, output_format="default-img")