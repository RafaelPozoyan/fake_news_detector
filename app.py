import os
import re
import pickle
import numpy as np
import streamlit as st
from PIL import Image
from gensim.models import KeyedVectors
import nltk
import json

with open("results/metrics/metrics.json", "r", encoding="utf-8") as f:
    metrics = json.load(f)

st.set_page_config(page_title="Детектор фейковых новостей", page_icon="🔍", layout="wide")

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

set_styles()

@st.cache_data
def load_stopwords():
    from nltk.corpus import stopwords

    nltk.download('stopwords')
    return set(stopwords.words('russian'))

def preprocess_text(text, stopwords_list):
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r'http\S+|www\S+|https\S+', '', t)
    t = re.sub(r'\S+@\S+', '', t)
    t = re.sub(r'<.*?>', '', t)
    t = re.sub(r'[^а-яёa-z0-9\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    toks = [w for w in t.split() if w not in stopwords_list and len(w) > 2]
    return " ".join(toks)

STOPWORDS = load_stopwords()

# Загрузка артефактов
@st.cache_resource
def load_artifacts():
    clf = None; kv = None; metrics = None

    if os.path.exists("models/fake_news_w2v_lr.pkl"):
        with open("models/fake_news_w2v_lr.pkl", "rb") as f:
            clf = pickle.load(f)

    if os.path.exists("models/w2v_vectors.kv"):
        kv = KeyedVectors.load("models/w2v_vectors.kv")

    if os.path.exists("models/metrics.pkl"):
        with open("models/metrics.pkl", "rb") as f:
            metrics = pickle.load(f)
    
    return clf, kv, metrics

clf, kv, train_metrics = load_artifacts()


def load_model():
    """Загрузка обученных моделей и векторизатора"""
    try:
        model_randfor_tf = pickle.load(open('models/Random Forest_model_tf.pkl', 'rb'))
        model_naibayes_tf = pickle.load(open('models/Naive Bayes_model_tf.pkl', 'rb'))
        model_logreg_tf = pickle.load(open('models/Logistic Regression_model_tf.pkl', 'rb'))
        vectorizer_tf = pickle.load(open('models/tfidf_vectorizer_tf.pkl', 'rb'))
        return model_randfor_tf, model_naibayes_tf, model_logreg_tf, vectorizer_tf, True
    except FileNotFoundError:
        return None, None, False

# Загружаем модель
model_randfor_tf, model_naibayes_tf, model_logreg_tf, vectorizer_tf, model_loaded = load_model()
stopwords_list = load_stopwords()

# Функции признаков
def doc_vector(tokens, kv_model):
    vecs = [kv_model[w] for w in tokens if w in kv_model]

    if not vecs:
        return np.zeros(kv_model.vector_size, dtype=np.float32)
    
    return np.vstack(vecs).mean(axis=0)

def cosine(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)

    if nu == 0 or nv == 0:
        return 0.0
    
    return float(np.dot(u, v) / (nu * nv))

def jaccard(a_tokens, b_tokens):
    A, B = set(a_tokens), set(b_tokens)

    if not A and not B:
        return 0.0
    
    return len(A & B) / max(1, len(A | B))

def overlap_ratio(a_tokens, b_tokens):
    A, B = set(a_tokens), set(b_tokens)
    
    return 0.0 if not A else len(A & B) / len(A)

def build_feature_vector(headline_clean, body_clean, kv_model, max_len=150):
    htoks = headline_clean.split()[:max_len]
    btoks = body_clean.split()[:max_len]

    h_vec = doc_vector(htoks, kv_model)
    b_vec = doc_vector(btoks, kv_model)

    cos_sim = cosine(h_vec, b_vec)
    jacc = jaccard(htoks, btoks)
    ovr = overlap_ratio(htoks, btoks)
    diff = np.abs(h_vec - b_vec)
    prod = h_vec * b_vec
    l2 = np.linalg.norm(h_vec - b_vec)

    feats = np.hstack([h_vec, b_vec, diff, prod, [cos_sim, jacc, ovr, l2]])
    return feats, {"cosine": cos_sim, "jaccard": jacc, "overlap": ovr, "l2": l2}

def predict(headline_raw, body_raw, clf_model, kv_model):
    headline_clean = preprocess_text(headline_raw, STOPWORDS)
    body_clean = preprocess_text(body_raw, STOPWORDS)
    
    if len(headline_clean) < 2 or len(body_clean) < 5:
        return None, None, headline_clean, body_clean, None
    
    X, rel = build_feature_vector(headline_clean, body_clean, kv_model)
    X = X.reshape(1, -1)
    
    prob = clf_model.predict_proba(X)[0]
    pred = int(clf_model.predict(X)[0])
    
    return pred, prob, headline_clean, body_clean, rel

# Фиксированные пороги и правила
DEFAULT_THRESHOLDS = {
    "proba_real": 0.55,   
    "cos_min": 0.20,      
    "jacc_min": 0.005,     
    "overlap_min": 0.10,  
    "l2_max": 14.0,       
}
HARD_RULES = {
    "very_low_cos": 0.10,
    "zero_overlap": 0.00,
}

def decide_with_rules(prob_real, rel, thresholds=DEFAULT_THRESHOLDS, hard_rules=HARD_RULES):
    reasons = []

    if rel["cosine"] < hard_rules["very_low_cos"]:
        reasons.append(f"Низкая косинусная близость (cosine): {rel['cosine']:.3f}")
        return 0, reasons
    
    if rel["overlap"] <= hard_rules["zero_overlap"]:
        reasons.append("В заголовке нет слов, встречающихся в тексте (overlap=0)")
        return 0, reasons

    soft_ok = True
    if rel["cosine"] < thresholds["cos_min"]:
        soft_ok = False; reasons.append(f"cosine ниже порога ({rel['cosine']:.3f} < {thresholds['cos_min']})")
    
    if rel["jaccard"] < thresholds["jacc_min"]:
        soft_ok = False; reasons.append(f"Jaccard ниже порога ({rel['jaccard']:.3f} < {thresholds['jacc_min']})")
    
    if rel["overlap"] < thresholds["overlap_min"]:
        soft_ok = False; reasons.append(f"overlap ниже порога ({rel['overlap']:.3f} < {thresholds['overlap_min']})")
    
    if rel["l2"] > thresholds["l2_max"]:
        soft_ok = False; reasons.append(f"L2 выше порога ({rel['l2']:.3f} > {thresholds['l2_max']})")

    if prob_real >= thresholds["proba_real"] and soft_ok:
        return 1, reasons
    else:
    
        if prob_real >= thresholds["proba_real"]:
            reasons.append(f"Вероятность модели высокая ({prob_real*100:.1f}%), но связи между заголовком и текстом нет")
    
        return 0, reasons

# Сайдбар
with st.sidebar:
    logo = Image.open('assets/logo.png')
    st.image(logo, output_format="round-logo")

    st.markdown("---")
    st.markdown(
        """
        <div style='color: #FFFFE0;'>
            <h3 style='color: #FFFFE0;'>О проекте</h3>
            <p>Система автоматической детекции фейковых новостей. При введении заголовка и основного текста статьи, сайт проверит,
            является ли инфоповод подлинным.</p>
            </p>При решении задачи используются логистическая регрессия и Word2Vec. Материал проверяется на согласованность заголовка и основного текста.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    
    if kv is not None:
        st.markdown("<div style='color:#FFFFE0;'><h4>Статус Word2Vec:</h4><p>Модели и эмбеддинги загружены</p></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:#FFFFE0;'><h4>Статус Word2Vec:</h4><p>Нет моделей или эмбеддингов в models</p></div>", unsafe_allow_html=True)

    if model_loaded:
        st.markdown("<div style='color:#FFFFE0;'><h4>Статус TF-IDF:</h4><p>Модели и эмбеддинги загружены</p></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:#FFFFE0;'><h4>Статус TF-IDF:</h4><p>Нет моделей или эмбеддингов в models</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    # if train_metrics:
    #     st.markdown(
    #         f"""
    #         <div style='color:#FFFFE0;'>
    #             <h4>Метрики:</h4>
    #             <ul>
    #                 <li><strong>Модель:</strong> {train_metrics.get("best_model_name","Logistic Regression + Word2Vec")}</li>
    #                 <li><strong>Val Accuracy:</strong> {train_metrics.get("val_accuracy","-"):.3f}</li>
    #                 <li><strong>Val F1:</strong> {train_metrics.get("val_f1","-"):.3f}</li>
    #             </ul>
    #         </div>
    #         """,
    #         unsafe_allow_html=True
    #     )

# Главный контент
st.title('Детектор фейковых новостей')

st.markdown("""
    <style>
    div[data-testid="stExpander"] div[role="button"] {
        background-color: #2e86de;
        color: white;
        border-radius: 5px;
    }
    div[data-testid="stExpander"] div[role="region"] {
        background-color: #dff9fb;
    }
    </style>
""", unsafe_allow_html=True)
with st.expander("Как пользоваться?", expanded=False):
    st.markdown("""<div style='color:#FFFFE0;'>
    1. Введите заголовок новости в первое поле
    </div>""", unsafe_allow_html=True)
    st.markdown("""<div style='color:#FFFFE0;'>
    2. Вставьте текст новости во второе поле
    </div>""", unsafe_allow_html=True)
    st.markdown("""<div style='color:#FFFFE0;'>
    3. Нажмите кнопку "Проверить новость"
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
 
with st.container():
    headline = st.text_input('Заголовок новости:', placeholder='Вставьте заголовок новости')
    body = st.text_area('Текст новости:', height=250, placeholder='Вставьте основной текст новости')
    c1, c2, c3 = st.columns([1, 2, 1])
    
    with c2:
        check_button = st.button('Проверить новость', use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Предсказание

if check_button:
    st.markdown("---")
    if not headline or not body:
        st.warning('⚠️ Пожалуйста, заполните заголовок и текст новости')
    else:
        with st.spinner('🔄 Анализирую новость...'):
            try:
                with open("models/logisticregression_model.pkl", "rb") as f:
                    clf_lr = pickle.load(f)
                with open("models/randomforest_model.pkl", "rb") as f:
                    clf_rf = pickle.load(f)

                # prediction для каждой модели
                pred_lr, prob_lr, h_clean, b_clean, rel_lr = predict(headline, body, clf_lr, kv)
                pred_rf, prob_rf, _, _, rel_rf = predict(headline, body, clf_rf, kv)

                # Проверка бизнес-правил
                final_label_lr, reasons_lr = decide_with_rules(prob_lr[1], rel_lr)
                final_label_rf, reasons_rf = decide_with_rules(prob_rf[1], rel_rf)
                
                # =================================
                # Вывод результатов работы Word2Vec
                # =================================
                st.markdown("### Word2Vec")
                col1, col2 = st.columns(2)

                # Logistic Regression
                with col1:
                    if final_label_lr == 1:
                        st.success('✅ **РЕАЛЬНАЯ НОВОСТЬ**')
                        st.markdown(
                            f"""
                            <div class='metric-container'>
                                <div class='metric-label'>Logistic Regression</div>
                                <div class='metric-value'>{prob_lr[1]*100:.1f}%</div>
                                <div class='metric-label'>Уверенность</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # with st.expander("Семантическая связь заголовка и основного текста"):
                        #     if rel:
                        #         c1, c2, c3, c4 = st.columns(4)
                        #         c1.metric("Cosine", f"{rel['cosine']:.3f}")
                        #         c2.metric("Jaccard", f"{rel['jaccard']:.3f}")
                        #         c3.metric("Overlap", f"{rel['overlap']:.3f}")
                        #         c4.metric("L2(h-b)", f"{rel['l2']:.3f}")

                        #     st.markdown("**Тексты после предобработки:**")
                        #     st.write(f"- Заголовок: {h_clean}")
                        #     show_body = (' '.join(b_clean.split()[:120]) + ' ...') if len(b_clean.split())>120 else b_clean
                        #     st.write(f"- Текст: {show_body}") 

                    else:
                        st.error('❌ **ФЕЙКОВАЯ НОВОСТЬ**')
                        st.markdown(
                            f"""
                            <div class='metric-container' style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);'>
                                <div class='metric-label'>Logistic Regression</div>
                                <div class='metric-value' style='color: #b91c1c;'>{(1-prob_lr[1])*100:.1f}%</div>
                                <div class='metric-label'>Уверенность</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        if reasons_lr:
                            with st.expander("Почему сработали бизнес-правила?"):
                                for r in reasons_lr:
                                    st.write(f"- {r}")
                        # with st.expander("Семантическая связь заголовка и основного текста"):
                        #     if rel:
                        #         c1, c2, c3, c4 = st.columns(4)
                        #         c1.metric("Cosine", f"{rel['cosine']:.3f}")
                        #         c2.metric("Jaccard", f"{rel['jaccard']:.3f}")
                        #         c3.metric("Overlap", f"{rel['overlap']:.3f}")
                        #         c4.metric("L2(h-b)", f"{rel['l2']:.3f}")

                        #     st.markdown("**Тексты после предобработки:**")
                        #     st.write(f"- Заголовок: {h_clean}")
                        #     show_body = (' '.join(b_clean.split()[:120]) + ' ...') if len(b_clean.split())>120 else b_clean
                        #     st.write(f"- Текст: {show_body}") 

                # Random Forest
                with col2:
                    if final_label_rf == 1:
                        st.success('✅ **РЕАЛЬНАЯ НОВОСТЬ**')
                        st.markdown(
                            f"""
                            <div class='metric-container'>
                                <div class='metric-label'>Random Forest</div>
                                <div class='metric-value'>{prob_rf[1]*100:.1f}%</div>
                                <div class='metric-label'>Уверенность</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        # with st.expander("Семантическая связь заголовка и основного текста"):
                        #     if rel:
                        #         c1, c2, c3, c4 = st.columns(4)
                        #         c1.metric("Cosine", f"{rel['cosine']:.3f}")
                        #         c2.metric("Jaccard", f"{rel['jaccard']:.3f}")
                        #         c3.metric("Overlap", f"{rel['overlap']:.3f}")
                        #         c4.metric("L2(h-b)", f"{rel['l2']:.3f}")

                        #     st.markdown("**Тексты после предобработки:**")
                        #     st.write(f"- Заголовок: {h_clean}")
                        #     show_body = (' '.join(b_clean.split()[:120]) + ' ...') if len(b_clean.split())>120 else b_clean
                        #     st.write(f"- Текст: {show_body}")                        
                        
                    else:
                        st.error('❌ **ФЕЙКОВАЯ НОВОСТЬ**')
                        st.markdown(
                            f"""
                            <div class='metric-container' style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);'>
                                <div class='metric-label'>Random Forest</div>
                                <div class='metric-value' style='color: #b91c1c;'>{(1-prob_rf[1])*100:.1f}%</div>
                                <div class='metric-label'>Уверенность</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        if reasons_rf:
                            with st.expander("Почему сработали бизнес-правила?"):
                                for r in reasons_rf:
                                    st.write(f"- {r}")
                        # with st.expander("Семантическая связь заголовка и основного текста"):
                        #     if rel:
                        #         c1, c2, c3, c4 = st.columns(4)
                        #         c1.metric("Cosine", f"{rel['cosine']:.3f}")
                        #         c2.metric("Jaccard", f"{rel['jaccard']:.3f}")
                        #         c3.metric("Overlap", f"{rel['overlap']:.3f}")
                        #         c4.metric("L2(h-b)", f"{rel['l2']:.3f}")

                        #     st.markdown("**Тексты после предобработки:**")
                        #     st.write(f"- Заголовок: {h_clean}")
                        #     show_body = (' '.join(b_clean.split()[:120]) + ' ...') if len(b_clean.split())>120 else b_clean
                        #     st.write(f"- Текст: {show_body}")


                # =================================
                # Вывод результатов TF-IDF
                # =================================

                headline_clean = preprocess_text(headline, stopwords_list)
                body_clean = preprocess_text(body, stopwords_list)
                
                # Комбинируем
                body_words = body_clean.split()
                combined_text = f"{headline_clean} {' '.join(body_words)}"

                text_vec = vectorizer_tf.transform([combined_text])
                
                # предикты на random forest
                prediction_rf = model_randfor_tf.predict(text_vec)[0]
                probabilities_rf = model_randfor_tf.predict_proba(text_vec)[0]

                # предикты на naive bayes
                prediction_nb = model_naibayes_tf.predict(text_vec)[0]
                probabilities_nb = model_naibayes_tf.predict_proba(text_vec)[0]  

                # предикты на logistic regression
                prediction_lr = model_logreg_tf.predict(text_vec)[0]
                probabilities_lr = model_logreg_tf.predict_proba(text_vec)[0]  

                st.markdown("### TF-IDF")

                col1, col2, col3 = st.columns(3)

                # random forest
                with col1:
                    if prediction_rf == 1:
                        # Реальная новость
                        confidence = probabilities_rf[1] * 100
                        st.success(f'✅ **РЕАЛЬНАЯ НОВОСТЬ**')
                        
                        st.markdown(
                            f"""
                            <div class='metric-container'>
                                <div class='metric-label'>Random Forest</div>
                                <div class='metric-value'>{confidence:.1f}%</div>
                                <div class='metric-label'>Уверенность</div>
                                <div class='metric-label'><strong>Val Accuracy:</strong> {metrics["Random Forest"]["val_acc"]:.3f}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                    else:
                        # Фейковая новость
                        confidence = probabilities_rf[0] * 100
                        st.error(f'❌ **ФЕЙКОВАЯ НОВОСТЬ**')
                        
                        st.markdown(
                            f"""
                            <div class='metric-container' style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);'>
                                <div class='metric-label'>Random Forest</div>
                                <div class='metric-value' style='color: #b91c1c;'>{confidence:.1f}%</div>
                                <div class='metric-label'>Уверенность</div>
                                <div class='metric-label'><strong>Val Accuracy:</strong> {metrics["Random Forest"]["val_acc"]:.3f}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                # naive bayes
                with col2:
                    if prediction_nb == 1:
                        # Реальная новость
                        confidence = probabilities_nb[1] * 100
                        st.success(f'✅ **РЕАЛЬНАЯ НОВОСТЬ**')
                        
                        st.markdown(
                            f"""
                            <div class='metric-container'>
                                <div class='metric-label'>Naive Bayes</div>
                                <div class='metric-value'>{confidence:.1f}%</div>
                                <div class='metric-label'>Уверенность</div>
                                <div class='metric-label'><strong>Val Accuracy:</strong> {metrics["Naive Bayes"]["val_acc"]:.3f}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )                        
                        
                    else:
                        # Фейковая новость
                        confidence = probabilities_nb[0] * 100
                        st.error(f'❌ **ФЕЙКОВАЯ НОВОСТЬ**')
                        
                        st.markdown(
                            f"""
                            <div class='metric-container' style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);'>
                                <div class='metric-label'>Naive Bayes</div>
                                <div class='metric-value' style='color: #b91c1c;'>{confidence:.1f}%</div>
                                <div class='metric-label'>Уверенность</div>
                                <div class='metric-label'><strong>Val Accuracy:</strong> {metrics["Naive Bayes"]["val_acc"]:.3f}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                # logistic regression
                with col3:
                    if prediction_lr == 1:
                        # Реальная новость
                        confidence = probabilities_lr[1] * 100
                        st.success(f'✅ **РЕАЛЬНАЯ НОВОСТЬ**')
                        
                        st.markdown(
                            f"""
                            <div class='metric-container'>
                                <div class='metric-label'>Logistic Regression</div>
                                <div class='metric-value'>{confidence:.1f}%</div>
                                <div class='metric-label'>Уверенность</div>
                                <div class='metric-label'><strong>Val Accuracy:</strong> {metrics["Logistic Regression"]["val_acc"]:.3f}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                    else:
                        # Фейковая новость
                        confidence = probabilities_lr[0] * 100
                        st.error(f'❌ **ФЕЙКОВАЯ НОВОСТЬ**')
                        
                        st.markdown(
                            f"""
                            <div class='metric-container' style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);'>
                                <div class='metric-label'>Logistic Regression</div>
                                <div class='metric-value' style='color: #b91c1c;'>{confidence:.1f}%</div>
                                <div class='metric-label'>Уверенность</div>
                                <div class='metric-label'><strong>Val Accuracy:</strong> {metrics["Logistic Regression"]["val_acc"]:.3f}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    



            except Exception as e:
                st.error(f'❌ Ошибка: {str(e)}')
 
st.markdown("---")

with st.expander("Обзор подходов"):
    st.info("Векторизация через TF-IDF")

    st.markdown("В начале использовал векторизацию через ***TF-IDF*** для оценки важности слов в датасете. Score был примерно ***0.958***," \
    " но не удавалось проверить согласованность заголовка и основного текста новости. При указании правильного текста инфоповода и изменении" \
    " заголовка, модель все равно воспринимала новость, как правдивую, а все предсказания были с маленькой уверенностью." \
    " В итоге принял решение реализовать проект через ***Word2Vec***, чтобы учитывать контекст.")
    
    st.markdown("---")
    st.image("assets/models_scores.png", caption="Сравнение моделей")

    st.markdown("При написании кода сравнил ***3 модели***:")
    st.markdown("* Logistic Regression")
    st.markdown("* Naive Bayes")
    st.markdown("* Random Forest")
    st.markdown("Наибольший скор выдала ***Логистическая регрессия***, как видно по графикам.")

    st.markdown("---")
    st.image("assets/text_lenght.png", caption="Распределение длины текстов")
    
    st.markdown("---")
    st.image("assets/wordcloud_vectorized.png", caption="Облако наиболее значимых слов")


