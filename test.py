import os
import numpy as np
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import joblib

def load_files(folder_path):
    """Загружает текстовые файлы из указанной папки."""
    file_contents = {}
    for filename in tqdm(os.listdir(folder_path), desc="Загрузка файлов"):
        if filename.endswith('.txt'):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                file_contents[filename] = f.read()
    return file_contents

model_path = 'learn_LLM/sgd_classifier.joblib'
vectorizer_path = 'learn_LLM/tfidf_vectorizer.joblib'

main_dataset_path = 'data_set_excel/output.txt'

def read_main_dataset(file_path):
    questions = []
    answers = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in tqdm(lines, desc="Чтение строк основного датасета"):
            if '@' in line:
                parts = line.rsplit('@', 1)
                if len(parts) == 2:
                    question, answer = parts
                    # Проверка на валидность данных
                    if question.strip() and answer.strip():
                        questions.append(question.strip())
                        answers.append(answer.strip())
                    
    return questions, answers

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    classifier = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Модель и векторизатор успешно загружены!")
else:
    main_questions, main_answers = read_main_dataset(main_dataset_path)
    questions = main_questions #+ file_questions
    answers = main_answers
    X_train, X_test, y_train, y_test = train_test_split(questions, answers, test_size=0.2, random_state=42)

    # Создание и обучение модели
    vectorizer = TfidfVectorizer()
    classifier = SGDClassifier()

    # Преобразование текстов в векторное представление
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Получение уникальных классов
    classes = np.unique(y_train)

    n_epochs = 1 # количество эпох

    print("Обучение модели...")
    for epoch in tqdm(range(n_epochs), desc="Эпохи обучения"):
        try:
            X_train_shuffled, y_train_shuffled = shuffle(X_train_vec, y_train)
            classifier.partial_fit(X_train_shuffled, y_train_shuffled, classes=classes)
        except Exception as e:
            print(f"Эпоха {epoch} пропущена из-за ошибки: {e}")

    # Сохранение модели и векторизатора
    joblib.dump(classifier, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print("Модель и векторизатор успешно обучены и сохранены!")

def load_excel(file_path):
    """Загружает данные из Excel файла."""
    print("Загрузка данных из Excel...")
    return pd.read_excel(file_path)

def train_tfidf_model(file_contents):
    """Создает TF-IDF модель на основе содержимого файлов."""
    print("Обучение TF-IDF модели...")
    vectorizer = TfidfVectorizer()
    filenames = list(file_contents.keys())
    documents = list(file_contents.values())
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix, filenames

def save_model(vectorizer, tfidf_matrix, filenames, model_path):
    """Сохраняет модель TF-IDF на диск."""
    print("Сохранение модели TF-IDF...")
    with open(model_path, 'wb') as f:
        pickle.dump((vectorizer, tfidf_matrix, filenames), f)

def load_model(model_path):
    """Загружает модель TF-IDF с диска."""
    print("Загрузка сохраненной модели TF-IDF...")
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def prepare_data_for_logistic_regression(data):
    """Подготавливает данные для обучения логистической регрессии."""
    print("Подготовка данных для модели...")
    texts = data.iloc[:, 4].fillna('') + ' ' + data.iloc[:, 5].fillna('')
    solutions = data.iloc[:, 6].fillna('')
    return texts, solutions

def create_logistic_pipeline():
    """Создает пайплайн для логистической регрессии."""
    print("Создание модели логистической регрессии...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LogisticRegression())
    ])
    return pipeline

def read_main_dataset(file_path):
    questions = []
    answers = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in tqdm(lines, desc="Чтение строк основного датасета"):
            if '@' in line:
                parts = line.rsplit('@', 1)
                if len(parts) == 2:
                    question, answer = parts
                    # Проверка на валидность данных
                    if question.strip() and answer.strip():
                        questions.append(question.strip())
                        answers.append(answer.strip())
                    
    return questions, answers

if __name__ == "__main__":
    # Путь к данным
    folder_path = "vse_v_txt"
    model_path = "tfidf_model.pkl"
    excel_path = "data_work/21.xlsx"

    # Загрузка текстовых файлов
    print("Загрузка текстовых файлов...")
    file_contents = load_files(folder_path)

    # Обучение TF-IDF модели
    vectorizer, tfidf_matrix, filenames = train_tfidf_model(file_contents)

    # Загрузка данных из Excel
    data = load_excel(excel_path)

    # Подготовка данных для логистической регрессии
    texts, solutions = prepare_data_for_logistic_regression(data)

    # Создание и обучение модели логистической регрессии
    print("Создание и обучение модели логистической регрессии...")
    X = texts
    y = (solutions != '').astype(int)  # Бинарная классификация: есть решение или нет

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logistic_model = create_logistic_pipeline()
    logistic_model.fit(tqdm(X_train, desc="Обучение"), y_train)

    while True:
        # Получение запроса от пользователя
        query = input("Введите запрос (или 'выход' для завершения): ")
        if query.lower() == 'выход':
            break

        # Поиск решения с помощью TF-IDF
        print("Поиск решения через TF-IDF...")
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        tfidf_index = similarities.argmax()

        # Поиск решения с помощью логистической регрессии
        print("Поиск решения через логистическую регрессию...")
        #logistic_prediction = logistic_model.predict([query])[0]

        user_question_vec = vectorizer.transform([query])
        model_answer = classifier.predict(user_question_vec)[0]

        # Вывод результатов
        if similarities[tfidf_index] > 0:
            print(f"TF-IDF наиболее релевантный файл: {filenames[tfidf_index]}")
            relevant_text = file_contents[filenames[tfidf_index]]
            relevant_snippet = "...".join(relevant_text.split(".")[:2])  # Первые два предложения
            print(f"Релевантный текст: {relevant_snippet}")
        #if model_answer > 0:
            """solution_index = solutions.index[solutions != ''][0]
            solution_text = solutions.iloc[solution_index]"""
            #print(f"Логистическая регрессия: Найдено решение: {model_answer}")
        else:
            print("Логистическая регрессия: Решение не найдено.")
