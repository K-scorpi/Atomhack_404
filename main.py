import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import os

# Определение путей для сохранения модели и векторизатора
model_path = 'learn_LLM/sgd_classifier.joblib'
vectorizer_path = 'learn_LLM/tfidf_vectorizer.joblib'

# Функция для чтения и предобработки данных из нескольких файлов
def read_and_preprocess(file_paths):
    questions = []
    answers = []
    
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in tqdm(lines, desc=f"Чтение строк из {file_path}"):
                if '@' in line:
                    parts = line.rsplit('@', 1)
                    if len(parts) == 2:
                        question, answer = parts
                        # Проверка на валидность данных
                        if question.strip() and answer.strip():
                            questions.append(question.strip())
                            answers.append(answer.strip())
    
    return questions, answers

# Функция для чтения основного датасета
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

# Получение списка всех текстовых файлов
#data_files = [f"vse_v_txt/{i}.txt" for i in range(1, 29)]
main_dataset_path = 'data_set_excel/output.txt'

# Проверка наличия сохраненной модели и векторизатора
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    classifier = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Модель и векторизатор успешно загружены!")
else:
    # Чтение и предобработка данных из основного датасета и текстовых файлов
    main_questions, main_answers = read_main_dataset(main_dataset_path)
    #file_questions, file_answers = read_and_preprocess(data_files)

    # Объединение данных
    questions = main_questions #+ file_questions
    answers = main_answers #+ file_answers

    # Проверка, что есть данные для обучения
    #if not questions or not answers:
    #    raise ValueError("Нет валидных данных для обучения модели.")

    # Разделение данных на обучающую и тестовую выборки
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

# Функция для уточнения и получения подтверждения от пользователя
def get_clarification(prediction):
    while True:
        print("Ответ:", prediction)
        confirmation = input("Это ответ, который вы искали? (да/нет): ").strip().lower()
        if confirmation == 'да':
            return True
        elif confirmation == 'нет':
            return False
        else:
            print("Пожалуйста, ответьте 'да' или 'нет'.")

# Функция для поиска ответа в текстовых файлах
def search_in_files(question, file_paths):
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                if '@' in line:
                    parts = line.rsplit('@', 1)
                    if len(parts) == 2:
                        file_question, answer = parts
                        if file_question.strip() == question.strip():
                            return answer.strip()
    return None

# Функция для выбора наиболее релевантного ответа
def choose_best_answer(question, model_answer, file_answer):
    question_vec = vectorizer.transform([question])
    model_answer_vec = vectorizer.transform([model_answer])
    file_answer_vec = vectorizer.transform([file_answer])

    model_similarity = cosine_similarity(question_vec, model_answer_vec)
    file_similarity = cosine_similarity(question_vec, file_answer_vec)

    if model_similarity > file_similarity:
        return model_answer
    else:
        return file_answer

# Цикл для получения вопросов от пользователя
print("Введите свои вопросы. Напишите 'stop', чтобы завершить сеанс.")
while True:
    user_question = input("Ваш вопрос: ").strip()
    if user_question.lower() in ['stop', 'стоп']:
        break
    try:
        # Предсказание от модели
        user_question_vec = vectorizer.transform([user_question])
        model_answer = classifier.predict(user_question_vec)[0]

        # Поиск ответа в текстовых файлах
        #file_answer = search_in_files(user_question, data_files)

        # Если ответ найден и в файлах
        
        best_answer = model_answer

        attempts = 2
        while attempts > 0:
            if get_clarification(best_answer):
                print("Рад, что смог помочь!")
                break
            else:
                attempts -= 1
                if attempts == 0:
                    print("Перевожу на оператора.")
                else:
                    user_question = input("Можете ли вы переформулировать ваш вопрос или предоставить больше деталей? ").strip()

                    # Обновить предсказание от модели
                    user_question_vec = vectorizer.transform([user_question])
                    model_answer = classifier.predict(user_question_vec)[0]

                    # Обновить поиск ответа в текстовых файлах
                    #file_answer = search_in_files(user_question, data_files)

                    """if file_answer:
                        best_answer = choose_best_answer(user_question, model_answer, file_answer)
                    else:"""
                    best_answer = model_answer

    except Exception as e:
        print(f"Произошла ошибка при обработке вопроса: {e}")
