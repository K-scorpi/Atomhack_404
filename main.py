import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import joblib
import os

# Определение путей для сохранения модели и векторизатора
model_path = 'sgd_classifier.joblib'
vectorizer_path = 'tfidf_vectorizer.joblib'

# Проверка наличия сохраненной модели и векторизатора
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    classifier = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Model and vectorizer loaded successfully!")
else:
    # Функция для чтения и предобработки данных
    def read_and_preprocess(file_path):
        questions = []
        answers = []
        
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in tqdm(lines, desc="Reading lines"):
                if '@' in line:
                    parts = line.rsplit('@', 1)
                    if len(parts) == 2:
                        question, answer = parts
                        # Проверка на валидность данных
                        if question.strip() and answer.strip():
                            questions.append(question.strip())
                            answers.append(answer.strip())
                        
        return questions, answers

    # Чтение и предобработка данных
    questions, answers = read_and_preprocess('data_set_excel/output.txt')

    # Проверка, что есть данные для обучения
    if not questions or not answers:
        raise ValueError("No valid data to train the model.")

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

    n_epochs = 15  # количество эпох

    print("Training the model...")
    for epoch in tqdm(range(n_epochs), desc="Training epochs"):
        try:
            X_train_shuffled, y_train_shuffled = shuffle(X_train_vec, y_train)
            classifier.partial_fit(X_train_shuffled, y_train_shuffled, classes=classes)
        except Exception as e:
            print(f"Epoch {epoch} skipped due to an error: {e}")

    # Сохранение модели и векторизатора
    joblib.dump(classifier, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print("Model and vectorizer trained and saved successfully!")

# Функция для уточнения и получения подтверждения от пользователя
def get_clarification(prediction):
    while True:
        print("Answer:", prediction)
        confirmation = input("Is this the answer you were looking for? (yes/no): ").strip().lower()
        if confirmation == 'yes':
            return True
        elif confirmation == 'no':
            return False
        else:
            print("Please respond with 'yes' or 'no'.")

# Цикл для получения вопросов от пользователя
print("Enter your questions. Type 'stop' to end the session.")
while True:
    user_question = input("Your question: ").strip()
    if user_question.lower() in ['stop', 'стоп']:
        break
    try:
        attempts = 2
        while attempts > 0:
            user_question_vec = vectorizer.transform([user_question])
            prediction = classifier.predict(user_question_vec)[0]

            if get_clarification(prediction):
                print("Glad I could help!")
                break
            else:
                attempts -= 1
                if attempts == 0:
                    print("Перевожу на оператора.")
                else:
                    user_question = input("Could you please rephrase your question or provide more details? ").strip()
    except Exception as e:
        print(f"An error occurred while processing the question: {e}")
