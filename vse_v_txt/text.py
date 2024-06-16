import os
from transformers import pipeline

# Функция для чтения всех текстовых файлов в папке
def load_text_files(folder_path):
    all_texts = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                all_texts += file.read() + "\n"
    return all_texts

# Путь к папке с текстовыми файлами
folder_path = 'vse_v_txt'

# Загрузка и объединение всех текстов
all_texts = load_text_files(folder_path)

# Инициализация модели для вопрос-ответ
qa_model = pipeline("question-answering")

# Функция для ответа на вопросы
def answer_question(question, context):
    result = qa_model(question=question, context=context)
    return result['answer']

# Пример использования
question = "Введите ваш вопрос здесь"
answer = answer_question(question, all_texts)
print(answer)
