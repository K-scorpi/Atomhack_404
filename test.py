import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_files(folder_path):
    """Загружает текстовые файлы из указанной папки."""
    file_contents = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                file_contents[filename] = f.read()
    return file_contents

def train_tfidf_model(file_contents):
    """Создает TF-IDF модель на основе содержимого файлов."""
    vectorizer = TfidfVectorizer()
    filenames = list(file_contents.keys())
    documents = list(file_contents.values())
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix, filenames

def search_query(query, vectorizer, tfidf_matrix, filenames):
    """Выполняет поиск запроса и возвращает наиболее релевантный результат."""
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    max_index = similarities.argmax()
    return filenames[max_index], similarities[max_index]

if __name__ == "__main__":
    # Путь к папке с текстовыми файлами
    folder_path = "vse_v_txt"

    # Загрузка текстовых файлов
    print("Загрузка текстовых файлов...")
    file_contents = load_files(folder_path)

    # Обучение модели
    print("Обучение модели TF-IDF...")
    vectorizer, tfidf_matrix, filenames = train_tfidf_model(file_contents)

    while True:
        # Получение запроса от пользователя
        query = input("Введите запрос (или 'выход' для завершения): ")
        if query.lower() == 'выход':
            break

        # Поиск
        filename, similarity = search_query(query, vectorizer, tfidf_matrix, filenames)

        # Вывод результата
        if similarity > 0:
            print(f"Наиболее релевантный файл: {filename} (похожесть: {similarity:.4f})")
            print("Содержимое файла:")
            print(file_contents[filename])
        else:
            print("Совпадений не найдено.")
