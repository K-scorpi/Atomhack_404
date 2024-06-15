from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm  # Импортируем tqdm для прогресс-бара

class QuestionAnswerModel:
    def __init__(self):
        self.questions = []
        self.answers = []
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression()

    def load_data(self, filename):
        """Считывает данные из файла и добавляет их в модель."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Чтение данных", unit="строк"):  # Прогресс-бар при чтении
                    if '@' in line:  # Проверка наличия @ в строке
                        parts = line.strip().rsplit('@', 1)  # Разделение по последнему @
                        if len(parts) == 2:  # Проверка на корректный формат данных
                            self.questions.append(parts[0])
                            self.answers.append(parts[1])
                        else:
                            print(f"Ошибка в строке: {line} - Неверный формат данных")  # Выводим сообщение об ошибке
                    else:
                        print(f"Строка без @: {line} - удалена")  # Выводим сообщение о удаленной строке
        except FileNotFoundError:
            print(f"Ошибка: Файл {filename} не найден.")
        except UnicodeDecodeError:
            print(f"Ошибка: Проблема с кодировкой в файле {filename}.")
        except Exception as e:
            print(f"Ошибка при чтении файла: {e}")  # Выводим общий тип ошибки

    def train(self):
        """Обучает модель на имеющихся данных."""
        X = self.vectorizer.fit_transform(self.questions)
        y = self.answers
        # Имитация многошагового процесса
        for i in tqdm(range(10), desc="Обучение модели", unit="%"):
            self.model.fit(X, y) 
            # (В данном случае фактическое обучение происходит только в первом шаге)

    def predict(self, question):
        """Предсказывает ответ на заданный вопрос."""
        question_vector = self.vectorizer.transform([question])
        prediction = self.model.predict(question_vector)[0]
        return prediction

if __name__ == '__main__':
    model = QuestionAnswerModel()

    # Загрузка данных из файла
    model.load_data('output2.txt')

    # Обучение модели
    model.train()

    # Ввод новых вопросов в цикле
    while True:
        new_question = input("Введите вопрос: ")
        if new_question.lower() in ('стоп', 'stop'):
            break
        # Предсказание ответа
        answer = model.predict(new_question)
        print("Ответ:", answer)