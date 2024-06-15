import pandas as pd

def extract_and_combine(file_path):
  """
  Извлекает значения из 6 и 8 колонок файла Excel, объединяет их в одну строку с разделителем "@" 
  и дублирует 6 колонку для каждой темы из 8 колонки, разделенных запятой.

  Args:
    file_path (str): Путь к файлу Excel.

  Returns:
    list: Список строк с объединенными значениями.
  """

  df = pd.read_excel(file_path)
  combined_rows = []

  for index, row in df.iterrows():
    value_6 = row[5]  # 6-я колонка
    topics = str(row[7]).split(',')  # 8-я колонка, преобразуем в строку и разделяем по запятой

    for topic in topics:
      combined_row = f"{value_6}@{topic.strip()}"
      combined_rows.append(combined_row)

  return combined_rows

# Пример использования:
file_path = "otvet.xlsx"  # Замените на ваш путь к файлу
combined_data = extract_and_combine(file_path)

# Сохранение результата в txt-файл
with open("output.txt", "w") as f:
  for row in combined_data:
    f.write(row + "\n")

print("Результат сохранен в output.txt")
# Все скопируется вместе с функцией