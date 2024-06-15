import pandas as pd

def extract_and_combine(file_path):
  """
  Извлекает значения из 6 и 7 колонок файла Excel, объединяет их в одну строку с разделителем "@" 
  и записывает в файл. Если в 6 или 7 колонке пустое значение, не записывает.

  Args:
    file_path (str): Путь к файлу Excel.

  Returns:
    None
  """

  df = pd.read_excel(file_path)

  with open("output.txt", "w") as f:
    for index, row in df.iterrows():
      value_6 = row[5]  # 6-я колонка
      value_7 = row[6]  # 7-я колонка

      if pd.notna(value_6) and pd.notna(value_7):
        combined_row = f"{value_6}@{value_7}"
        f.write(combined_row + "\n")

# Пример использования:
file_path = "data_work/21.xlsx"  # Замените на ваш путь к файлу
extract_and_combine(file_path)

print("Результат сохранен в output.txt")