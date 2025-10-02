import json
import os 
path = os.path.dirname(__file__)
# Чтение файла
with open(os.path.join(path, 'reviews.json'), 'r', encoding='utf-8') as file:

    data = json.load(file)

# Извлечение нужных данных
extracted_data = []
for review in data:
    extracted_review = {
        "authorName": review.get("authorName"),
        "rating": review.get("rating"),
        "date": review.get("date"),
        "reviewTag": review.get("reviewTag"),
        "text": review.get("text")
    }
    extracted_data.append(extracted_review)

# Сохранение в новый JSON файл
with open(os.path.join(path,'extracted_reviews.json'), 'w', encoding='utf-8') as output_file:
    json.dump(extracted_data, output_file, ensure_ascii=False, indent=2)

print(f"Обработано {len(extracted_data)} отзывов. Данные сохранены в extracted_reviews.json")