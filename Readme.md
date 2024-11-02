# Проєкт машинного навчання: Прогнозування схвалення кредитів

## Огляд проєкту
Цей проєкт спрямований на розробку моделі машинного навчання, що прогнозує статус схвалення кредиту на основі різних атрибутів клієнта та кредиту. Основні етапи включають очищування даних, їх попередню обробку, видалення викидів, кодування категоріальних ознак, масштабування ознак і навчання моделі логістичної регресії.

## Структура проєкту
- `data/variant_1.csv`: Файл з початковим набором даних.
- `data/new_variant_1.csv`: Підготовлений набір даних.
- `data/train_split.csv`: Навчальний набір даних після розділення.
- `data/new_input.csv`: Набір даних для тестування та передбачень.
- `requirements.txt`: Містить всі необхідні бібліотеки та залежності для проєкту.
- `remove_outliers_iqr.py`: Скрипт для видалення викидів за допомогою методу IQR.
- `model_training.py`: Скрипт для навчання моделі логістичної регресії та її оцінки.
- `main.py`: Головний скрипт для попередньої обробки даних, розділення на навчальний та тестовий набори та запуску навчання моделі.
- `predict.py`: Скрипт для завантаження навченого алгоритму та здійснення передбачень на нових даних.

## Вимоги
Встановіть необхідні пакети за допомогою:
```bash
pip install -r requirements.txt
```

## Використання
Запустіть `main.py` для завантаження набору даних, видалення викидів, заповнення пропущених значень, кодування категоріальних ознак, нормалізації числових ознак та розділення даних, а також запуску моделі для навчання.

