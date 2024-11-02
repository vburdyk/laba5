import os.path
import pandas as pd
import warnings

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from remove_outliers_iqr import remove_outliers_iqr
from model_training import model_training

# Ігноруємо незначні попередження
warnings.simplefilter('ignore')

# Перевіряємо, чи існує файл даних
print(os.path.exists("data/variant_1.csv"))
df = pd.read_csv("data/variant_1.csv")

# Стовпці, з яких будемо видаляти викиди
columns_to_process = ['income', 'loan_amount', 'Upfront_charges', 'property_value', 'LTV', 'rate_of_interest']

# Видаляємо викиди за кожним з вказаних стовпців
for column in columns_to_process:
    original_count = len(df)
    df = remove_outliers_iqr(df, column)
    removed_count = original_count - len(df)
    print(f"Removed {removed_count} outliers from {column}")

# Методи заповнення пропущених значень для кожного стовпця
fill_methods = {
    'rate_of_interest': 'mean',
    'Interest_rate_spread': 'mean',
    'Upfront_charges': 'mean',
    'term': 'mean',
    'property_value': 'mean',
    'income': 'mean',
    'LTV': 'mean',
    'loan_limit': 'mode',
    'approv_in_adv': 'mode',
    'loan_purpose': 'mode',
    'Neg_ammortization': 'mode',
    'age': 'mode',
    'submission_of_application': 'mode',
}

# Заповнюємо пропущені значення згідно з вибраними методами (mean або mode)
for column, method in fill_methods.items():
    if method == 'mean':
        df[column] = df[column].fillna(df[column].mean())
    elif method == 'mode':
        df[column] = df[column].fillna(df[column].mode()[0])

# Список категоріальних стовпців для перетворення
categorical_columns = [
    'loan_limit', 'Gender', 'approv_in_adv', 'loan_type', 'loan_purpose', 'Credit_Worthiness',
    'open_credit', 'business_or_commercial', 'Neg_ammortization', 'interest_only', 'lump_sum_payment',
    'construction_type', 'occupancy_type', 'Secured_by', 'total_units', 'credit_type',
    'co-applicant_credit_type', 'age', 'submission_of_application', 'Region', 'Security_Type'
]

# Кодуємо категоріальні стовпці з використанням LabelEncoder
le = LabelEncoder()
for column in categorical_columns:
    df[column] = le.fit_transform(df[column])

# Перетворюємо значення 'Status' на текстові ('yes' для 1 і 'no' для інших значень)
df['Status'] = df['Status'].apply(lambda x: 'yes' if x == 1 else 'no')

# Нормалізуємо числові стовпці
scaler_columns = ['loan_amount', 'income', 'Upfront_charges', 'property_value']
scaler = StandardScaler()
df[scaler_columns] = scaler.fit_transform(df[scaler_columns])

# Зберігаємо підготовлений набір даних у новий файл
df.to_csv('data/new_variant_1.csv', index=False)

# Завантажуємо оброблені дані для подальшого розділення
data = pd.read_csv('data/new_variant_1.csv')

# Розділяємо дані на train і new_input у співвідношенні 90:10
train, new_input = train_test_split(data, test_size=0.1, random_state=42)

# Зберігаємо розділені дані в окремі CSV файли
train.to_csv('data/train_split.csv', index=False)
new_input.to_csv('data/new_input.csv', index=False)

# Завантажуємо тренувальні дані та запускаємо функцію навчання моделі
df = pd.read_csv('data/train_split.csv')
model_training(df)
