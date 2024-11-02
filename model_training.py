import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


def model_training(df):
    # Вибір змінних для X та y
    X = df[['loan_limit', 'Gender', 'approv_in_adv', 'loan_type', 'loan_purpose', 'Credit_Worthiness', 'open_credit',
            'business_or_commercial', 'loan_amount', 'rate_of_interest', 'Interest_rate_spread', 'Upfront_charges', 'term',
            'Neg_ammortization', 'interest_only', 'lump_sum_payment', 'property_value', 'construction_type',
            'occupancy_type', 'Secured_by', 'total_units', 'income', 'credit_type', 'Credit_Score', 'co-applicant_credit_type',
            'age', 'submission_of_application', 'LTV', 'Region', 'Security_Type']]
    y = df['Status']

    # Розділення даних на тренувальний, валідаційний і тестовий набори
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42)

    # Ініціалізація і тренування моделі
    model = LogisticRegression(C=1, class_weight='balanced', max_iter=200, penalty='l1', solver='liblinear')
    print('Training model')
    model.fit(X_train, y_train)

    # Оцінка моделі на тестових даних
    y_pred_test = model.predict(X_test)
    print('Test set metrics:\n', metrics.classification_report(y_test, y_pred_test))

    # Оцінка моделі на валідаційних даних
    y_pred_valid = model.predict(X_valid)
    print('Validation set metrics:\n', metrics.classification_report(y_valid, y_pred_valid))

    # Збереження моделі
    joblib.dump(model, 'logistic_regression.pkl')

