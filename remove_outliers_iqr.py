def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 3 * IQR
    upper_limit = Q3 + 3 * IQR

    # Створюємо маску для викидів
    outlier_mask = (df[column] < lower_limit) | (df[column] > upper_limit)

    # Видаляємо всі рядки, де є аномалії
    df = df[~outlier_mask]

    return df