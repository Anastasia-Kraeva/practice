from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RANDOM_SEED: int = 678
SERIES_SIZE: int = 1000
RANDOM_LOW: int = -10000
RANDOM_HIGH: int = 10000
HIST_BINS: int = 30
PLOT_FIGSIZE_FIRST: Tuple[int, int] = (14, 5)
PLOT_FIGSIZE_SECOND: Tuple[int, int] = (10, 6)
DISPLAY_HEAD_LIMIT: int = 100
DATA_FILE_CSV: str = "dataset.csv"


def generate_random_series(
        n: int = SERIES_SIZE,
        low: int = RANDOM_LOW,
        high: int = RANDOM_HIGH,
        seed: int = RANDOM_SEED
) -> pd.Series:
    """
    Генерирует Series из случайных целых чисел

    :param n: Количество чисел
    :param low: Нижняя граница диапазона
    :param high: Верхняя граница диапазона
    :param seed: Зерно для генератора случайных чисел
    """
    if n <= 0:
        raise ValueError("Количество элементов должно быть положительным")
    if low >= high:
        raise ValueError("low должно быть меньше high")

    np.random.seed(seed)
    data = np.random.randint(low, high, size=n)
    return pd.Series(data, name="значение")


def save_dataset_to_csv(series: pd.Series, csv_path: str) -> None:
    """
    Сохраняет Dataset в формате *.csv.

    :param series: Исходная серия данных
    :param csv_path: Путь для сохранения файла
    """
    series.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Данные сохранены в файл: {csv_path}")


def load_dataset_from_csv(csv_path: str) -> pd.Series:
    """
    Загружает Dataset из файла

    :param csv_path: Путь к файлу
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    return df["значение"]


def clean_data(series: pd.Series) -> pd.Series:
    """
    Выполняет очистку данных от "цифрового мусора"

    :param series: Исходная серия данных
    """
    cleaned = series.copy()

    before_nan = cleaned.isna().sum()
    cleaned = cleaned.dropna()

    before_duplicates = cleaned.duplicated().sum()
    cleaned = cleaned.drop_duplicates()

    before_outliers = ((cleaned < RANDOM_LOW) | (cleaned > RANDOM_HIGH)).sum()
    cleaned = cleaned[(cleaned >= RANDOM_LOW) & (cleaned <= RANDOM_HIGH)]

    print(f"Удалено пропущенных значений: {before_nan}")
    print(f"Удалено дубликатов: {before_duplicates}")
    print(f"Удалено выбросов: {before_outliers}")
    print(f"Исходный размер: {len(series)}")
    print(f"Размер после очистки: {len(cleaned)}")

    return cleaned


def calculate_statistics(series: pd.Series) -> Dict[str, float]:
    """
    Рассчитывает стандартные числовые характеристики для набора данных

    :param series: Объект pd.Series с числовыми данными
    """
    return {
        "min": series.min(),
        "duplicate_count": series.duplicated().sum(),
        "max": series.max(),
        "sum": series.sum(),
        "std": series.std(ddof=1),
    }


def print_statistics(stats: Dict[str, float]) -> None:
    """
    Выводит статистические характеристики в консоль с пояснениями

    :param stats: Словарь со статистическими характеристиками
    """
    print("\n" + "-" * 40)
    print("Стандартные числовые характеристики")
    print(f"- Минимальное значение: {stats['min']}")
    print(f"- Количество повторяющихся значений: {stats['duplicate_count']}")
    print(f"- Максимальное значение: {stats['max']}")
    print(f"- Сумма всех чисел: {stats['sum']}")
    print(f"- Среднеквадратическое отклонение: {stats['std']:.4f}")


def print_dataframe_summary(df: pd.DataFrame) -> None:
    """
    Выводит сводную информацию о DataFrame

    :param df: DataFrame для анализа
    """
    print("\n" + "-" * 40)
    print("Сводная информация о dataframe")
    print(f"Количество строк: {len(df)}")
    print(f"Количество столбцов: {len(df.columns)}")
    print(f"\nТипы данных:\n{df.dtypes}")
    print(df.describe())
    print("\n")


def round_to_hundreds(values: pd.Series) -> np.ndarray:
    """
    Округляет значения до сотен по математическому правилу

    :param values: Исходная серия данных
    """
    return np.around(values / 100) * 100


def plot_line_chart(series: pd.Series, limit: int = DISPLAY_HEAD_LIMIT) -> None:
    """
    Строит линейный график

    :param series: Исходная серия данных
    :param limit: Количество отображаемых первых элементов
    """
    plt.subplot(1, 2, 1)
    plt.plot(series.head(limit).values, color="blue", linewidth=1)
    plt.title(f"Линейный график (первые {limit} значений)")
    plt.xlabel("Индекс")
    plt.ylabel("Значение")
    plt.grid(True, alpha=0.3)


def plot_histogram(series: pd.Series, bins: int = HIST_BINS) -> None:
    """
    Строит гистограмму с округлением значений до сотен

    :param series: Исходная серия данных
    :param bins: Количество бинов гистограммы
    """
    rounded_data = round_to_hundreds(series)

    plt.subplot(1, 2, 2)
    plt.hist(rounded_data, bins=bins, color="green", alpha=0.7, edgecolor="black")
    plt.title("Гистограмма (данные округлены до сотен)")
    plt.xlabel("Значение (округленное)")
    plt.ylabel("Частота")
    plt.grid(True, alpha=0.3)


def show_initial_plots(series: pd.Series) -> None:
    """Отображает линейный график и гистограмму"""
    plt.figure(figsize=PLOT_FIGSIZE_FIRST)
    plot_line_chart(series)
    plot_histogram(series)
    plt.tight_layout()
    plt.show()


def create_dataframe_with_sorted_columns(series: pd.Series) -> pd.DataFrame:
    """
    Создает DataFrame из Series и добавляет столбцы

    :param series: Исходная серия данных
    """
    df = pd.DataFrame({"Исходные_данные": series})
    df["Сортировка_по_возрастанию"] = series.sort_values().reset_index(drop=True)
    df["Сортировка_по_убыванию"] = series.sort_values(ascending=False).reset_index(drop=True)
    return df


def plot_sorted_comparison(series: pd.Series, limit: int = DISPLAY_HEAD_LIMIT) -> None:
    """
    Строит два линейных графика: отсортированные значения по возрастанию и убыванию

    :param series: Исходная серия данных
    :param limit: Количество отображаемых первых элементов
    """
    asc_sorted = series.sort_values().reset_index(drop=True).head(limit)
    desc_sorted = series.sort_values(ascending=False).reset_index(drop=True).head(limit)

    plt.figure(figsize=PLOT_FIGSIZE_SECOND)
    plt.plot(asc_sorted.values, color="red", linewidth=1.5, label="По возрастанию")
    plt.plot(desc_sorted.values, color="purple", linewidth=1.5, label="По убыванию")
    plt.title(f"Сравнение отсортированных данных (первые {limit} значений)")
    plt.xlabel("Индекс")
    plt.ylabel("Значение")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Точка входа для запуска программы"""
    generated_data = generate_random_series()
    print(f"Сгенерировано {len(generated_data)} чисел")
    print(f"Первые 10 элементов:\n{generated_data.head(10)}")

    print("\n" + "-" * 40)
    print("Сохранение dataset в csv")
    save_dataset_to_csv(generated_data, DATA_FILE_CSV)

    print("\n" + "-" * 40)
    print("Загрузка данных из csv")
    loaded_data = load_dataset_from_csv(DATA_FILE_CSV)
    print(f"Загружено {len(loaded_data)} чисел из файла {DATA_FILE_CSV}")

    print("\n" + "-" * 40)
    print("Очистка от цифрового мусора")
    cleaned_data = clean_data(loaded_data)

    statistics = calculate_statistics(cleaned_data)
    print_statistics(statistics)

    df = create_dataframe_with_sorted_columns(cleaned_data)
    print_dataframe_summary(df)

    print("\n" + "-" * 40)
    print("Визуализация данных (линейный график и гистограмма)")
    show_initial_plots(cleaned_data)

    print("\n" + "-" * 40)
    print("Визуализация отсортированных данных (график сравнения)")
    plot_sorted_comparison(cleaned_data)


if __name__ == "__main__":
    main()