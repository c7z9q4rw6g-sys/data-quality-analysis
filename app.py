import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
import plotly.express as px
import plotly.graph_objects as go

def get_engine(db_type, host, port, database, user, password):
    """Создаёт подключение к СУБД через SQLAlchemy."""
    if db_type == "PostgreSQL":
        url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    elif db_type == "MySQL":
        url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    elif db_type == "SQLite":
        url = f"sqlite:///{database}"
    else:
        st.error("Неподдерживаемый тип СУБД")
        return None
    return create_engine(url)

def load_table(engine, table_name, limit=10000):
    """Загружает данные из таблицы в DataFrame (с ограничением по строкам)."""
    query = f"SELECT * FROM {table_name} LIMIT {limit}"
    return pd.read_sql(query, engine)

class Preprocessor:
    """Класс для предобработки данных: очистка строк, кодирование категорий, нормализация."""

    @staticmethod
    def clean_strings(df):
        """Очищает строковые столбцы: удаляет пробелы, приводит к нижнему регистру."""
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
        return df

    @staticmethod
    def encode_categorical(df):
        """Кодирует категориальные признаки в числовые метки (LabelEncoder)."""
        le = LabelEncoder()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = le.fit_transform(df[col].astype(str))
        return df

    @staticmethod
    def normalize_numeric(df):
        """Нормализует числовые признаки (StandardScaler)."""
        scaler = StandardScaler()
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            df[num_cols] = scaler.fit_transform(df[num_cols])
        return df

class DuplicateDetector:
    """Упрощённый детектор дубликатов на основе хеширования текстовых полей."""

    @staticmethod
    def detect_duplicates(df, threshold=0.85):
        """
        Обнаруживает дубликаты и возвращает DataFrame с колонкой duplicate_group,
        а также список групп дубликатов.
        """
        dupe_groups = []
        df['duplicate_group'] = -1
        group_id = 0
        processed = set()
        text_cols = df.select_dtypes(include=['object']).columns
        if len(text_cols) == 0:
            return df, []
        df['_key'] = df[text_cols].astype(str).sum(axis=1).str[:50]
        for idx, row in df.iterrows():
            if idx in processed:
                continue
            key = row['_key']
            similar = df[df['_key'].str.contains(key[:10], na=False)].index.tolist()
            if len(similar) > 1:
                dupe_groups.append(similar)
                for i in similar:
                    df.at[i, 'duplicate_group'] = group_id
                    processed.add(i)
                group_id += 1
        df.drop('_key', axis=1, inplace=True)
        return df, dupe_groups

class OutlierDetector:
    """Класс для обнаружения аномалий методами Isolation Forest и LOF."""

    @staticmethod
    def detect_outliers_iforest(df, contamination=0.05):
        """Обнаружение аномалий с помощью Isolation Forest."""
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            df['is_outlier'] = 0
            return df
        X = df[num_cols].fillna(0)
        clf = IForest(contamination=contamination, random_state=42)
        clf.fit(X)
        df['is_outlier'] = clf.labels_
        df['outlier_score'] = clf.decision_scores_
        return df

    @staticmethod
    def detect_outliers_lof(df, contamination=0.05, n_neighbors=20):
        """Обнаружение аномалий с помощью Local Outlier Factor."""
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            df['is_outlier'] = 0
            return df
        X = df[num_cols].fillna(0)
        clf = LOF(contamination=contamination, n_neighbors=n_neighbors)
        clf.fit(X)
        df['is_outlier'] = clf.labels_
        df['outlier_score'] = clf.decision_scores_
        return df

class MissingHandler:
    """Класс для заполнения пропусков (простое, kNN)."""

    @staticmethod
    def simple_impute(df, strategy='mean'):
        """Заполнение пропусков средним, медианой или модой."""
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(include=['object']).columns
        imp_num = SimpleImputer(strategy=strategy)
        if len(num_cols) > 0:
            df[num_cols] = imp_num.fit_transform(df[num_cols])
        imp_cat = SimpleImputer(strategy='most_frequent')
        if len(cat_cols) > 0:
            df[cat_cols] = imp_cat.fit_transform(df[cat_cols])
        return df

    @staticmethod
    def knn_impute(df, k=5):
        """Заполнение пропусков методом k ближайших соседей (требует fancyimpute)."""
        try:
            from fancyimpute import KNN
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) == 0:
                return df
            X = df[num_cols].values
            X_filled = KNN(k=k).fit_transform(X)
            df[num_cols] = X_filled
        except ImportError:
            st.warning("fancyimpute не установлен, используем простое заполнение медианой")
            df = MissingHandler.simple_impute(df, strategy='median')
        return df

def generate_report(df, dupe_groups, outlier_method):
    """Формирует итоговую таблицу с колонкой problem_type (перечисление проблем)."""
    report_df = df.copy()
    report_df['problem_type'] = ''
    if 'duplicate_group' in report_df.columns:
        report_df.loc[report_df['duplicate_group'] != -1, 'problem_type'] += 'дубликат;'
    if 'is_outlier' in report_df.columns:
        report_df.loc[report_df['is_outlier'] == -1, 'problem_type'] += 'аномалия;'
    missing_mask = report_df.isnull().any(axis=1)
    report_df.loc[missing_mask, 'problem_type'] += 'пропуск;'
    report_df['problem_type'] = report_df['problem_type'].str.rstrip(';')
    return report_df

def plot_outliers(df, col):
    """Возвращает интерактивную гистограмму с выделением аномалий для выбранного столбца."""
    if col not in df.columns or 'is_outlier' not in df.columns:
        return go.Figure()
    fig = px.histogram(df, x=col, color='is_outlier', nbins=30,
                       title=f'Распределение {col} с выделением аномалий',
                       color_discrete_map={-1: 'red', 0: 'blue'})
    return fig

def plot_missing_heatmap(df):
    """Возвращает тепловую карту пропусков."""
    missing = df.isnull().astype(int)
    fig = px.imshow(missing.T, title='Тепловая карта пропусков (жёлтый = пропуск)',
                    color_continuous_scale='Viridis', aspect='auto')
    return fig

def main():
    """Главная функция приложения Streamlit."""
    st.set_page_config(page_title="Анализ качества данных", layout="wide")
    st.title("Интеллектуальная система анализа качества данных")

    with st.sidebar:
        st.header("Подключение к БД")
        db_type = st.selectbox("Тип СУБД", ["PostgreSQL", "MySQL", "SQLite"])
        host = st.text_input("Хост", "localhost")
        port = st.text_input("Порт", "5432" if db_type=="PostgreSQL" else "3306")
        database = st.text_input("Имя базы данных")
        user = st.text_input("Пользователь")
        password = st.text_input("Пароль", type="password")
        if st.button("Подключиться"):
            engine = get_engine(db_type, host, port, database, user, password)
            if engine:
                st.session_state['engine'] = engine
                if db_type == "SQLite":
                    query = "SELECT name FROM sqlite_master WHERE type='table'"
                else:
                    query = "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
                try:
                    tables = pd.read_sql(query, engine)
                    st.session_state['tables'] = tables.iloc[:,0].tolist()
                    st.success("Подключено!")
                except Exception as e:
                    st.error(f"Ошибка: {e}")

    if 'engine' in st.session_state:
        engine = st.session_state['engine']
        tables = st.session_state.get('tables', [])
        if tables:
            selected_table = st.selectbox("Выберите таблицу", tables)
            if st.button("Загрузить данные"):
                df = load_table(engine, selected_table)
                st.session_state['df_raw'] = df
                st.success(f"Загружено {len(df)} записей")

    if 'df_raw' in st.session_state:
        df = st.session_state['df_raw'].copy()
        st.subheader("Исходные данные (первые 10 строк)")
        st.dataframe(df.head(10))

        tab1, tab2, tab3 = st.tabs(["Настройка анализа", "Результаты", "Отчёт"])

        with tab1:
            st.header("Настройка параметров")
            col1, col2 = st.columns(2)
            with col1:
                detect_duplicates = st.checkbox("Обнаружить дубликаты", value=True)
                detect_outliers = st.checkbox("Обнаружить аномалии", value=True)
                outlier_method = st.selectbox("Метод аномалий", ["Isolation Forest", "LOF"])
                contamination = st.slider("Доля выбросов (contamination)", 0.01, 0.2, 0.05)
            with col2:
                handle_missing = st.checkbox("Обработать пропуски", value=True)
                missing_method = st.selectbox("Метод заполнения пропусков",
                                              ["Простое (среднее/мода)", "kNN (k=5)"])
            run = st.button("Запустить анализ")

        if run:
            df_proc = df.copy()
            prep = Preprocessor()
            df_proc = prep.clean_strings(df_proc)
            df_proc_encoded = prep.encode_categorical(df_proc.copy())
            df_proc_numeric = prep.normalize_numeric(df_proc_encoded)

            dupe_groups = []
            if detect_duplicates:
                df_proc_numeric, dupe_groups = DuplicateDetector.detect_duplicates(df_proc_numeric)
                st.session_state['dupe_groups'] = dupe_groups

            if detect_outliers:
                if outlier_method == "Isolation Forest":
                    df_proc_numeric = OutlierDetector.detect_outliers_iforest(df_proc_numeric, contamination)
                else:
                    df_proc_numeric = OutlierDetector.detect_outliers_lof(df_proc_numeric, contamination)

            if handle_missing:
                if missing_method == "Простое (среднее/мода)":
                    df_proc_numeric = MissingHandler.simple_impute(df_proc_numeric)
                else:
                    df_proc_numeric = MissingHandler.knn_impute(df_proc_numeric)

            st.session_state['df_processed'] = df_proc_numeric
            st.session_state['analysis_complete'] = True
            st.success("Анализ завершён! Перейдите на вкладку «Результаты».")

        with tab2:
            if st.session_state.get('analysis_complete', False):
                df_res = st.session_state['df_processed']
                dupe_groups = st.session_state.get('dupe_groups', [])
                report_df = generate_report(df_res, dupe_groups, outlier_method)
                st.subheader("Таблица с выявленными проблемами")
                st.dataframe(report_df)

                col_plot = st.selectbox("Выберите числовой столбец для гистограммы аномалий",
                                        df_res.select_dtypes(include=[np.number]).columns)
                if col_plot:
                    fig = plot_outliers(df_res, col_plot)
                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("Тепловая карта пропусков (исходные данные)")
                fig_miss = plot_missing_heatmap(df)
                st.plotly_chart(fig_miss, use_container_width=True)

                st.subheader("Статистика")
                total = len(df_res)
                n_duplicates = len(dupe_groups) if dupe_groups else 0
                n_outliers = (df_res['is_outlier'] == -1).sum() if 'is_outlier' in df_res else 0
                n_missing = df.isnull().any(axis=1).sum()
                st.metric("Всего записей", total)
                st.metric("Групп дубликатов", n_duplicates)
                st.metric("Аномалий", n_outliers)
                st.metric("Строк с пропусками (исходно)", n_missing)
            else:
                st.info("Запустите анализ на вкладке «Настройка анализа»")

        with tab3:
            if st.session_state.get('analysis_complete', False):
                df_res = st.session_state['df_processed']
                csv = df_res.to_csv(index=False).encode('utf-8')
                st.download_button("Скачать отчёт (CSV)", csv, "report.csv", "text/csv")
                st.markdown("### HTML-отчёт")
                st.write("Для сохранения полного отчёта используйте CSV и скриншоты графиков.")
            else:
                st.info("Сначала выполните анализ")

if __name__ == "__main__":
   main()
