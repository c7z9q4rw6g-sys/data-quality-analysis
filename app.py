import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pyod.models.iforest import IForest
import plotly.express as px
import plotly.graph_objects as go

NO_DUPLICATE_GROUP = -1
NORMAL_LABEL = 0
OUTLIER_LABEL = 1
INTERNAL_ANALYSIS_COLUMNS = {'duplicate_group', 'is_outlier', 'outlier_score'}

# ----------------------------------------------------------------------
# Подключение к БД
# ----------------------------------------------------------------------
def get_engine(db_type, host, port, database, user, password):
    try:
        if db_type == "PostgreSQL":
            url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        elif db_type == "MySQL":
            url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        elif db_type == "SQLite":
            url = f"sqlite:///{database}"
        else:
            st.error("Неподдерживаемый тип СУБД")
            return None
        engine = create_engine(url)
        with engine.connect() as conn:
            pass
        return engine
    except Exception as e:
        st.error(f"Ошибка подключения: {e}")
        return None

def load_table(engine, table_name, limit=50000):
    try:
        query = f'SELECT * FROM "{table_name}" LIMIT {limit}'
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки таблицы: {e}")
        return None

def load_table_columns(engine, table_name):
    try:
        inspector = inspect(engine)
        return [col['name'] for col in inspector.get_columns(table_name)]
    except Exception as e:
        st.error(f"Ошибка получения колонок таблицы: {e}")
        return []

# ----------------------------------------------------------------------
# Предобработка
# ----------------------------------------------------------------------
class Preprocessor:
    @staticmethod
    def clean_strings(df):
        """Очистка строковых колонок для поиска дубликатов"""
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
        return df

    @staticmethod
    def safe_encode_categorical(df):
        """Кодирование категорий с обработкой пропусков"""
        df_encoded = df.copy()
        for col in df_encoded.select_dtypes(include=['object']).columns:
            df_encoded[col] = df_encoded[col].fillna('missing').astype(str)
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
        return df_encoded

# ----------------------------------------------------------------------
# Обнаружение дубликатов (по всем колонкам, исключая автоинкрементные)
# ----------------------------------------------------------------------
def detect_duplicates(df, exclude_cols=None):
    """
    Ищет полные дубликаты строк, игнорируя указанные колонки (например, id, timestamp).
    Возвращает датафрейм с меткой 'duplicate_group' и список групп.
    """
    if exclude_cols is None:
        exclude_cols = []
    # колонки для сравнения – все, кроме исключённых
    compare_cols = [c for c in df.columns if c not in exclude_cols]
    if not compare_cols:
        return df, []
    
    # Очищаем строки для корректного сравнения
    df_clean = Preprocessor.clean_strings(df.copy())
    
    # Находим дубликаты
    df_clean['duplicate_group'] = NO_DUPLICATE_GROUP
    dupe_groups = []
    # Сначала все строки, у которых есть дубликаты
    dup_mask = df_clean[compare_cols].duplicated(keep=False)
    dup_df = df_clean[dup_mask]
    if not dup_df.empty:
        grouped = dup_df.groupby(compare_cols, dropna=False, sort=False).groups
        group_id = 0
        for key, indices in grouped.items():
            if len(indices) > 1:
                dupe_groups.append(list(indices))
                for idx in indices:
                    df_clean.at[idx, 'duplicate_group'] = group_id
                group_id += 1
    # Переносим метки обратно в исходный df
    df['duplicate_group'] = df_clean['duplicate_group']
    return df, dupe_groups

# ----------------------------------------------------------------------
# Обнаружение аномалий (только числовые колонки, Isolation Forest)
# ----------------------------------------------------------------------
def detect_outliers_numeric(df, contamination=0.1):
    """Обнаружение аномалий только по числовым колонкам"""
    num_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in INTERNAL_ANALYSIS_COLUMNS
    ]
    if len(num_cols) == 0:
        df['is_outlier'] = NORMAL_LABEL
        df['outlier_score'] = 0.0
        return df
    # Заполняем пропуски медианой для числовых (чтобы не искажать)
    X = df[num_cols].copy()
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    # Нормализуем для Isolation Forest
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    # Модель
    clf = IForest(contamination=contamination, random_state=42, n_estimators=100)
    clf.fit(X_scaled)
    df['is_outlier'] = clf.labels_   # 1 = аномалия, 0 = норма (формат PyOD)
    df['outlier_score'] = clf.decision_scores_
    return df

# ----------------------------------------------------------------------
# Заполнение пропусков
# ----------------------------------------------------------------------
def impute_missing(df):
    """Заполнение пропусков: числовые – медианой, категории – модой"""
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(num_cols) > 0:
        imp_num = SimpleImputer(strategy='median')
        df[num_cols] = imp_num.fit_transform(df[num_cols])
    if len(cat_cols) > 0:
        imp_cat = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = imp_cat.fit_transform(df[cat_cols])
    return df

# ----------------------------------------------------------------------
# Отчёты и визуализация
# ----------------------------------------------------------------------
def get_duplicate_mask(df):
    if 'duplicate_group' not in df.columns:
        return pd.Series(False, index=df.index)
    return df['duplicate_group'] != NO_DUPLICATE_GROUP

def get_outlier_mask(df):
    if 'is_outlier' not in df.columns:
        return pd.Series(False, index=df.index)
    return df['is_outlier'] == OUTLIER_LABEL

def get_missing_mask(df):
    return df.isnull().any(axis=1)

def generate_report(df, dupe_groups=None, original_df=None):
    report = df.copy()
    report['problem_type'] = ''
    duplicate_mask = get_duplicate_mask(report)
    outlier_mask = get_outlier_mask(report)
    missing_mask = get_missing_mask(original_df if original_df is not None else report)

    report.loc[duplicate_mask, 'problem_type'] += 'дубликат;'
    report.loc[outlier_mask, 'problem_type'] += 'аномалия;'
    report.loc[missing_mask.reindex(report.index, fill_value=False), 'problem_type'] += 'пропуск;'
    report['problem_type'] = report['problem_type'].str.rstrip(';')
    report.loc[report['problem_type'] == '', 'problem_type'] = 'нет проблем'
    return report

def plot_outliers_histogram(df, col):
    if col not in df.columns or 'is_outlier' not in df.columns:
        return go.Figure()
    fig = px.histogram(df, x=col, color='is_outlier', nbins=40,
                       title=f'Распределение {col} (красное – аномалии)',
                       color_discrete_map={OUTLIER_LABEL: 'red', NORMAL_LABEL: 'blue'})
    return fig

def plot_missing_heatmap(df):
    missing = df.isnull().astype(int)
    if missing.sum().sum() == 0:
        return None
    fig = px.imshow(missing.T, title='Тепловая карта пропусков (жёлтый = пропуск)',
                    color_continuous_scale='Viridis', aspect='auto')
    return fig

# ----------------------------------------------------------------------
# Главное приложение
# ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Анализ качества данных", layout="wide")
    st.title("📊 Интеллектуальная система анализа качества данных")

    # --- Боковая панель ---
    with st.sidebar:
        st.header("🔌 Подключение к БД")
        db_type = st.selectbox("Тип СУБД", ["SQLite", "PostgreSQL", "MySQL"])
        if db_type == "SQLite":
            host = ""
            port = ""
            database = st.text_input("Имя файла .db", "database.db")
            user = ""
            password = ""
        else:
            host = st.text_input("Хост", "localhost")
            port = st.text_input("Порт", "5432" if db_type=="PostgreSQL" else "3306")
            database = st.text_input("Имя базы данных")
            user = st.text_input("Пользователь")
            password = st.text_input("Пароль", type="password")

        if st.button("Подключиться"):
            engine = get_engine(db_type, host, port, database, user, password)
            if engine:
                st.session_state['engine'] = engine
                try:
                    if db_type == "SQLite":
                        query = "SELECT name FROM sqlite_master WHERE type='table'"
                    else:
                        query = "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
                    tables_df = pd.read_sql(query, engine)
                    st.session_state['tables'] = tables_df.iloc[:,0].tolist()
                    st.success("Подключено!")
                except Exception as e:
                    st.error(f"Ошибка получения таблиц: {e}")

    # --- Выбор таблицы ---
    if 'engine' in st.session_state and 'tables' in st.session_state:
        selected_table = st.selectbox("📁 Выберите таблицу", st.session_state['tables'])
        table_columns = load_table_columns(st.session_state['engine'], selected_table)
        exclude_cols = st.multiselect("Колонки, которые НЕ учитывать при поиске дубликатов (например, id, timestamp)",
                                      table_columns,
                                      default=[])
        if st.button("Загрузить данные"):
            df_raw = load_table(st.session_state['engine'], selected_table, limit=50000)
            if df_raw is not None:
                st.session_state['df_raw'] = df_raw
                st.session_state['exclude_cols'] = exclude_cols
                st.session_state['selected_table'] = selected_table
                st.success(f"Загружено {len(df_raw)} записей")

    # --- Анализ ---
    if 'df_raw' in st.session_state:
        df_raw = st.session_state['df_raw']
        exclude_cols = st.session_state.get('exclude_cols', [])

        st.subheader("📋 Исходные данные (первые 10 строк)")
        st.dataframe(df_raw.head(10))

        # Краткая диагностика перед анализом
        col1, col2, col3 = st.columns(3)
        col1.metric("Всего строк", len(df_raw))
        col2.metric("Столбцов", len(df_raw.columns))
        col3.metric("Строк с пропусками (исходно)", df_raw.isnull().any(axis=1).sum())

        tab1, tab2, tab3 = st.tabs(["⚙️ Настройки", "📈 Результаты", "📄 Экспорт"])

        with tab1:
            st.header("Параметры анализа")
            c1, c2 = st.columns(2)
            with c1:
                detect_dup = st.checkbox("Обнаружить дубликаты", value=True)
                dup_info = st.info("Будут найдены полные дубликаты по всем колонкам, кроме исключённых.")
            with c2:
                detect_out = st.checkbox("Обнаружить аномалии", value=True)
                contamination = st.slider("Доля выбросов (contamination)", 0.01, 0.5, 0.1, 0.01)
                st.info("Рекомендуется 0.05–0.1 для реальных данных.")
            handle_miss = st.checkbox("Заполнить пропуски (числовые – медианой, категории – модой)", value=True)

            if st.button("🚀 ЗАПУСТИТЬ АНАЛИЗ", type="primary"):
                progress = st.progress(0)
                status = st.empty()
                df_work = df_raw.copy()

                # 1. Дубликаты
                if detect_dup:
                    status.text("Поиск дубликатов...")
                    df_work, dupe_groups = detect_duplicates(df_work, exclude_cols=exclude_cols)
                    st.session_state['dupe_groups'] = dupe_groups
                else:
                    df_work['duplicate_group'] = NO_DUPLICATE_GROUP
                    st.session_state['dupe_groups'] = []
                progress.progress(25)

                # 2. Подготовка для аномалий (кодирование категорий для числовой модели? лучше не нужно – аномалии только по числам)
                # Для аномалий используем только числовые колонки, поэтому кодирование не требуется
                # Но нам нужны исходные числовые значения
                if detect_out:
                    status.text("Поиск аномалий (только числовые колонки)...")
                    df_work = detect_outliers_numeric(df_work, contamination=contamination)
                else:
                    df_work['is_outlier'] = NORMAL_LABEL
                    df_work['outlier_score'] = 0.0
                progress.progress(50)

                # 3. Заполнение пропусков (опционально)
                if handle_miss:
                    status.text("Заполнение пропусков...")
                    df_work = impute_missing(df_work)
                progress.progress(75)

                # Сохраняем
                st.session_state['df_processed'] = df_work
                st.session_state['analysis_complete'] = True
                status.text("Анализ завершён!")
                progress.progress(100)
                st.success("Перейдите на вкладку «Результаты»")

        with tab2:
            if st.session_state.get('analysis_complete', False):
                df_res = st.session_state['df_processed']
                dupe_groups = st.session_state.get('dupe_groups', [])
                report_df = generate_report(df_res, dupe_groups, original_df=df_raw)

                st.subheader("🔍 Таблица с выявленными проблемами")
                st.dataframe(report_df.head(50))

                # Детальная статистика
                st.subheader("📊 Детальная статистика")
                d1, d2, d3, d4, d5 = st.columns(5)
                d1.metric("Всего записей", len(df_res))
                d2.metric("Групп дубликатов", len(dupe_groups))
                n_dup_rows = int(get_duplicate_mask(df_res).sum())
                d3.metric("Строк-дубликатов", n_dup_rows)
                n_out = int(get_outlier_mask(df_res).sum())
                n_miss = df_raw.isnull().any(axis=1).sum()
                d4.metric("Аномалий (Isolation Forest)", n_out)
                d5.metric("Строк с пропусками (исходно)", n_miss)

                # Список колонок с пропусками
                cols_with_missing = [c for c in df_raw.columns if df_raw[c].isnull().any()]
                if cols_with_missing:
                    st.write("**Колонки, содержащие пропуски:**", ", ".join(cols_with_missing))
                else:
                    st.success("Пропусков в данных нет!")

                # Графики
                st.divider()
                g1, g2 = st.columns(2)
                with g1:
                    st.subheader("📉 Гистограмма аномалий")
                    numeric_cols = df_res.select_dtypes(include=[np.number]).columns.tolist()
                    numeric_cols = [c for c in numeric_cols if c not in ['duplicate_group', 'outlier_score', 'is_outlier']]
                    if numeric_cols:
                        chosen = st.selectbox("Выберите числовой столбец", numeric_cols, key="hist")
                        fig = plot_outliers_histogram(df_res, chosen)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Нет числовых колонок для построения гистограммы.")

                with g2:
                    st.subheader("🌡️ Тепловая карта пропусков (исходные данные)")
                    fig_miss = plot_missing_heatmap(df_raw)
                    if fig_miss:
                        st.plotly_chart(fig_miss, use_container_width=True)
                    else:
                        st.success("Нет пропусков для отображения.")

                # Дополнительная диагностика: покажем несколько примеров аномалий и дубликатов
                with st.expander("🔎 Примеры найденных проблем"):
                    outlier_mask = get_outlier_mask(df_res)
                    if outlier_mask.any():
                        st.write("**Примеры аномальных записей:**")
                        st.dataframe(df_res[outlier_mask].head(10))
                    if dupe_groups:
                        st.write("**Примеры дубликатов (первые 3 группы):**")
                        for i, grp in enumerate(dupe_groups[:3]):
                            st.write(f"Группа {i+1}: {grp}")
                            st.dataframe(df_res.loc[grp])
            else:
                st.info("Сначала настройте параметры и запустите анализ.")

        with tab3:
            if st.session_state.get('analysis_complete', False):
                df_res = st.session_state['df_processed']
                report_df = generate_report(df_res, st.session_state.get('dupe_groups', []), original_df=df_raw)
                csv = report_df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Скачать отчёт (CSV)", csv, "quality_report.csv", "text/csv")
                st.markdown("Отчёт включает все исходные колонки + `duplicate_group`, `is_outlier`, `outlier_score`, `problem_type`.")
            else:
                st.info("Нет данных для экспорта. Запустите анализ.")

if __name__ == "__main__":
    main()
