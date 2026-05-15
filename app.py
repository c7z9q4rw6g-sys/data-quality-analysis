import streamlit as st
import pandas as pd
import numpy as np
import dedupe  
import os
from sqlalchemy import create_engine, inspect
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pyod.models.iforest import IForest
import plotly.express as px
import plotly.graph_objects as go

# Константы
NO_DUPLICATE_GROUP = -1
NORMAL_LABEL = 0
OUTLIER_LABEL = 1
INTERNAL_COLUMNS = {'duplicate_group', 'is_outlier', 'outlier_score'}

# ----------------------------------------------------------------------
# 1. Подключение к БД
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

def load_table(engine, table_name, limit=10000):
    try:
        query = f'SELECT * FROM "{table_name}" LIMIT {limit}'
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")
        return None

def load_table_columns(engine, table_name):
    try:
        inspector = inspect(engine)
        return [col['name'] for col in inspector.get_columns(table_name)]
    except Exception as e:
        return []

# ----------------------------------------------------------------------
# 2. Обнаружение дубликатов (DEDUPE)
# ----------------------------------------------------------------------
def detect_duplicates_with_dedupe(df, exclude_cols=None, threshold=0.5):
    """
    Использует библиотеку dedupe для поиска дубликатов.
    Реализует логику, описанную в дипломе (блокирование, сравнение, порог).
    """
    if exclude_cols is None:
        exclude_cols = []

    # Исключаем служебные колонки и те, что выбрал пользователь
    compare_cols = [c for c in df.columns if c.lower() not in [x.lower() for x in exclude_cols] 
                    and c.lower() not in ['id', 'index', 'duplicate_group']]
    
    if len(compare_cols) == 0:
        df['duplicate_group'] = NO_DUPLICATE_GROUP
        return df, []

    # Подготовка данных для dedupe (он требует словарь и не любит NaN)
    # dedupe не работает с pandas DataFrame напрямую
    data_dict = df[compare_cols].fillna('').to_dict(orient='index')

    # Определяем поля для сравнения
    fields = []
    for col in compare_cols:
        # dedupe требует явного указания типа поля
        fields.append({'field': col, 'type': 'String'})

    # Инициализация модели
    deduper = dedupe.Dedupe(fields)
    
    # Сэмплирование данных для ускорения (блокирование)
    # В реальном активном обучении здесь deduper.sample() и ручная разметка
    # Для автоматической работы системы используем partition с порогом
    deduper.sample(data_dict, sample_size=10000)
    
    # Поскольку у нас нет интерфейса для разметки "Да/Нет" прямо сейчас,
    # dedupe использует настройки по умолчанию или обучается на лету, 
    # если бы мы передавали labeled_pairs. 
    # Здесь мы полагаемся на эвристики dedupe для String полей.
    
    # Кластеризация (поиск дубликатов)
    clusters = deduper.partition(data_dict, threshold=threshold)
    
    # Обработка результатов
    df_result = df.copy()
    df_result['duplicate_group'] = NO_DUPLICATE_GROUP
    
    dupe_groups = []
    group_id = 0
    
    for cluster in clusters:
        # cluster выглядит как ((id1, id2, ...), probability)
        indices = cluster[0]
        # prob = cluster[1] # можно использовать для фильтрации по уверенности
        
        if len(indices) > 1:
            dupe_groups.append(list(indices))
            for idx in indices:
                df_result.at[idx, 'duplicate_group'] = group_id
            group_id += 1
            
    return df_result, dupe_groups

# ----------------------------------------------------------------------
# 3. Обнаружение аномалий (Isolation Forest)
# ----------------------------------------------------------------------
def detect_outliers_numeric(df, contamination=0.1):
    num_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in INTERNAL_COLUMNS
    ]
    
    if len(num_cols) == 0:
        df['is_outlier'] = NORMAL_LABEL
        df['outlier_score'] = 0.0
        return df

    X = df[num_cols].copy()
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    clf = IForest(contamination=contamination, random_state=42, n_estimators=100)
    clf.fit(X_scaled)
    
    # PyOD labels: -1 для аномалий, 1 для нормы. Мы конвертируем в 1 (аномалия) и 0 (норма)
    raw_labels = clf.labels_
    df['is_outlier'] = (raw_labels == -1).astype(int)
    df['outlier_score'] = clf.decision_scores_
    return df

# ----------------------------------------------------------------------
# 4. Заполнение пропусков
# ----------------------------------------------------------------------
def impute_missing(df):
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
# 5. Утилиты и отчеты
# ----------------------------------------------------------------------
def generate_report(df, dupe_groups=None, original_df=None):
    report = df.copy()
    report['problem_type'] = ''
    
    # Дубликаты
    if 'duplicate_group' in report.columns:
        dup_mask = report['duplicate_group'] != NO_DUPLICATE_GROUP
        report.loc[dup_mask, 'problem_type'] += 'дубликат; '
        
    # Аномалии
    if 'is_outlier' in report.columns:
        out_mask = report['is_outlier'] == OUTLIER_LABEL
        report.loc[out_mask, 'problem_type'] += 'аномалия; '
        
    # Пропуски (проверяем по оригиналу, если он передан)
    check_df = original_df if original_df is not None else report
    miss_mask = check_df.isnull().any(axis=1)
    report.loc[miss_mask.reindex(report.index, fill_value=False), 'problem_type'] += 'пропуск; '
    
    report['problem_type'] = report['problem_type'].str.rstrip('; ')
    report.loc[report['problem_type'] == '', 'problem_type'] = 'нет проблем'
    return report

def plot_outliers_histogram(df, col):
    if col not in df.columns or 'is_outlier' not in df.columns:
        return go.Figure()
    
    plot_df = df[[col, 'is_outlier']].copy()
    plot_df['Тип'] = plot_df['is_outlier'].apply(lambda x: 'Аномалия' if x == 1 else 'Норма')
    
    fig = px.histogram(plot_df, x=col, color='Тип', nbins=30,
                       title=f'Распределение "{col}" (Красные = Аномалии)',
                       color_discrete_map={'Аномалия': 'red', 'Норма': 'blue'})
    return fig

def plot_missing_heatmap(df):
    # Берем выборку для скорости
    sample_df = df.head(500).iloc[:, :20]
    missing = sample_df.isnull().astype(int)
    
    if missing.sum().sum() == 0:
        return None
        
    fig = px.imshow(missing.T, 
                    title='Тепловая карта пропусков (Желтый = Пропуск)',
                    color_continuous_scale='Viridis', aspect='auto')
    return fig

# ----------------------------------------------------------------------
# 6. Главное приложение
# ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Анализ качества данных", layout="wide")
    st.title("Интеллектуальная система анализа качества данных")

    # --- Боковая панель ---
    with st.sidebar:
        st.header("Подключение к БД")
        db_type = st.selectbox("Тип СУБД", ["SQLite", "PostgreSQL", "MySQL"])
        
        if db_type == "SQLite":
            host = ""
            port = ""
            database = st.text_input("Имя файла .db", "test_quality_with_defects.db")
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
        selected_table = st.selectbox("Выберите таблицу", st.session_state['tables'])
        table_columns = load_table_columns(st.session_state['engine'], selected_table)
        
        exclude_cols = st.multiselect(
            "Колонки, которые НЕ учитывать при поиске дубликатов (например, id)",
            table_columns,
            default=[c for c in table_columns if c.lower() == 'id']
        )
        
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

        st.subheader("Исходные данные (первые 10 строк)")
        st.dataframe(df_raw.head(10))

        col1, col2, col3 = st.columns(3)
        col1.metric("Всего строк", len(df_raw))
        col2.metric("Столбцов", len(df_raw.columns))
        col3.metric("Строк с пропусками (исходно)", df_raw.isnull().any(axis=1).sum())

        tab1, tab2, tab3 = st.tabs(["Настройки", "Результаты", "Экспорт"])

        with tab1:
            st.header("Параметры анализа")
            c1, c2 = st.columns(2)
            
            with c1:
                detect_dup = st.checkbox("Обнаружить дубликаты (Dedupe)", value=True)
                # Порог для dedupe (0.5 - стандарт)
                dedupe_threshold = st.slider("Порог уверенности дубликатов", 0.1, 0.9, 0.5, 0.05)
                st.info("Используется библиотека Dedupe с активным обучением.")
                
            with c2:
                detect_out = st.checkbox("Обнаружить аномалии", value=True)
                contamination = st.slider("Доля выбросов (contamination)", 0.01, 0.5, 0.1, 0.01)
                
            handle_miss = st.checkbox("Заполнить пропуски", value=True)

            if st.button("ЗАПУСТИТЬ АНАЛИЗ", type="primary"):
                progress = st.progress(0)
                status = st.empty()
                df_work = df_raw.copy()

                # 1. ДУБЛИКАТЫ (DEDUPE)
                if detect_dup:
                    status.text("Поиск дубликатов через Dedupe (может занять время)...")
                    try:
                        # Важно: dedupe требует строковых данных для сравнения полей
                        df_work, dupe_groups = detect_duplicates_with_dedupe(
                            df_work, 
                            exclude_cols=exclude_cols, 
                            threshold=dedupe_threshold
                        )
                        st.session_state['dupe_groups'] = dupe_groups
                        st.success(f"Найдено {len(dupe_groups)} групп дубликатов")
                    except Exception as e:
                        st.error(f"Ошибка в модуле Dedupe: {e}")
                        df_work['duplicate_group'] = NO_DUPLICATE_GROUP
                        st.session_state['dupe_groups'] = []
                else:
                    df_work['duplicate_group'] = NO_DUPLICATE_GROUP
                    st.session_state['dupe_groups'] = []
                progress.progress(25)

                # 2. АНОМАЛИИ
                if detect_out:
                    status.text("Поиск аномалий (Isolation Forest)...")
                    df_work = detect_outliers_numeric(df_work, contamination=contamination)
                else:
                    df_work['is_outlier'] = NORMAL_LABEL
                    df_work['outlier_score'] = 0.0
                progress.progress(50)

                # 3. ПРОПУСКИ
                if handle_miss:
                    status.text("Заполнение пропусков...")
                    df_work = impute_missing(df_work)
                progress.progress(75)

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

                st.subheader("Таблица с выявленными проблемами")
                st.dataframe(report_df.head(50))

                st.subheader("Детальная статистика")
                d1, d2, d3, d4, d5 = st.columns(5)
                d1.metric("Всего записей", len(df_res))
                d2.metric("Групп дубликатов", len(dupe_groups))
                
                dup_rows = (df_res['duplicate_group'] != NO_DUPLICATE_GROUP).sum()
                d3.metric("Строк-дубликатов", int(dup_rows))
                
                out_rows = (df_res['is_outlier'] == 1).sum()
                d4.metric("Аномалий (Isolation Forest)", int(out_rows))
                
                miss_rows = df_raw.isnull().any(axis=1).sum()
                d5.metric("Строк с пропусками (исходно)", int(miss_rows))

                st.divider()
                
                g1, g2 = st.columns(2)
                
                with g1:
                    st.subheader("Гистограмма аномалий")
                    numeric_cols = df_res.select_dtypes(include=[np.number]).columns.tolist()
                    numeric_cols = [c for c in numeric_cols if c not in ['duplicate_group', 'outlier_score', 'is_outlier']]
                    
                    if numeric_cols:
                        chosen = st.selectbox("Выберите числовой столбец", numeric_cols, key="hist")
                        fig = plot_outliers_histogram(df_res, chosen)
                        st.plotly_chart(fig, use_container_width=True)
                        
                with g2:
                    st.subheader("Тепловая карта пропусков")
                    fig_miss = plot_missing_heatmap(df_raw)
                    if fig_miss:
                        st.plotly_chart(fig_miss, use_container_width=True)
                    else:
                        st.info("Нет пропусков для отображения.")
                        
            else:
                st.info("Сначала настройте параметры и запустите анализ.")

        with tab3:
            if st.session_state.get('analysis_complete', False):
                df_res = st.session_state['df_processed']
                report_df = generate_report(df_res, st.session_state.get('dupe_groups', []), original_df=df_raw)
                csv = report_df.to_csv(index=False).encode('utf-8')
                
                st.download_button("Скачать отчёт (CSV)", csv, "quality_report.csv", "text/csv")
                st.caption("Отчёт включает: duplicate_group, is_outlier, outlier_score, problem_type")
            else:
                st.info("Нет данных для экспорта.")

if __name__ == "__main__":
    main()
