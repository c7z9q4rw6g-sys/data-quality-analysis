import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from pyod.models.iforest import IForest
import plotly.express as px
import plotly.graph_objects as go

# --- ФУНКЦИИ ПОДКЛЮЧЕНИЯ (ТВОИ, БЕЗ ИЗМЕНЕНИЙ) ---

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
        query = f"SELECT * FROM \"{table_name}\" LIMIT {limit}"
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки таблицы: {e}")
        return None

# --- КЛАССЫ ОБРАБОТКИ (ИСПРАВЛЕННАЯ ЛОГИКА) ---

class Preprocessor:
    @staticmethod
    def clean_strings(df):
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
        return df

    @staticmethod
    def encode_categorical(df):
        le = LabelEncoder()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna('missing').astype(str)
            df[col] = le.fit_transform(df[col])
        return df

class DuplicateDetector:
    @staticmethod
    def detect_duplicates_full(df):
        """
        Ищет ПОЛНЫЕ дубликаты строк (по всем колонкам).
        Это гарантирует поиск тех копий, что мы добавили через SQL.
        """
        df['duplicate_group'] = -1
        dupe_groups = []
        
        # Находим индексы всех строк, которые имеют дубликаты
        duplicate_mask = df.duplicated(keep=False)
        duplicates_df = df[duplicate_mask]
        
        if not duplicates_df.empty:
            # Группируем по ВСЕМ колонкам, чтобы найти идентичные строки
            cols_to_check = list(df.columns)
            # Исключаем служебные колонки если есть, но пока берем все
            
            grouped = duplicates_df.groupby(cols_to_check).groups
            
            group_id = 0
            for key, indices in grouped.items():
                if len(indices) > 1:
                    dupe_groups.append(list(indices))
                    for idx in indices:
                        df.at[idx, 'duplicate_group'] = group_id
                    group_id += 1
                    
        return df, dupe_groups

class OutlierDetector:
    @staticmethod
    def detect_outliers_iforest_robust(df, contamination=0.1):
        """
        Более надежный поиск аномалий.
        Работает только с числовыми колонками.
        """
        num_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(num_cols) == 0:
            df['is_outlier'] = 0
            df['outlier_score'] = 0.0
            return df
            
        # Берем числовые данные. Заполняем пропуски нулями для работы модели.
        X = df[num_cols].copy().fillna(0)
        
        # Isolation Forest лучше работает на сырых данных или слабо нормализованных.
        # Стандартный скалер может скрыть выбросы, если их мало.
        # Попробуем без жесткого скалера, или со слабым.
        
        clf = IForest(contamination=contamination, random_state=42, n_estimators=100)
        clf.fit(X)
        
        # В PyOD: 1 - норма, -1 - аномалия
        df['is_outlier'] = clf.labels_
        df['outlier_score'] = clf.decision_scores_
        
        return df

class MissingHandler:
    @staticmethod
    def simple_impute(df):
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(include=['object']).columns
        
        if len(num_cols) > 0:
            imp_num = SimpleImputer(strategy='mean')
            df[num_cols] = imp_num.fit_transform(df[num_cols])
            
        if len(cat_cols) > 0:
            imp_cat = SimpleImputer(strategy='most_frequent')
            df[cat_cols] = imp_cat.fit_transform(df[cat_cols])
            
        return df

# --- ВИЗУАЛИЗАЦИЯ ---

def generate_report(df, dupe_groups):
    report_df = df.copy()
    report_df['problem_type'] = ''
    
    # Дубликаты
    if 'duplicate_group' in report_df.columns:
        mask_dupes = report_df['duplicate_group'] != -1
        report_df.loc[mask_dupes, 'problem_type'] += 'дубликат; '
        
    # Аномалии (-1 в PyOD)
    if 'is_outlier' in report_df.columns:
        mask_outliers = report_df['is_outlier'] == -1
        report_df.loc[mask_outliers, 'problem_type'] += 'аномалия; '
        
    report_df['problem_type'] = report_df['problem_type'].str.strip('; ')
    report_df['problem_type'] = report_df['problem_type'].replace('', 'Нет проблем')
    
    return report_df

def plot_outliers(df, col):
    if col not in df.columns or 'is_outlier' not in df.columns:
        return go.Figure()
    
    plot_df = df[[col, 'is_outlier']].copy()
    plot_df['Тип'] = plot_df['is_outlier'].apply(lambda x: 'Аномалия' if x == -1 else 'Норма')
    
    fig = px.histogram(plot_df, x=col, color='Тип', 
                       title=f'Распределение "{col}" (Красные = Аномалии)',
                       color_discrete_map={'Аномалия': 'red', 'Норма': 'blue'},
                       nbins=30)
    return fig

def plot_missing_heatmap(df_original):
    """Тепловая карта по ИСХОДНЫМ данным."""
    sample_df = df_original.head(500).iloc[:, :20] # Берем побольше строк
    missing = sample_df.isnull().astype(int)
    
    if missing.sum().sum() == 0:
        return None
        
    fig = px.imshow(missing.T, 
                    title='Тепловая карта пропусков (Желтый = Пропуск)',
                    color_continuous_scale='Viridis',
                    aspect='auto')
    return fig

# --- ГЛАВНОЕ ПРИЛОЖЕНИЕ ---

def main():
    st.set_page_config(page_title="Анализ качества данных", layout="wide")
    st.title("Интеллектуальная система анализа качества данных")
    
    # БОКОВАЯ ПАНЕЛЬ (ТВОЯ)
    with st.sidebar:
        st.header("Подключение к БД")
        db_type = st.selectbox("Тип СУБД", ["SQLite", "PostgreSQL", "MySQL"])
        
        host = st.text_input("Хост", "localhost" if db_type != "SQLite" else "")
        port = st.text_input("Порт", "5432" if db_type == "PostgreSQL" else ("3306" if db_type == "MySQL" else ""))
        database = st.text_input("Имя базы данных / Путь к файлу", "final_diploma_db.db" if db_type == "SQLite" else "")
        
        if db_type == "SQLite":
            user = ""
            password = ""
        else:
            user = st.text_input("Пользователь", "postgres")
            password = st.text_input("Пароль", type="password", value="")
        
        if st.button("Подключиться"):
            if db_type == "SQLite" and not database:
                st.error("Для SQLite укажите имя файла .db")
            else:
                engine = get_engine(db_type, host, port, database, user, password)
                if engine:
                    st.session_state['engine'] = engine
                    st.success("Подключено успешно!")
                    
                    try:
                        if db_type == "SQLite":
                            query = "SELECT name FROM sqlite_master WHERE type='table'"
                        else:
                            query = "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
                        
                        tables_df = pd.read_sql(query, engine)
                        st.session_state['tables'] = tables_df.iloc[:,0].tolist()
                    except Exception as e:
                        st.error(f"Не удалось получить список таблиц: {e}")

    # ОСНОВНАЯ ЧАСТЬ
    if 'engine' in st.session_state and 'tables' in st.session_state:
        selected_table = st.selectbox("Выберите таблицу", st.session_state['tables'])
        
        if st.button("Загрузить данные"):
            df_raw = load_table(st.session_state['engine'], selected_table)
            if df_raw is not None:
                st.session_state['df_raw'] = df_raw
                st.success(f"Загружено {len(df_raw)} записей.")
                
    if 'df_raw' in st.session_state:
        df_raw = st.session_state['df_raw']
        
        st.subheader("Исходные данные (первые 10 строк)")
        st.dataframe(df_raw.head(10))
        
        tab1, tab2, tab3 = st.tabs(["Настройка анализа", "Результаты", "Экспорт"])
        
        with tab1:
            st.header("Параметры анализа")
            col1, col2 = st.columns(2)
            
            with col1:
                check_dupes = st.checkbox("Обнаружить дубликаты", value=True)
                check_outliers = st.checkbox("Обнаружить аномалии", value=True)
                # Увеличил дефолтное значение contamination до 0.1, чтобы точно что-то найти
                contamination = st.slider("Доля выбросов (contamination)", 0.01, 0.5, 0.1, 0.01)
                
            with col2:
                check_missing = st.checkbox("Обработать пропуски", value=True)
                
            if st.button("Запустить анализ", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # ВАЖНО: Работаем с копией сырых данных для поиска дубликатов
                df_work = df_raw.copy()
                dupe_groups = []
                
                # 1. Очистка строк (чтобы дубликаты нашли "ООО Ромашка" и "ооо ромашка ")
                status_text.text("Очистка строк...")
                df_work = Preprocessor.clean_strings(df_work)
                progress_bar.progress(20)
                
                # 2. Поиск дубликатов НА ЭТАПЕ СЫРЫХ/ОЧИЩЕННЫХ ДАННЫХ
                if check_dupes:
                    status_text.text("Поиск дубликатов...")
                    df_work, dupe_groups = DuplicateDetector.detect_duplicates_full(df_work)
                    st.session_state['dupe_groups'] = dupe_groups
                progress_bar.progress(40)
                
                # 3. Подготовка к ML (кодирование)
                status_text.text("Кодирование категорий...")
                df_ml = df_work.copy()
                df_ml = Preprocessor.encode_categorical(df_ml)
                
                # 4. Поиск аномалий НА ЧИСЛОВЫХ ДАННЫХ
                if check_outliers:
                    status_text.text("Поиск аномалий...")
                    df_ml = OutlierDetector.detect_outliers_iforest_robust(df_ml, contamination)
                progress_bar.progress(70)
                
                # 5. Обработка пропусков (для финального отчета, если нужно)
                if check_missing:
                    status_text.text("Заполнение пропусков...")
                    df_ml = MissingHandler.simple_impute(df_ml)
                progress_bar.progress(90)
                
                st.session_state['df_processed'] = df_ml
                st.session_state['analysis_complete'] = True
                status_text.text("Анализ завершен!")
                progress_bar.progress(100)
                st.success("Перейдите на вкладку «Результаты»")

        with tab2:
            if st.session_state.get('analysis_complete', False):
                df_res = st.session_state['df_processed']
                dupe_groups = st.session_state.get('dupe_groups', [])
                
                report_df = generate_report(df_res, dupe_groups)
                
                st.subheader("Таблица проблемных записей")
                st.dataframe(report_df.head(50))
                
                # СТАТИСТИКА
                st.subheader("Сводная статистика")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Всего записей", len(df_res))
                c2.metric("Найдено дубликатов", len(dupe_groups))
                
                # Подсчет аномалий (-1)
                n_outliers = int((df_res['is_outlier'] == -1).sum()) if 'is_outlier' in df_res else 0
                c3.metric("Найдено аномалий", n_outliers)
                
                # Пропуски по ИСХОДНИКУ
                n_missing_rows = df_raw.isnull().any(axis=1).sum()
                c4.metric("Строк с пропусками (исходно)", n_missing_rows)
                
                st.divider()
                
                col_plot1, col_plot2 = st.columns(2)
                
                with col_plot1:
                    st.subheader("Гистограмма аномалий")
                    num_cols_for_plot = df_res.select_dtypes(include=[np.number]).columns.tolist()
                    if num_cols_for_plot and 'is_outlier' in df_res.columns:
                        selected_col = st.selectbox("Выберите столбец", num_cols_for_plot, key="plot_col")
                        fig_hist = plot_outliers(df_res, selected_col)
                        st.plotly_chart(fig_hist, use_container_width=True)
                        
                with col_plot2:
                    st.subheader("Тепловая карта пропусков (Исходные данные)")
                    fig_heat = plot_missing_heatmap(df_raw)
                    if fig_heat:
                        st.plotly_chart(fig_heat, use_container_width=True)
                    else:
                        st.write("Пропусков не найдено.")
                        
            else:
                st.info("Сначала настройте параметры и запустите анализ.")

        with tab3:
            if st.session_state.get('analysis_complete', False):
                df_res = st.session_state['df_processed']
                csv = df_res.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Скачать полный отчёт (CSV)",
                    data=csv,
                    file_name='quality_report.csv',
                    mime='text/csv',
                )
            else:
                st.info("Сначала выполните анализ.")

if __name__ == "__main__":
    main()
