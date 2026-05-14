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

# --- ФУНКЦИИ ПОДКЛЮЧЕНИЯ К БД ---

def get_engine(db_type, host, port, database, user, password):
    """Создаёт подключение к СУБД через SQLAlchemy."""
    try:
        if db_type == "PostgreSQL":
            url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        elif db_type == "MySQL":
            url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        elif db_type == "SQLite":
            # Для SQLite путь к файлу указывается напрямую
            url = f"sqlite:///{database}"
        else:
            st.error("Неподдерживаемый тип СУБД")
            return None
        
        engine = create_engine(url)
        # Проверка соединения
        with engine.connect() as conn:
            pass 
        return engine
    except Exception as e:
        st.error(f"Ошибка подключения: {e}")
        return None

def load_table(engine, table_name, limit=10000):
    """Загружает данные из таблицы в DataFrame."""
    try:
        query = f"SELECT * FROM \"{table_name}\" LIMIT {limit}"
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки таблицы: {e}")
        return None

# --- КЛАССЫ ОБРАБОТКИ ДАННЫХ ---

class Preprocessor:
    @staticmethod
    def clean_strings(df):
        """Очищает строковые столбцы."""
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
        return df

    @staticmethod
    def encode_categorical(df):
        """Кодирует категориальные признаки."""
        le = LabelEncoder()
        for col in df.select_dtypes(include=['object']).columns:
            # Обрабатываем возможные NaN перед кодированием
            df[col] = df[col].fillna('missing').astype(str)
            df[col] = le.fit_transform(df[col])
        return df

    @staticmethod
    def normalize_numeric(df):
        """Нормализует числовые признаки."""
        scaler = StandardScaler()
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            # Заполняем пропуски нулями временно для нормализации, чтобы не упало
            df_temp = df[num_cols].fillna(0)
            df[num_cols] = scaler.fit_transform(df_temp)
        return df

class DuplicateDetector:
    @staticmethod
    def detect_duplicates_simple(df):
        """
        Простой детектор дубликатов по полному совпадению строк.
        Возвращает DataFrame с колонкой 'duplicate_group' и список групп.
        """
        df['duplicate_group'] = -1
        dupe_groups = []
        
        # Находим индексы дубликатов (keep=False помечает ВСЕ копии как дубликаты)
        duplicate_mask = df.duplicated(keep=False)
        duplicates_df = df[duplicate_mask]
        
        if not duplicates_df.empty:
            # Группируем дубликаты по значениям всех колонок (кроме служебных)
            cols_to_check = [c for c in df.columns if c != 'duplicate_group']
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
    def detect_outliers_iforest(df, contamination=0.05):
        """Обнаружение аномалий с помощью Isolation Forest."""
        num_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(num_cols) == 0:
            df['is_outlier'] = 0
            df['outlier_score'] = 0.0
            return df
            
        # Берем только числовые данные для модели
        X = df[num_cols].copy()
        # Если есть пропуски в числах, заполняем их средним для работы модели
        X = X.fillna(X.mean())
        
        clf = IForest(contamination=contamination, random_state=42)
        clf.fit(X)
        
        # В PyOD: 1 - норма, -1 - аномалия
        df['is_outlier'] = clf.labels_
        df['outlier_score'] = clf.decision_scores_
        
        return df

class MissingHandler:
    @staticmethod
    def simple_impute(df, strategy='mean'):
        """Заполнение пропусков средним/модой."""
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(include=['object']).columns
        
        if len(num_cols) > 0:
            imp_num = SimpleImputer(strategy=strategy if strategy in ['mean', 'median'] else 'mean')
            df[num_cols] = imp_num.fit_transform(df[num_cols])
            
        if len(cat_cols) > 0:
            imp_cat = SimpleImputer(strategy='most_frequent')
            df[cat_cols] = imp_cat.fit_transform(df[cat_cols])
            
        return df

# --- ФУНКЦИИ ВИЗУАЛИЗАЦИИ И ОТЧЕТОВ ---

def generate_report(df, dupe_groups):
    """Формирует итоговую таблицу с колонкой problem_type."""
    report_df = df.copy()
    report_df['problem_type'] = ''
    
    # 1. Дубликаты
    if 'duplicate_group' in report_df.columns:
        mask_dupes = report_df['duplicate_group'] != -1
        report_df.loc[mask_dupes, 'problem_type'] += 'дубликат; '
        
    # 2. Аномалии (В PyOD -1 это аномалия)
    if 'is_outlier' in report_df.columns:
        mask_outliers = report_df['is_outlier'] == -1
        report_df.loc[mask_outliers, 'problem_type'] += 'аномалия; '
        
    # Убираем лишние пробелы и точки с запятой
    report_df['problem_type'] = report_df['problem_type'].str.strip('; ')
    report_df['problem_type'] = report_df['problem_type'].replace('', 'Нет проблем')
    
    return report_df

def plot_outliers(df, col):
    """Гистограмма с выделением аномалий."""
    if col not in df.columns or 'is_outlier' not in df.columns:
        return go.Figure()
    
    # Создаем копию для графика, чтобы не менять оригинал
    plot_df = df[[col, 'is_outlier']].copy()
    plot_df['Тип'] = plot_df['is_outlier'].apply(lambda x: 'Аномалия' if x == -1 else 'Норма')
    
    fig = px.histogram(plot_df, x=col, color='Тип', 
                       title=f'Распределение "{col}" (Красные = Аномалии)',
                       color_discrete_map={'Аномалия': 'red', 'Норма': 'blue'},
                       nbins=30)
    return fig

def plot_missing_heatmap(df_original):
    """Тепловая карта пропусков ИСХОДНЫХ данных."""
    # Берем только первые 200 строк и 20 столбцов для скорости отрисовки, если данных много
    sample_df = df_original.head(200).iloc[:, :20]
    missing = sample_df.isnull().astype(int)
    
    if missing.sum().sum() == 0:
        st.info("В выборке данных для визуализации пропусков не обнаружено.")
        return None
        
    fig = px.imshow(missing.T, 
                    title='Тепловая карта пропусков (Желтый/Светлый = Пропуск)',
                    color_continuous_scale='Viridis',
                    aspect='auto')
    return fig

# --- ГЛАВНОЕ ПРИЛОЖЕНИЕ STREAMLIT ---

def main():
    st.set_page_config(page_title="Анализ качества данных", layout="wide")
    st.title("Интеллектуальная система анализа качества данных")
    
    # --- БОКОВАЯ ПАНЕЛЬ ---
    with st.sidebar:
        st.header("Подключение к БД")
        db_type = st.selectbox("Тип СУБД", ["SQLite", "PostgreSQL", "MySQL"])
        
        host = st.text_input("Хост", "localhost" if db_type != "SQLite" else "")
        port = st.text_input("Порт", "5432" if db_type == "PostgreSQL" else ("3306" if db_type == "MySQL" else ""))
        database = st.text_input("Имя базы данных / Путь к файлу", "test_quality_with_defects.db" if db_type == "SQLite" else "")
        
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
                    
                    # Получаем список таблиц
                    try:
                        if db_type == "SQLite":
                            query = "SELECT name FROM sqlite_master WHERE type='table'"
                        else:
                            query = "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
                        
                        tables_df = pd.read_sql(query, engine)
                        st.session_state['tables'] = tables_df.iloc[:,0].tolist()
                    except Exception as e:
                        st.error(f"Не удалось получить список таблиц: {e}")

    # --- ОСНОВНАЯ ЧАСТЬ ---
    
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
        
        # Вкладки
        tab1, tab2, tab3 = st.tabs(["Настройка анализа", "Результаты", "Экспорт"])
        
        # --- ВКЛАДКА 1: НАСТРОЙКИ ---
        with tab1:
            st.header("Параметры анализа")
            col1, col2 = st.columns(2)
            
            with col1:
                check_dupes = st.checkbox("Обнаружить дубликаты", value=True)
                check_outliers = st.checkbox("Обнаружить аномалии", value=True)
                outlier_method = st.selectbox("Метод аномалий", ["Isolation Forest", "LOF"])
                contamination = st.slider("Доля выбросов (contamination)", 0.01, 0.2, 0.05, 0.01)
                
            with col2:
                check_missing = st.checkbox("Обработать пропуски", value=True)
                missing_method = st.selectbox("Метод заполнения", ["Простое (среднее/мода)", "kNN (медленно)"])
                
            if st.button("Запустить анализ", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                df_work = df_raw.copy()
                dupe_groups = []
                
                # 1. Предобработка
                status_text.text("Очистка строк...")
                df_work = Preprocessor.clean_strings(df_work)
                progress_bar.progress(20)
                
                # 2. Дубликаты (до кодирования, чтобы видеть смысловые дубликаты)
                if check_dupes:
                    status_text.text("Поиск дубликатов...")
                    df_work, dupe_groups = DuplicateDetector.detect_duplicates_simple(df_work)
                    st.session_state['dupe_groups'] = dupe_groups
                progress_bar.progress(40)
                
                # 3. Кодирование и нормализация (для ML моделей)
                status_text.text("Кодирование категорий...")
                # Сохраняем копию для отображения, но для моделей нужна цифровая версия
                df_ml = df_work.copy()
                df_ml = Preprocessor.encode_categorical(df_ml)
                
                if check_outliers:
                    status_text.text(f"Поиск аномалий ({outlier_method})...")
                    if outlier_method == "Isolation Forest":
                        df_ml = OutlierDetector.detect_outliers_iforest(df_ml, contamination)
                    else:
                        # LOF требует больше времени, можно добавить проверку размера
                        if len(df_ml) < 5000:
                             clf = LOF(contamination=contamination, n_neighbors=20)
                             num_cols = df_ml.select_dtypes(include=[np.number]).columns
                             X = df_ml[num_cols].fillna(0)
                             clf.fit(X)
                             df_ml['is_outlier'] = clf.labels_
                             df_ml['outlier_score'] = clf.decision_scores_
                        else:
                            st.warning("LOF слишком медленный для больших данных, используем Isolation Forest")
                            df_ml = OutlierDetector.detect_outliers_iforest(df_ml, contamination)
                progress_bar.progress(70)
                
                # 4. Обработка пропусков
                if check_missing:
                    status_text.text("Заполнение пропусков...")
                    if missing_method == "Простое (среднее/мода)":
                        df_ml = MissingHandler.simple_impute(df_ml)
                    # kNN можно добавить аналогично, но он медленный
                progress_bar.progress(90)
                
                # Сохраняем результаты
                st.session_state['df_processed'] = df_ml
                st.session_state['analysis_complete'] = True
                status_text.text("Анализ завершен!")
                progress_bar.progress(100)
                st.success("Перейдите на вкладку «Результаты»")

        # --- ВКЛАДКА 2: РЕЗУЛЬТАТЫ ---
        with tab2:
            if st.session_state.get('analysis_complete', False):
                df_res = st.session_state['df_processed']
                dupe_groups = st.session_state.get('dupe_groups', [])
                
                # Генерируем отчет
                report_df = generate_report(df_res, dupe_groups)
                
                st.subheader("Таблица проблемных записей")
                # Показываем только те, где есть проблемы, или все? Лучше все, но с фильтром
                st.dataframe(report_df.head(50)) # Ограничим вывод для скорости
                
                # Статистика
                st.subheader("Сводная статистика")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Всего записей", len(df_res))
                c2.metric("Найдено дубликатов", len(dupe_groups))
                # В PyOD -1 это аномалия
                n_outliers = int((df_res['is_outlier'] == -1).sum()) if 'is_outlier' in df_res else 0
                c3.metric("Найдено аномалий", n_outliers)
                # Пропуски считаем по ИСХОДНЫМ данным
                n_missing_rows = df_raw.isnull().any(axis=1).sum()
                c4.metric("Строк с пропусками (исходно)", n_missing_rows)
                
                st.divider()
                
                # Графики
                col_plot1, col_plot2 = st.columns(2)
                
                with col_plot1:
                    st.subheader("Гистограмма аномалий")
                    num_cols_for_plot = df_res.select_dtypes(include=[np.number]).columns.tolist()
                    if num_cols_for_plot and 'is_outlier' in df_res.columns:
                        selected_col = st.selectbox("Выберите столбец", num_cols_for_plot, key="plot_col")
                        fig_hist = plot_outliers(df_res, selected_col)
                        st.plotly_chart(fig_hist, use_container_width=True)
                    else:
                        st.write("Нет числовых столбцов или аномалий для отображения.")
                        
                with col_plot2:
                    st.subheader("Тепловая карта пропусков (Исходные данные)")
                    fig_heat = plot_missing_heatmap(df_raw)
                    if fig_heat:
                        st.plotly_chart(fig_heat, use_container_width=True)
                    else:
                        st.write("Пропусков в выборке не найдено.")
                        
            else:
                st.info("Сначала настройте параметры и запустите анализ на вкладке слева.")

        # --- ВКЛАДКА 3: ЭКСПОРТ ---
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
