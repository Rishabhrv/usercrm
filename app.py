import streamlit as st
import warnings
import time
import base64
import json
import hashlib
import hmac
from dotenv import load_dotenv
import time
import numpy as np  
import pandas as pd
import os
from sqlalchemy import text
warnings.simplefilter('ignore')
from datetime import datetime, timedelta



# Set page configuration
st.set_page_config(
    layout="wide",  # Set layout to wide mode
    initial_sidebar_state="collapsed",
    page_icon="chart_with_upwards_trend",  
     page_title="Content Dashboard",
)

st.markdown("""
    <style>
            
        /* Remove Streamlit's default top padding */
        .main > div {
            padding-top: 0px !important;
        }
        /* Ensure the first element has minimal spacing */
        .block-container {
            padding-top: 30px !important;  /* Small padding for breathing room */
        }
            
    </style>
""", unsafe_allow_html=True)

load_dotenv()
SECRET_KEY = os.getenv('SECRET_KEY', 'default-secret-key') 

def validate_token():
    if 'token' not in st.session_state:
        params = st.query_params
        if 'token' in params:
            st.session_state.token = params['token']
        else:
            st.error("Access Denied: Login Required")
            st.stop()

    token = st.session_state.token

    try:
        parts = token.split('.')
        if len(parts) != 3:
            raise ValueError("Invalid token format")

        header = json.loads(base64.urlsafe_b64decode(parts[0] + '==').decode('utf-8'))
        payload = json.loads(base64.urlsafe_b64decode(parts[1] + '==').decode('utf-8'))

        signature = base64.urlsafe_b64decode(parts[2] + '==')
        expected_signature = hmac.new(
            SECRET_KEY.encode(),
            f"{parts[0]}.{parts[1]}".encode(),
            hashlib.sha256
        ).digest()

        if not hmac.compare_digest(signature, expected_signature):
            raise ValueError("Invalid token signature")

        if 'exp' in payload and payload['exp'] < time.time():
            raise ValueError("Token has expired")

        # Store validated user info in session_state
        st.session_state.user = payload['user']
        st.session_state.role = payload['role']

    except ValueError as e:
        st.error(f"Access Denied: {e}")
        st.stop()

#validate_token()

#user_role = st.session_state.get("role", "Guest")

user_role = "writer"
# user_role = "proofreader"

# user_role = st.pills("Select Role", ["writer", "proofreader", "formatter", "cover_designer"], default="writer", label_visibility='collapsed', key="user_role")
# user_role = user_role


st.cache_data.clear()

# --- Database Connection ---
def connect_db():
    try:
        @st.cache_resource
        def get_connection():
            return st.connection('mysql', type='sql')
        conn = get_connection()
        return conn
    except Exception as e:
        st.error(f"Error connecting to MySQL: {e}")
        st.stop()

# Connect to MySQL
conn = connect_db()


###################################################################################################################
#######################################  New Functions #############################################
###################################################################################################################

from datetime import datetime, timedelta

def fetch_books(months_back: int = 3, section: str = "writing") -> pd.DataFrame:
    conn = connect_db()
    cutoff_date = datetime.now().date() - timedelta(days=30 * months_back)
    cutoff_date_str = cutoff_date.strftime('%Y-%m-%d')
    
    # Define section-specific columns with consistent renaming
    section_columns = {
        "writing": {
            "base": ["writing_by AS 'Writing By'", "writing_start AS 'Writing Start'", "writing_end AS 'Writing End'"],
            "extra": [],
            "publish_filter": "AND is_publish_only = 0"  # Only for writing
        },
        "proofreading": {
            "base": ["proofreading_by AS 'Proofreading By'", "proofreading_start AS 'Proofreading Start'", "proofreading_end AS 'Proofreading End'"],
            "extra": ["writing_end AS 'Writing End'", "writing_by AS 'Writing By'"],
            "publish_filter": ""  # No filter for proofreading
        },
        "formatting": {
            "base": ["formatting_by AS 'Formatting By'", "formatting_start AS 'Formatting Start'", "formatting_end AS 'Formatting End'"],
            "extra": ["proofreading_end AS 'Proofreading End'", "book_pages AS 'Total Book Pages'"],
            "publish_filter": ""  # No filter for formatting
        },
        "cover": {
            "base": ["cover_by AS 'Cover By'", "cover_start AS 'Cover Start'", "cover_end AS 'Cover End'"],
            "extra": [],
            "publish_filter": ""  # No filter for cover
        }
    }
    config = section_columns.get(section, section_columns["writing"])
    columns = config["base"] + config["extra"]
    columns_str = ", ".join(columns)
    publish_filter = config["publish_filter"]
    
    query = f"""
        SELECT 
            book_id AS 'Book ID',
            title AS 'Title',
            date AS 'Date',
            {columns_str},
            is_publish_only AS 'Is Publish Only'
        FROM books 
        WHERE date >= '{cutoff_date_str}'
        {publish_filter}
        ORDER BY date DESC
    """
    df = conn.query(query, show_spinner=False)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df

# --- Reusable Month Selector ---
def render_month_selector(books_df):
    unique_months = sorted(books_df['Date'].apply(lambda x: x.strftime('%B %Y')).unique(), 
                          key=lambda x: datetime.strptime(x, '%B %Y'), reverse=False)
    default_month = unique_months[-1]  # Most recent month
    selected_month = st.pills("Select Month", unique_months, default=default_month, 
                             key=f"month_selector_{st.session_state.get('section', 'writing')}", 
                             label_visibility='collapsed')
    return selected_month

# --- Reusable Metrics ---
def render_metrics(books_df, selected_month, section):
    selected_month_dt = datetime.strptime(selected_month, '%B %Y')
    month_start = selected_month_dt.replace(day=1).date()
    month_end = (selected_month_dt.replace(day=1) + timedelta(days=31)).replace(day=1).date() - timedelta(days=1)
    filtered_books = books_df[(books_df['Date'] >= month_start) & (books_df['Date'] <= month_end)]

    total_books = len(filtered_books)
    completed_books = len(filtered_books[
        filtered_books[f'{section.capitalize()} End'].notnull() & 
        (filtered_books[f'{section.capitalize()} End'] != '0000-00-00 00:00:00')
    ])
    pending_books = len(filtered_books[
        filtered_books[f'{section.capitalize()} Start'].isnull() | 
        (filtered_books[f'{section.capitalize()} Start'] == '0000-00-00 00:00:00')
    ])

    col1, col2 = st.columns([11, 1], gap="large", vertical_alignment="bottom")
    with col1:
        st.subheader(f"Metrics of {selected_month}")
    with col2:
        if st.button(":material/refresh: Refresh", key=f"refresh_{section}", type="tertiary"):
            st.cache_data.clear()

    col1, col2, col3 = st.columns(3, border=True)
    with col1:
        st.metric(f"Books in {selected_month}", total_books)
    with col2:
        st.metric(f"{section.capitalize()} Done in {selected_month}", completed_books)
    with col3:
        st.metric(f"Pending in {selected_month}", pending_books)
    return filtered_books

# Helper function to fetch unique names (assumed to exist or can be added)
def fetch_unique_names(column_name, conn):
    query = f"SELECT DISTINCT {column_name} FROM books WHERE {column_name} IS NOT NULL"
    result = conn.query(query, show_spinner=False)
    return sorted(result[column_name].tolist())

# --- Helper Functions ---
def get_status(start, end, current_date):
    if pd.isnull(start) or start == '0000-00-00 00:00:00':
        return "Pending", None
    elif pd.notnull(start) and start != '0000-00-00 00:00:00' and pd.isnull(end):
        start_date = start.date() if isinstance(start, pd.Timestamp) else pd.to_datetime(start).date()
        days = (current_date - start_date).days
        return "Running", days
    return "-", None

def get_days_since_enrolled(enroll_date, current_date):
    if pd.notnull(enroll_date):
        date_enrolled = enroll_date if isinstance(enroll_date, datetime) else pd.to_datetime(enroll_date).date()
        return (current_date - date_enrolled).days
    return None

def get_worker_by(start, worker_by, worker_map=None):
    if pd.notnull(start) and start != '0000-00-00 00:00:00':
        worker = worker_by if pd.notnull(worker_by) else "Unknown Worker"
        if worker_map and worker in worker_map:
            return worker, worker_map[worker]
        return worker, None
    return "Not Assigned", None

###################################################################################################################
#######################################  New Functions #############################################
###################################################################################################################


def fetch_book_details(book_id, conn):
    query = f"SELECT title FROM books WHERE book_id = {book_id}"
    return conn.query(query, show_spinner=False)


# Reusable dialog for editing section details
@st.dialog("Edit Section Details", width='large')
def edit_section_dialog(book_id, conn, section):
    # Map section to display name and database columns
    section_config = {
        "writing": {"display": "Writing", "by": "writing_by", "start": "writing_start", "end": "writing_end"},
        "proofreading": {"display": "Proofreading", "by": "proofreading_by", "start": "proofreading_start", "end": "proofreading_end"},
        "formatting": {"display": "Formatting", "by": "formatting_by", "start": "formatting_start", "end": "formatting_end", "extra": "book_pages"},
        "cover": {"display": "Cover Page", "by": "cover_by", "start": "cover_start", "end": "cover_end"}
    }
    
    config = section_config.get(section, section_config["writing"])  # Default to writing if section invalid
    display_name = config["display"]

    # Fetch book title
    book_details = fetch_book_details(book_id, conn)
    if not book_details.empty:
        book_title = book_details.iloc[0]['title']
        st.markdown(f"<h3 style='color:#4CAF50;'>{book_id} : {book_title}</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"### {display_name} Details for Book ID: {book_id}")
        st.warning("Book title not found.")

    # Fetch current section data, including book_pages for formatting
    extra_column = ", " + config["extra"] if "extra" in config else ""
    query = f"SELECT {config['start']}, {config['end']}, {config['by']}{extra_column} FROM books WHERE book_id = {book_id}"
    book_data = conn.query(query, show_spinner=False)
    current_data = book_data.iloc[0].to_dict() if not book_data.empty else {}

    # Fetch unique names for the section
    names = fetch_unique_names(config["by"], conn)
    options = ["Select Worker"] + names + ["Add New..."]

    # Initialize session state
    for key in [f"{section}_by", f"{section}_start_date", f"{section}_start_time", f"{section}_end_date", f"{section}_end_time"]:
        if f"{key}_{book_id}" not in st.session_state:
            if "by" in key:
                st.session_state[f"{key}_{book_id}"] = current_data.get(config["by"], "")
            elif "start" in key:
                st.session_state[f"{key}_{book_id}"] = current_data.get(config["start"], None)
            else:  # end
                st.session_state[f"{key}_{book_id}"] = current_data.get(config["end"], None)
    
    # Initialize book_pages for formatting
    if section == "formatting" and f"book_pages_{book_id}" not in st.session_state:
        st.session_state[f"book_pages_{book_id}"] = current_data.get("book_pages", 0)

    # Custom CSS
    st.markdown("""
        <style>
        .field-label {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .changed {
            background-color: #FFF3E0;
            padding: 2px 6px;
            border-radius: 4px;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.form(key=f"{section}_form_{book_id}", border=False):
        st.markdown(f'<div class="field-label">{display_name} Worker</div>', unsafe_allow_html=True)
        selected_worker = st.selectbox(
            "Worker",
            options,
            index=options.index(st.session_state[f"{section}_by_{book_id}"]) if st.session_state[f"{section}_by_{book_id}"] in names else 0,
            key=f"{section}_select_{book_id}",
            label_visibility="collapsed",
            help=f"Select an existing {display_name.lower()} worker or add a new one."
        )
        
        # Handle "Add New..." logic
        if selected_worker == "Add New...":
            new_worker = st.text_input(
                "New Worker",
                value="",
                key=f"{section}_new_input_{book_id}",
                placeholder=f"Enter new {display_name.lower()} worker name...",
                label_visibility="collapsed"
            )
            worker = new_worker.strip() if new_worker.strip() else None  # Only use if non-empty
        else:
            worker = selected_worker if selected_worker != "Select Worker" else None  # Reset to None if "Select Worker"
        # Update session state only if a valid worker is selected or entered
        if worker:
            st.session_state[f"{section}_by_{book_id}"] = worker

        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.markdown(f'<div class="field-label">Start Date & Time</div>', unsafe_allow_html=True)
            start_date = st.date_input(
                "Start Date",
                value=st.session_state[f"{section}_start_date_{book_id}"],
                key=f"{section}_start_date_{book_id}",
                label_visibility="collapsed",
                help=f"When {display_name.lower()} began"
            )
            start_time = st.time_input(
                "Start Time",
                value=st.session_state[f"{section}_start_time_{book_id}"],
                key=f"{section}_start_time_{book_id}",
                label_visibility="collapsed"
            )
        with col2:
            st.markdown(f'<div class="field-label">End Date & Time</div>', unsafe_allow_html=True)
            end_date = st.date_input(
                "End Date",
                value=st.session_state[f"{section}_end_date_{book_id}"],
                key=f"{section}_end_date_{book_id}",
                label_visibility="collapsed",
                help=f"When {display_name.lower()} was completed (leave blank if ongoing)"
            )
            end_time = st.time_input(
                "End Time",
                value=st.session_state[f"{section}_end_time_{book_id}"],
                key=f"{section}_end_time_{book_id}",
                label_visibility="collapsed"
            )

        # Add Total Book Pages field for formatting only
        book_pages = None
        if section == "formatting":
            st.markdown('<div class="field-label">Total Book Pages</div>', unsafe_allow_html=True)
            book_pages = st.number_input(
                "Total Book Pages",
                min_value=0,
                value=st.session_state[f"book_pages_{book_id}"],
                key=f"book_pages_{book_id}",
                label_visibility="collapsed",
                help="Enter the total number of pages in the book"
            )

        start = f"{start_date} {start_time}" if start_date and start_time else None
        end = f"{end_date} {end_time}" if end_date and end_time else None

        col_save, col_cancel = st.columns([1, 1])
        with col_save:
            submit = st.form_submit_button("💾 Save and Close", use_container_width=True)
        with col_cancel:
            cancel = st.form_submit_button("Cancel", use_container_width=True, type="secondary")

        if submit:
            if start and end and start > end:
                st.error("Start must be before End.")
            else:
                with st.spinner(f"Saving {display_name} details..."):
                    time.sleep(2)
                    updates = {
                        config["start"]: start,
                        config["end"]: end,
                        config["by"]: worker
                    }
                    if section == "formatting":
                        updates["book_pages"] = book_pages
                    
                    with conn.session as session:
                        set_clause = ", ".join([f"{key} = :{key}" for key in updates.keys()])
                        query = f"UPDATE books SET {set_clause} WHERE book_id = :id"
                        params = updates.copy()
                        params["id"] = int(book_id)
                        session.execute(text(query), params)
                        session.commit()
                    st.success(f"✔️ Updated {display_name} details")
                    time.sleep(1)
                    st.rerun()

        elif cancel:
            for key in [f"{section}_by", f"{section}_start_date", f"{section}_start_time", f"{section}_end_date", f"{section}_end_time"]:
                st.session_state.pop(f"{key}_{book_id}", None)
            if section == "formatting":
                st.session_state.pop(f"book_pages_{book_id}", None)
            st.rerun()

# --- Updated CSS ---
st.markdown("""
    <style>
    .header-row {
        padding-bottom: 5px;
        margin-bottom: 10px;
    }
    .header {
        font-weight: bold;
    }
    .header-line {
        border-bottom: 1px solid #ddd;
        margin-top: -10px;
    }
    .pill {
        padding: 2px 6px;
        border-radius: 10px;
        font-size: 12px;
        display: inline-block;
        margin-right: 4px;
    }
    .since-enrolled {
        background-color: #FFF3E0;
        color: #FF9800;
        padding: 1px 4px;
        border-radius: 8px;
        font-size: 11px;
    }
    .section-start-not {
        background-color: #F5F5F5;
        color: #757575;
    }
    .section-start-date {
        background-color: #FFF3E0;
        color: #FF9800;
    }
    .worker-by-not {
        background-color: #F5F5F5;
        color: #757575;
    }
    /* Worker-specific colors (softer tones, reusable for all sections) */
    .worker-by-0 { background-color: #E3F2FD; color: #1976D2; } /* Blue */
    .worker-by-1 { background-color: #FCE4EC; color: #D81B60; } /* Pink */
    .worker-by-2 { background-color: #E0F7FA; color: #006064; } /* Cyan */
    .worker-by-3 { background-color: #F1F8E9; color: #558B2F; } /* Light Green */
    .worker-by-4 { background-color: #FFF3E0; color: #EF6C00; } /* Orange */
    .worker-by-5 { background-color: #F3E5F5; color: #8E24AA; } /* Purple */
    .worker-by-6 { background-color: #FFFDE7; color: #F9A825; } /* Yellow */
    .worker-by-7 { background-color: #EFEBE9; color: #5D4037; } /* Brown */
    .worker-by-8 { background-color: #E0E0E0; color: #424242; } /* Grey */
    .worker-by-9 { background-color: #E8EAF6; color: #283593; } /* Indigo */
    .status-pending {
        background-color: #FFEBEE;
        color: #F44336;
        font-weight: bold;
    }
    .status-running {
        background-color: #FFFDE7;
        color: #F9A825;
        font-weight: bold;
    }
    /* Standardized badge colors for Pending (red) and Running (yellow) */
    .status-badge-red {
        background-color: #FFEBEE;
        color: #F44336;
        padding: 4px 8px;
        border-radius: 12px;
        font-weight: bold;
        display: inline-flex;
        align-items: center;
    }
    .status-badge-yellow {
        background-color: #FFFDE7;
        color: #F9A825;
        padding: 4px 8px;
        border-radius: 12px;
        font-weight: bold;
        display: inline-flex;
        align-items: center;
    }
    .badge-count {
        background-color: rgba(255, 255, 255, 0.9);
        color: inherit;
        padding: 2px 6px;
        border-radius: 10px;
        margin-left: 6px;
        font-size: 12px;
        font-weight: normal;
    }
    </style>
""", unsafe_allow_html=True)

def render_table(books_df, title, column_sizes, color, section, role, is_running=False):
    if books_df.empty:
        st.warning(f"No {title.lower()} books available from the last 3 months.")
    else:
        cont = st.container(border=True)
        with cont:
            count = len(books_df)
            badge_color = 'yellow' if is_running else 'red'
            st.markdown(f"<h5><span class='status-badge-{badge_color}'>{title} Books <span class='badge-count'>{count}</span></span></h5>", 
                        unsafe_allow_html=True)
            st.markdown('<div class="header-row">', unsafe_allow_html=True)
            
            columns = ["Book ID", "Title", "Date", "Status"]
            if role == "proofreader":
                columns.extend(["Writing End", "Writing By"])
            elif role == "formatter":
                columns.extend(["Proofreading End"])
            if is_running:
                columns.extend([f"{section.capitalize()} Start", f"{section.capitalize()} By", "Action"])
            else:
                columns.append("Action")
            
            col_configs = st.columns(column_sizes[:len(columns)])
            for i, col in enumerate(columns):
                with col_configs[i]:
                    st.markdown(f'<span class="header">{col}</span>', unsafe_allow_html=True)
            st.markdown('</div><div class="header-line"></div>', unsafe_allow_html=True)

            current_date = datetime.now().date()
            if is_running and user_role == role:
                unique_workers = [w for w in books_df[f'{section.capitalize()} By'].unique() if pd.notnull(w)]
                worker_map = {worker: idx % 10 for idx, worker in enumerate(unique_workers)}
            else:
                worker_map = None

            for _, row in books_df.iterrows():
                col_configs = st.columns(column_sizes[:len(columns)])
                col_idx = 0
                
                with col_configs[col_idx]:
                    st.write(row['Book ID'])
                col_idx += 1
                with col_configs[col_idx]:
                    st.write(row['Title'])
                col_idx += 1
                with col_configs[col_idx]:
                    st.write(row['Date'].strftime('%Y-%m-%d') if pd.notnull(row['Date']) else "-")
                col_idx += 1
                with col_configs[col_idx]:
                    status, days = get_status(row[f'{section.capitalize()} Start'], row[f'{section.capitalize()} End'], current_date)
                    days_since = get_days_since_enrolled(row['Date'], current_date)
                    status_html = f'<span class="pill status-{"pending" if status == "Pending" else "running"}">{status}'
                    if days is not None:
                        status_html += f' {days}d'
                    status_html += '</span>'
                    if not is_running and days_since is not None:
                        status_html += f'<span class="since-enrolled">{days_since}d</span>'
                    st.markdown(status_html, unsafe_allow_html=True)
                col_idx += 1
                
                # Role-specific columns with NaT handling
                if role == "proofreader":
                    with col_configs[col_idx]:
                        writing_end = row['Writing End']
                        value = writing_end.strftime('%Y-%m-%d') if not pd.isna(writing_end) and writing_end != '0000-00-00 00:00:00' else "-"
                        st.markdown(f'<span>{value}</span>', unsafe_allow_html=True)
                    col_idx += 1
                    with col_configs[col_idx]:
                        writing_by = row['Writing By']
                        value = writing_by if pd.notnull(writing_by) and writing_by else "-"
                        st.markdown(f'<span>{value}</span>', unsafe_allow_html=True)
                    col_idx += 1
                elif role == "formatter":
                    with col_configs[col_idx]:
                        proofreading_end = row['Proofreading End']
                        value = proofreading_end.strftime('%Y-%m-%d') if not pd.isna(proofreading_end) and proofreading_end != '0000-00-00 00:00:00' else "-"
                        st.markdown(f'<span>{value}</span>', unsafe_allow_html=True)
                    col_idx += 1
                
                # Running-specific columns
                if is_running and user_role == role:
                    with col_configs[col_idx]:
                        start = row[f'{section.capitalize()} Start']
                        if pd.notnull(start) and start != '0000-00-00 00:00:00':
                            st.markdown(f'<span class="pill section-start-date">{start.strftime("%d %B %Y")}</span>', 
                                        unsafe_allow_html=True)
                        else:
                            st.markdown('<span class="pill section-start-not">Not started</span>', unsafe_allow_html=True)
                    col_idx += 1
                    with col_configs[col_idx]:
                        worker, worker_idx = get_worker_by(row[f'{section.capitalize()} Start'], 
                                                          row[f'{section.capitalize()} By'], worker_map)
                        class_name = f"worker-by-{worker_idx}" if worker_idx is not None else "worker-by-not"
                        st.markdown(f'<span class="pill {class_name}">{worker}</span>', unsafe_allow_html=True)
                    col_idx += 1
                    with col_configs[col_idx]:
                        if st.button("Edit", key=f"edit_{section}_{row['Book ID']}"):
                            edit_section_dialog(row['Book ID'], conn, section)
                elif not is_running and user_role == role:
                    with col_configs[col_idx]:
                        if st.button("Edit", key=f"edit_{section}_{row['Book ID']}"):
                            edit_section_dialog(row['Book ID'], conn, section)

# --- Section Configuration ---
sections = {
    "writing": {"role": "writer", "color": "green"},
    "proofreading": {"role": "proofreader", "color": "blue"},
    "formatting": {"role": "formatter", "color": "purple"},
    "cover": {"role": "cover_designer", "color": "orange"}
}

# --- Column Sizes (Adjusted for New Columns) ---
column_sizes_running = [0.7, 5.5, 1, 1, 1.2, 1.2, 1, 1, 1]  # Adjusted for proofreader/formatter extras
column_sizes_pending = [0.7, 5.5, 1, 1, 1.2, 1.2, 1]  # Adjusted for proofreader/formatter extras

for section, config in sections.items():
    if user_role == config["role"] or user_role == "admin":
        st.session_state['section'] = section
        books_df = fetch_books(months_back=4, section=section)
        
        # Filter books based on section dependencies
        if section == "writing":
            running_books = books_df[
                books_df['Writing Start'].notnull() & 
                books_df['Writing End'].isnull()
            ]
            pending_books = books_df[
                books_df['Writing Start'].isnull()
            ]
        elif section == "proofreading":
            running_books = books_df[
                books_df['Proofreading Start'].notnull() & 
                books_df['Proofreading End'].isnull()
            ]
            pending_books = books_df[
                books_df['Writing End'].notnull() & 
                books_df['Proofreading Start'].isnull()
            ]
        elif section == "formatting":
            running_books = books_df[
                books_df['Formatting Start'].notnull() & 
                books_df['Formatting End'].isnull()
            ]
            pending_books = books_df[
                books_df['Proofreading End'].notnull() & 
                books_df['Formatting Start'].isnull()
            ]
        elif section == "cover":
            running_books = books_df[
                books_df['Cover Start'].notnull() & 
                books_df['Cover End'].isnull()
            ]
            pending_books = books_df[
                books_df['Formatting End'].notnull() & 
                books_df['Cover Start'].isnull()
            ]
        
        # Render UI components
        selected_month = render_month_selector(books_df)
        st.caption(f"Welcome: {user_role}")
        render_metrics(books_df, selected_month, section)
        render_table(running_books, f"{section.capitalize()} Running", column_sizes_running, config["color"], section, config["role"], is_running=True)
        render_table(pending_books, f"{section.capitalize()} Pending", column_sizes_pending, config["color"], section, config["role"], is_running=False)





