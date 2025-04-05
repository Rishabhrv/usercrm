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

user_role = st.session_state.get("role", "Guest")


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

# --- Helper Functions ---
def get_status(writing_start, writing_end, current_date):
    if pd.isnull(writing_start) or writing_start == '0000-00-00 00:00:00':
        return "Pending", None
    elif (pd.notnull(writing_start) and writing_start != '0000-00-00 00:00:00' and pd.isnull(writing_end)):
        writing_days = (current_date - writing_start.date()).days
        return "Running", writing_days
    return "-", None

def get_days_since_enrolled(enroll_date, current_date):
    if pd.notnull(enroll_date):
        date_enrolled = enroll_date if isinstance(enroll_date, datetime) else pd.to_datetime(enroll_date).date()
        return (current_date - date_enrolled).days
    return None

def get_writing_by(writing_start, writing_by):
    if pd.notnull(writing_start) and writing_start != '0000-00-00 00:00:00':
        return writing_by if pd.notnull(writing_by) else "Unknown Writer"
    return "Not Assigned"

# Fetch all books with configurable months
def fetch_books(months_back: int = 3) -> pd.DataFrame:
    conn = connect_db()
    cutoff_date = datetime.now().date() - timedelta(days=30 * months_back)
    cutoff_date_str = cutoff_date.strftime('%Y-%m-%d')
    
    query = f"""
        SELECT 
            book_id AS 'Book ID',
            title AS 'Title',
            date AS 'Date',
            writing_by AS 'Writing By',
            writing_start AS 'Writing Start',
            writing_end AS 'Writing End',
            is_publish_only AS 'Is Publish Only'
        FROM books 
        WHERE is_publish_only = 0
        AND date >= '{cutoff_date_str}'
        ORDER BY date DESC
    """
    df = conn.query(query, show_spinner=False)
    df['Date'] = pd.to_datetime(df['Date']).dt.date  # Normalize to date for consistency
    return df

# --- Dialog Functions ---
def fetch_unique_names(column, conn):
    query = f"SELECT DISTINCT {column} AS name FROM books WHERE {column} IS NOT NULL AND {column} != ''"
    return sorted(conn.query(query, show_spinner=False)['name'].tolist())

def fetch_book_details(book_id, conn):
    query = f"SELECT title FROM books WHERE book_id = {book_id}"
    return conn.query(query, show_spinner=False)

@st.dialog("Edit Writing Details", width='large')
def edit_writing_dialog(book_id, conn):
    # Assuming this remains unchanged as per your comment
    book_details = fetch_book_details(book_id, conn)
    if not book_details.empty:
        book_title = book_details.iloc[0]['title']
        st.markdown(f"<h3 style='color:#4CAF50;'>{book_id} : {book_title}</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"### Writing Details for Book ID: {book_id}")
        st.warning("Book title not found.")

    query = f"SELECT writing_start, writing_end, writing_by FROM books WHERE book_id = {book_id}"
    book_data = conn.query(query, show_spinner=False)
    current_data = book_data.iloc[0].to_dict() if not book_data.empty else {}

    writing_names = fetch_unique_names("writing_by", conn)
    writing_options = ["Select Writer"] + writing_names + ["Add New..."]

    if f"writing_by_{book_id}" not in st.session_state:
        st.session_state[f"writing_by_{book_id}"] = current_data.get('writing_by', "")
    if f"writing_start_date_{book_id}" not in st.session_state:
        st.session_state[f"writing_start_date_{book_id}"] = current_data.get('writing_start', None)
    if f"writing_start_time_{book_id}" not in st.session_state:
        st.session_state[f"writing_start_time_{book_id}"] = current_data.get('writing_start', None)
    if f"writing_end_date_{book_id}" not in st.session_state:
        st.session_state[f"writing_end_date_{book_id}"] = current_data.get('writing_end', None)
    if f"writing_end_time_{book_id}" not in st.session_state:
        st.session_state[f"writing_end_time_{book_id}"] = current_data.get('writing_end', None)

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

    with st.form(key=f"writing_form_{book_id}", border=False):
        st.markdown('<div class="field-label">Writer</div>', unsafe_allow_html=True)
        selected_writer = st.selectbox(
            "",
            writing_options,
            index=writing_options.index(st.session_state[f"writing_by_{book_id}"]) if st.session_state[f"writing_by_{book_id}"] in writing_names else 0,
            key=f"writing_select_{book_id}",
            label_visibility="collapsed",
            help="Select an existing writer or add a new one."
        )
        if selected_writer == "Add New...":
            writing_by = st.text_input(
                "",
                value="",
                key=f"writing_new_input_{book_id}",
                placeholder="Enter new writer name...",
                label_visibility="collapsed"
            )
        else:
            writing_by = selected_writer if selected_writer != "Select Writer" else st.session_state[f"writing_by_{book_id}"]
        st.session_state[f"writing_by_{book_id}"] = writing_by

        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.markdown('<div class="field-label">Start Date & Time</div>', unsafe_allow_html=True)
            writing_start_date = st.date_input(
                "",
                value=st.session_state[f"writing_start_date_{book_id}"],
                key=f"writing_start_date_{book_id}",
                label_visibility="collapsed",
                help="When writing began"
            )
            writing_start_time = st.time_input(
                "",
                value=st.session_state[f"writing_start_time_{book_id}"],
                key=f"writing_start_time_{book_id}",
                label_visibility="collapsed"
            )
        with col2:
            st.markdown('<div class="field-label">End Date & Time</div>', unsafe_allow_html=True)
            writing_end_date = st.date_input(
                "",
                value=st.session_state[f"writing_end_date_{book_id}"],
                key=f"writing_end_date_{book_id}",
                label_visibility="collapsed",
                help="When writing was completed (leave blank if ongoing)"
            )
            writing_end_time = st.time_input(
                "",
                value=st.session_state[f"writing_end_time_{book_id}"],
                key=f"writing_end_time_{book_id}",
                label_visibility="collapsed"
            )

        writing_start = f"{writing_start_date} {writing_start_time}" if writing_start_date and writing_start_time else None
        writing_end = f"{writing_end_date} {writing_end_time}" if writing_end_date and writing_end_time else None

        col_save, col_cancel = st.columns([1, 1])
        with col_save:
            submit = st.form_submit_button("💾 Save and Close", use_container_width=True)
        with col_cancel:
            cancel = st.form_submit_button("Cancel", use_container_width=True, type="secondary")

        if submit:
            if writing_start and writing_end and writing_start > writing_end:
                st.error("Start must be before End.")
            else:
                with st.spinner("Saving Writing details..."):
                    time.sleep(2)
                    updates = {
                        "writing_start": writing_start,
                        "writing_end": writing_end,
                        "writing_by": writing_by if writing_by and writing_by != "Select Writer" else None
                    }
                    with conn.session as session:
                        set_clause = ", ".join([f"{key} = :{key}" for key in updates.keys()])
                        query = f"UPDATE books SET {set_clause} WHERE book_id = :id"
                        params = updates.copy()
                        params["id"] = int(book_id)
                        session.execute(text(query), params)
                        session.commit()
                    st.success("✔️ Updated Writing details")
                    time.sleep(1)
                    st.rerun()

        elif cancel:
            st.session_state.pop(f"writing_by_{book_id}", None)
            st.session_state.pop(f"writing_start_date_{book_id}", None)
            st.session_state.pop(f"writing_start_time_{book_id}", None)
            st.session_state.pop(f"writing_end_date_{book_id}", None)
            st.session_state.pop(f"writing_end_time_{book_id}", None)
            st.rerun()

# Custom CSS for Table
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
    .writing-start-not {
        background-color: #F5F5F5;
        color: #757575;
    }
    .writing-start-date {
        background-color: #FFF3E0;
        color: #FF9800;
    }
    .writing-by-not {
        background-color: #F5F5F5;
        color: #757575;
    }
    /* Writer-specific colors (softer tones) */
    .writing-by-0 { background-color: #E3F2FD; color: #1976D2; } /* Blue */
    .writing-by-1 { background-color: #FCE4EC; color: #D81B60; } /* Pink */
    .writing-by-2 { background-color: #E0F7FA; color: #006064; } /* Cyan */
    .writing-by-3 { background-color: #F1F8E9; color: #558B2F; } /* Light Green */
    .writing-by-4 { background-color: #FFF3E0; color: #EF6C00; } /* Orange */
    .writing-by-5 { background-color: #F3E5F5; color: #8E24AA; } /* Purple */
    .writing-by-6 { background-color: #FFFDE7; color: #F9A825; } /* Yellow */
    .writing-by-7 { background-color: #EFEBE9; color: #5D4037; } /* Brown */
    .writing-by-8 { background-color: #E0E0E0; color: #424242; } /* Grey */
    .writing-by-9 { background-color: #E8EAF6; color: #283593; } /* Indigo */
    .status-pending {
        background-color: #FFF3E0;
        color: #FF9800;
        font-weight: bold;          /* Emphasize status */

    }
    .status-running {
        background-color: #E8F5E9;  /* Keep green but softer */
        color: #4CAF50;

        font-weight: bold;          /* Emphasize status */
    }
    .status-badge-green {
        background-color: #E8F5E9;
        color: #4CAF50;
        padding: 4px 8px;
        border-radius: 12px;
        font-weight: bold;
        display: inline-flex;
        align-items: center;
    }
    .status-badge-red {
        background-color: #FFEBEE;
        color: #F44336;
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

# Fetch books for the last 3 months
all_books = fetch_books(months_back=4)

# Filter pending books for table
pending_books = all_books[
    (all_books['Writing Start'].isnull() | (all_books['Writing Start'] == '0000-00-00 00:00:00')) |
    ((all_books['Writing Start'].notnull()) & (all_books['Writing Start'] != '0000-00-00 00:00:00') & (all_books['Writing End'].isnull()))
]

# Extract unique months from all_books
unique_months = sorted(all_books['Date'].apply(lambda x: x.strftime('%B %Y')).unique(), key=lambda x: datetime.strptime(x, '%B %Y'), reverse=False)
default_month = unique_months[-1]  # Most recent month

selected_month = st.pills("Select Month", unique_months, default=default_month, key="month_selector", label_visibility='collapsed')

# Filter books for selected month for metrics
selected_month_dt = datetime.strptime(selected_month, '%B %Y')
month_start = selected_month_dt.replace(day=1).date()
month_end = (selected_month_dt.replace(day=1) + timedelta(days=31)).replace(day=1).date() - timedelta(days=1)
filtered_books = all_books[(all_books['Date'] >= month_start) & (all_books['Date'] <= month_end)]


col1, col2 = st.columns([11, 1], gap="large", vertical_alignment="bottom")

with col1:
    # Metrics
    st.subheader(f"Metrics of {selected_month}")

with col2:
    if st.button(":material/refresh: Refresh", key="refresh_books", type="tertiary"):
        st.cache_data.clear()

total_pending = len(all_books[
    all_books['Writing Start'].isnull() |
    (all_books['Writing Start'] == '0000-00-00 00:00:00')
])

books_in_selected_month = len(filtered_books)
books_written_selected_month = len(filtered_books[
    filtered_books['Writing End'].notnull() &
    (filtered_books['Writing End'] != '0000-00-00 00:00:00')
])
books_pending_selected_month = len(filtered_books[
    filtered_books['Writing Start'].isnull() |
    (filtered_books['Writing Start'] == '0000-00-00 00:00:00')
])
books_in_written_status = len(all_books[
    all_books['Writing Start'].notnull() &
    (all_books['Writing Start'] != '0000-00-00 00:00:00') &
    (all_books['Writing End'].isnull() | (all_books['Writing End'] == '0000-00-00 00:00:00'))
])

col1, col2, col3, col4, col5 = st.columns(5, border=True)
with col1:
    st.metric("Total Pending Books", total_pending)
with col2:
    st.metric(f"Books in {selected_month}", books_in_selected_month)
with col3:
    st.metric(f"Written in {selected_month}", books_written_selected_month)
with col4:
    st.metric(f"Pending in {selected_month}", books_pending_selected_month)
with col5:
    st.metric("Currently Running", books_in_written_status)


# Split pending_books into Running and Pending
running_books = pending_books[
    pending_books['Writing Start'].notnull() & 
    (pending_books['Writing Start'] != '0000-00-00 00:00:00') & 
    pending_books['Writing End'].isnull()
].sort_values(by='Date', ascending=False)

pending_books_only = pending_books[
    pending_books['Writing Start'].isnull() | 
    (pending_books['Writing Start'] == '0000-00-00 00:00:00')
].sort_values(by='Date', ascending=False)

# Update get_writing_by to assign writer indices
def get_writing_by(writing_start, writing_by, writer_map=None):
    if pd.notnull(writing_start) and writing_start != '0000-00-00 00:00:00':
        writer = writing_by if pd.notnull(writing_by) else "Unknown Writer"
        if writer_map and writer in writer_map:
            return writer, writer_map[writer]
        return writer, None  # Default if no map or writer not found
    return "Not Assigned", None

# Function to render a table
def render_table(books_df, title, column_size, color, is_running=False):
    if books_df.empty:
        st.warning(f"No {title.lower()} books available from the last 3 months.")
    else:
        cont = st.container(border=True)
        with cont:
            count = len(books_df)
            st.markdown(
                f"<h5><span class='status-badge-{color}'>{title} Books <span class='badge-count'>{count}</span></span></h5>",
                unsafe_allow_html=True
            )
            st.markdown('<div class="header-row">', unsafe_allow_html=True)
            
            # Adjust columns based on table type
            if is_running:
                col1, col2, col3, col4, col5, col6, col7 = st.columns(column_size)
                with col1:
                    st.markdown('<span class="header">Book ID</span>', unsafe_allow_html=True)
                with col2:
                    st.markdown('<span class="header">Title</span>', unsafe_allow_html=True)
                with col3:
                    st.markdown('<span class="header">Date</span>', unsafe_allow_html=True)
                with col4:
                    st.markdown('<span class="header">Status</span>', unsafe_allow_html=True)
                with col5:
                    st.markdown('<span class="header">Writing Start</span>', unsafe_allow_html=True)
                with col6:
                    st.markdown('<span class="header">Writing By</span>', unsafe_allow_html=True)
                with col7:
                    st.markdown('<span class="header">Action</span>', unsafe_allow_html=True)
            else:
                col1, col2, col3, col4, col5 = st.columns(column_size[:5])  # Exclude Writing Start and Writing By
                with col1:
                    st.markdown('<span class="header">Book ID</span>', unsafe_allow_html=True)
                with col2:
                    st.markdown('<span class="header">Title</span>', unsafe_allow_html=True)
                with col3:
                    st.markdown('<span class="header">Date</span>', unsafe_allow_html=True)
                with col4:
                    st.markdown('<span class="header">Status</span>', unsafe_allow_html=True)
                with col5:
                    st.markdown('<span class="header">Action</span>', unsafe_allow_html=True)
            st.markdown('</div><div class="header-line"></div>', unsafe_allow_html=True)

            current_date = datetime.now().date()
            # Create a writer map for Running table
            if is_running:
                unique_writers = [w for w in books_df['Writing By'].unique() if pd.notnull(w) and w != "Not Assigned"]
                writer_map = {writer: idx % 10 for idx, writer in enumerate(unique_writers)}  # Limit to 10 colors
            else:
                writer_map = None

            for _, row in books_df.iterrows():
                if is_running:
                    col1, col2, col3, col4, col5, col6, col7 = st.columns(column_size)
                else:
                    col1, col2, col3, col4, col5 = st.columns(column_size[:5])

                with col1:
                    st.write(row['Book ID'])
                with col2:
                    st.write(row['Title'])
                with col3:
                    st.write(row['Date'].strftime('%Y-%m-%d') if pd.notnull(row['Date']) else "-")
                with col4:
                    status, writing_days = get_status(row['Writing Start'], row['Writing End'], current_date)
                    days_since = get_days_since_enrolled(row['Date'], current_date)
                    status_html = f'<span class="pill status-{"pending" if status == "Pending" else "running"}">{status}'
                    if writing_days is not None:
                        status_html += f' {writing_days}d'
                    status_html += '</span>'
                    # Only show since-enrolled for Pending table
                    if not is_running and days_since is not None:
                        status_html += f'<span class="since-enrolled">{days_since}d</span>'
                    st.markdown(status_html, unsafe_allow_html=True)
                if is_running:
                    with col5:
                        if pd.notnull(row['Writing Start']) and row['Writing Start'] != '0000-00-00 00:00:00':
                            st.markdown(f'<span class="pill writing-start-date">{row["Writing Start"].strftime("%d %B %Y")}</span>', 
                                       unsafe_allow_html=True)
                        else:
                            st.markdown('<span class="pill writing-start-not">Not started</span>', unsafe_allow_html=True)
                    with col6:
                        writing_by, writer_idx = get_writing_by(row['Writing Start'], row['Writing By'], writer_map)
                        class_name = f"writing-by-{writer_idx}" if writer_idx is not None else "writing-by-not"
                        st.markdown(f'<span class="pill {class_name}">{writing_by}</span>', unsafe_allow_html=True)
                    with col7:
                        if st.button("Edit", key=f"edit_{row['Book ID']}"):
                            edit_writing_dialog(row['Book ID'], conn)
                else:
                    with col5:
                        if st.button("Edit", key=f"edit_{row['Book ID']}"):
                            edit_writing_dialog(row['Book ID'], conn)

# Column sizes remain the same as before
column_size_dict_running = {
    "Book ID": 0.7,
    "Title": 5.5,
    "Date": 1,
    "Status": 1,
    "Writing Start": 1.2,
    "Writing By": 1,
    "Action": 1
}
column_size_dict_pending = {
    "Book ID": 0.7,
    "Title": 5.5,
    "Date": 1,
    "Status": 1,
    "Action": 1
}

# Render both tables
render_table(running_books, "Running", list(column_size_dict_running.values()), 'green', is_running=True)
render_table(pending_books_only, "Pending", list(column_size_dict_pending.values()), 'red', is_running=False)





