import streamlit as st
import warnings 
import pandas as pd
import os
import jwt
from sqlalchemy import text
warnings.simplefilter('ignore')
import requests
import datetime


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
            padding-top: 47px !important;  /* Small padding for breathing room */
        }
            
    </style>
""", unsafe_allow_html=True)


# Configuration
FLASK_VALIDATE_URL = "https://crmserver.agvolumes.com/validate_token"
FLASK_USER_DETAILS_URL = "https://crmserver.agvolumes.com/user_details"
JWT_SECRET = st.secrets["general"]["JWT_SECRET"]
FLASK_LOGIN_URL = "https://crmserver.agvolumes.com/login"
FLASK_LOGOUT_URL = "https://crmserver.agvolumes.com/logout"
VALID_ROLES = {"admin", "user"}
VALID_APPS = {"main", "operations"}


# Configuration
# FLASK_VALIDATE_URL = "http://localhost:5000/validate_token"
# JWT_SECRET = st.secrets["general"]["JWT_SECRET"]
# FLASK_USER_DETAILS_URL = "http://localhost:5000/user_details"
# FLASK_LOGIN_URL = "http://localhost:5000/login"
# FLASK_LOGOUT_URL = "http://localhost:5000/logout"
# VALID_ROLES = {"admin", "user"}
# VALID_APPS = {"main", "operations"}

def validate_token():
    # Check if token exists in session state or query params
    if 'token' not in st.session_state:
        token = st.query_params.get("token")
        if not token:
            st.error("Access denied: Please log in first")
            st.markdown(f"[Go to Login]({FLASK_LOGIN_URL})")
            st.stop()
        st.session_state.token = token if isinstance(token, str) else token[0]

    token = st.session_state.token

    try:
        # Local validation: only check for user_id and exp
        decoded = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        if 'user_id' not in decoded or 'exp' not in decoded:
            raise jwt.InvalidTokenError("Missing user_id or exp")

        # Server-side token validation
        response = requests.post(FLASK_VALIDATE_URL, json={"token": token}, timeout=5)
        if response.status_code != 200 or not response.json().get('valid'):
            error = response.json().get('error', 'Invalid token')
            raise jwt.InvalidTokenError(error)

        # Fetch user details
        details_response = requests.post(FLASK_USER_DETAILS_URL, json={"token": token}, timeout=5)
        if details_response.status_code != 200 or not details_response.json().get('valid'):
            error = details_response.json().get('error', 'Unable to fetch user details')
            raise jwt.InvalidTokenError(f"User details error: {error}")

        user_details = details_response.json()
        role = user_details['role'].lower()
        app = user_details['app'].lower()
        access = user_details['access']

        if role not in VALID_ROLES:
            raise jwt.InvalidTokenError(f"Invalid role '{role}'")
        if app not in VALID_APPS:
            raise jwt.InvalidTokenError(f"Invalid app '{app}'")
        
        if app == 'operations':
            valid_access = {"writer", "proofreader", "formatter", "cover_designer"}
            if not (len(access) == 1 and access[0] in valid_access):
                raise jwt.InvalidTokenError(f"Invalid access for operations app: {access}")

        st.session_state.access = access

    except jwt.ExpiredSignatureError:
        st.error("Access denied: Token expired. Please log in again.")
        st.markdown(f"[Go to Login]({FLASK_LOGIN_URL})")
        clear_auth_session()
        st.stop()
    except jwt.InvalidSignatureError:
        st.error("Access denied: Invalid token signature. Please log in again.")
        st.markdown(f"[Go to Login]({FLASK_LOGIN_URL})")
        clear_auth_session()
        st.stop()
    except jwt.DecodeError:
        st.error("Access denied: Token decoding failed. Please log in again.")
        st.markdown(f"[Go to Login]({FLASK_LOGIN_URL})")
        clear_auth_session()
        st.stop()
    except jwt.InvalidTokenError as e:
        st.error(f"Access denied: {str(e)}. Please log in again.")
        st.markdown(f"[Go to Login]({FLASK_LOGIN_URL})")
        clear_auth_session()
        st.stop()
    except requests.RequestException:
        st.error("Access denied: Unable to contact authentication server. Please try again later.")
        st.markdown(f"[Go to Login]({FLASK_LOGIN_URL})")
        clear_auth_session()
        st.stop()

def clear_auth_session():
    # Clear authentication-related session state keys
    keys_to_clear = ['token', 'email', 'role', 'app', 'access', 'exp']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    # Clear query parameters to prevent token reuse
    st.query_params.clear()


# Run validation
validate_token()

user_role = st.session_state.get("access", [])[0]

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

def fetch_books(months_back: int = 4, section: str = "writing") -> pd.DataFrame:
    conn = connect_db()
    cutoff_date = datetime.now().date() - timedelta(days=30 * months_back)
    cutoff_date_str = cutoff_date.strftime('%Y-%m-%d')
    
    section_columns = {
        "writing": {
            "base": [
                "writing_by AS 'Writing By'", 
                "writing_start AS 'Writing Start'", 
                "writing_end AS 'Writing End'",
                "book_pages AS 'Number of Book Pages'",  # Added here for Completed table
                "syllabus_path AS 'Syllabus Path'"
            ],
            "extra": [],
            "publish_filter": "AND is_publish_only = 0"
        },
        "proofreading": {
            "base": [
                "proofreading_by AS 'Proofreading By'", 
                "proofreading_start AS 'Proofreading Start'", 
                "proofreading_end AS 'Proofreading End'"
            ],
            "extra": [
                "writing_end AS 'Writing End'", 
                "writing_by AS 'Writing By'",
                "book_pages AS 'Number of Book Pages'"  # Added here for Pending and Completed
            ],
            "publish_filter": ""
        },
        "formatting": {
            "base": [
                "formatting_by AS 'Formatting By'", 
                "formatting_start AS 'Formatting Start'", 
                "formatting_end AS 'Formatting End'",
                "book_pages AS 'Number of Book Pages'"  # Moved from extra, renamed
            ],
            "extra": ["proofreading_end AS 'Proofreading End'"],
            "publish_filter": ""
        },
        "cover": {
            "base": [
                "cover_by AS 'Cover By'", 
                "front_cover_start AS 'Front Cover Start'", 
                "front_cover_end AS 'Front Cover End'", 
                "back_cover_start AS 'Back Cover Start'", 
                "back_cover_end AS 'Back Cover End'", 
                "apply_isbn AS 'Apply ISBN'", 
                "isbn AS 'ISBN'"
            ],
            "extra": [
                "formatting_end AS 'Formatting End'",
                "(SELECT MIN(ba.photo_recive) FROM book_authors ba WHERE ba.book_id = b.book_id) AS 'All Photos Received'",
                "(SELECT MIN(ba.author_details_sent) FROM book_authors ba WHERE ba.book_id = b.book_id) AS 'All Details Sent'"
            ],
            "publish_filter": ""
        }
    }
    config = section_columns.get(section, section_columns["writing"])
    columns = config["base"] + config["extra"]
    columns_str = ", ".join(columns)
    publish_filter = config["publish_filter"]
    
    if section == "cover":
        query = f"""
            SELECT 
                b.book_id AS 'Book ID',
                b.title AS 'Title',
                b.date AS 'Date',
                {columns_str},
                b.is_publish_only AS 'Is Publish Only',
                GROUP_CONCAT(CONCAT(a.name, ' (Pos: ', ba.author_position, ', Photo: ', ba.photo_recive, ', Sent: ', ba.author_details_sent, ')') SEPARATOR ', ') AS 'Author Details'
            FROM books b
            LEFT JOIN book_authors ba ON b.book_id = ba.book_id
            LEFT JOIN authors a ON ba.author_id = a.author_id
            WHERE b.date >= '{cutoff_date_str}'
            {publish_filter}
            GROUP BY b.book_id, b.title, b.date, b.cover_by, b.front_cover_start, b.front_cover_end, b.back_cover_start, b.back_cover_end, b.apply_isbn, b.isbn, b.is_publish_only
            ORDER BY b.date DESC
        """
    else:
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

def fetch_author_details(book_id):
    conn = connect_db()
    query = f"""
        SELECT
            ba.author_id AS 'Author ID', 
            a.name AS 'Author Name',
            ba.author_position AS 'Position',
            ba.photo_recive AS 'Photo Received',
            ba.author_details_sent AS 'Details Sent'
        FROM book_authors ba
        JOIN authors a ON ba.author_id = a.author_id
        WHERE ba.book_id = {book_id}
    """
    df = conn.query(query, show_spinner=False)
    return df


@st.dialog("Author Details", width='large')
def show_author_details_dialog(book_id):
    # Fetch book details (title and ISBN)
    conn = connect_db()
    book_query = f"SELECT title, isbn FROM books WHERE book_id = {book_id}"
    book_data = conn.query(book_query, show_spinner=False)
    book_title = book_data.iloc[0]['title'] if not book_data.empty else "Unknown Title"
    isbn = book_data.iloc[0]['isbn'] if not book_data.empty and pd.notnull(book_data.iloc[0]['isbn']) else "Not Assigned"

    # Fetch author details
    author_details_df = fetch_author_details(book_id)
    
    # Custom CSS for table styling
    st.markdown("""
        <style>
        .dialog-header {
            font-size: 20px;
            color: #4CAF50;
            margin-bottom: 10px;
            font-weight: bold;
        }
        .info-label {
            font-weight: bold;
            color: #333;
            margin-top: 10px;
            margin-bottom: 5px;
        }
        .info-value {
            padding: 5px 10px;
            border-radius: 5px;
            background-color: #F5F5F5;
            display: inline-block;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        th {
            background-color: #4CAF50;
            color: white;
            padding: 10px;
            text-align: left;
            font-weight: bold;
        }
        td {
            padding: 10px;
            border-bottom: 1px solid #E0E0E0;
        }
        .pill-yes {
            background-color: #C8E6C9;
            color: #2E7D32;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
        }
        .pill-no {
            background-color: #E0E0E0;
            color: #616161;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
        }
        .close-button {
            margin-top: 20px;
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown(f'<div class="dialog-header">Book ID: {book_id} - {book_title}</div>', unsafe_allow_html=True)

    # ISBN Display
    st.markdown('<div class="info-label">ISBN</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="info-value">{isbn}</span>', unsafe_allow_html=True)

    # Author Details Table
    if not author_details_df.empty:
        # Prepare HTML table
        table_html = '<table><tr><th>Author ID</th><th>Author Name</th><th>Position</th><th>Photo Received</th><th>Details Received</th></tr>'
        for _, row in author_details_df.iterrows():
            photo_class = "pill-yes" if row["Photo Received"] else "pill-no"
            details_class = "pill-yes" if row["Details Sent"] else "pill-no"
            table_html += (
                f'<tr>'
                f'<td>{row["Author ID"]}</td>'
                f'<td>{row["Author Name"]}</td>'
                f'<td>{row["Position"]}</td>'
                f'<td><span class="{photo_class}">{"Yes" if row["Photo Received"] else "No"}</span></td>'
                f'<td><span class="{details_class}">{"Yes" if row["Details Sent"] else "No"}</span></td>'
                f'</tr>'
            )
        table_html += '</table>'
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.warning("No author details available.")

# --- Reusable Month Selector ---
def render_month_selector(books_df):
    unique_months = sorted(books_df['Date'].apply(lambda x: x.strftime('%B %Y')).unique(), 
                          key=lambda x: datetime.strptime(x, '%B %Y'), reverse=False)
    default_month = unique_months[-1]  # Most recent month
    selected_month = st.pills("Select Month", unique_months, default=default_month, 
                             key=f"month_selector_{st.session_state.get('section', 'writing')}", 
                             label_visibility='collapsed')
    return selected_month

from datetime import datetime, timedelta

def render_metrics(books_df, selected_month, section):
    # Convert selected_month (e.g., "April 2025") to date range
    selected_month_dt = datetime.strptime(selected_month, '%B %Y')
    month_start = selected_month_dt.replace(day=1).date()
    month_end = (selected_month_dt.replace(day=1) + timedelta(days=31)).replace(day=1).date() - timedelta(days=1)
    filtered_books = books_df[(books_df['Date'] >= month_start) & (books_df['Date'] <= month_end)]

    total_books = len(filtered_books)
    
    # Adjust completion and pending logic for cover section
    if section == "cover":
        completed_books = len(filtered_books[
            filtered_books['Front Cover End'].notnull() & 
            (filtered_books['Front Cover End'] != '0000-00-00 00:00:00') & 
            filtered_books['Back Cover End'].notnull() & 
            (filtered_books['Back Cover End'] != '0000-00-00 00:00:00')
        ])
        pending_books = len(filtered_books[
            filtered_books['Formatting End'].notnull() & 
            (filtered_books['Formatting End'] != '0000-00-00 00:00:00') & 
            (filtered_books['Front Cover Start'].isnull() | (filtered_books['Front Cover Start'] == '0000-00-00 00:00:00')) & 
            (filtered_books['Back Cover Start'].isnull() | (filtered_books['Back Cover Start'] == '0000-00-00 00:00:00'))
        ])
    else:
        completed_books = len(filtered_books[
            filtered_books[f'{section.capitalize()} End'].notnull() & 
            (filtered_books[f'{section.capitalize()} End'] != '0000-00-00 00:00:00')
        ])
        pending_books = len(filtered_books[
            filtered_books[f'{section.capitalize()} Start'].isnull() | 
            (filtered_books[f'{section.capitalize()} Start'] == '0000-00-00 00:00:00')
        ])

    # Render UI
    col1, col2  = st.columns([11, 1], gap="large", vertical_alignment="bottom")
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


def fetch_book_details(book_id, conn):
    query = f"SELECT title FROM books WHERE book_id = {book_id}"
    return conn.query(query, show_spinner=False)


from time import sleep
from sqlalchemy.sql import text

@st.dialog("Rate User", width='large')
def rate_user_dialog(book_id, conn):
    # Fetch book title
    book_details = fetch_book_details(book_id, conn)
    if not book_details.empty:
        book_title = book_details.iloc[0]['title']
        st.markdown(f"<h3 style='color:#4CAF50;'>{book_id} : {book_title}</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"### Rate User for Book ID: {book_id}")
        st.warning("Book title not found.")
    
    sentiment_mapping = ["one", "two", "three", "four", "five"]
    selected = st.feedback("stars")
    if selected is not None:
        st.markdown(f"You selected {sentiment_mapping[selected]} star(s).")


@st.dialog("Edit Section Details", width='large')
def edit_section_dialog(book_id, conn, section):
    # Map section to display name and database columns
    section_config = {
        "writing": {"display": "Writing", "by": "writing_by", "start": "writing_start", "end": "writing_end"},
        "proofreading": {"display": "Proofreading", "by": "proofreading_by", "start": "proofreading_start", "end": "proofreading_end"},
        "formatting": {"display": "Formatting", "by": "formatting_by", "start": "formatting_start", "end": "formatting_end"},
        "cover": {
            "display": "Cover Page",
            "by": "cover_by",
            "front_start": "front_cover_start",
            "front_end": "front_cover_end",
            "back_start": "back_cover_start",
            "back_end": "back_cover_end"
        }
    }
    
    config = section_config.get(section, section_config["writing"])  # Default to writing if section invalid
    display_name = config["display"]

    # Fetch book title and current book_pages
    book_details = fetch_book_details(book_id, conn)
    if not book_details.empty:
        book_title = book_details.iloc[0]['title']
        current_book_pages = book_details.iloc[0].get('book_pages', 0)  # Fetch existing book_pages
        st.markdown(f"<h3 style='color:#4CAF50;'>{book_id} : {book_title}</h3>", unsafe_allow_html=True)
    else:
        book_title = "Unknown Title"
        current_book_pages = 0
        st.markdown(f"### {display_name} Details for Book ID: {book_id}")
        st.warning("Book title not found.")

    # Fetch current section data
    if section == "cover":
        query = f"""
            SELECT 
                front_cover_start AS 'Front Cover Start', 
                front_cover_end AS 'Front Cover End', 
                back_cover_start AS 'Back Cover Start', 
                back_cover_end AS 'Back Cover End', 
                cover_by AS 'Cover By'
            FROM books 
            WHERE book_id = {book_id}
        """
    else:
        query = f"SELECT {config['start']}, {config['end']}, {config['by']}, book_pages FROM books WHERE book_id = {book_id}"
    book_data = conn.query(query, show_spinner=False)
    current_data = book_data.iloc[0].to_dict() if not book_data.empty else {}

    # Fetch unique names for the section
    names = fetch_unique_names(config["by"], conn)
    options = ["Select Worker"] + names + ["Add New..."]

    # Initialize session state
    if section == "cover":
        keys = [
            f"{section}_by",
            f"{section}_front_start_date", f"{section}_front_start_time",
            f"{section}_front_end_date", f"{section}_front_end_time",
            f"{section}_back_start_date", f"{section}_back_start_time",
            f"{section}_back_end_date", f"{section}_back_end_time"
        ]
        defaults = {
            "cover_by": current_data.get("Cover By", ""),
            "cover_front_start_date": current_data.get("Front Cover Start", None),
            "cover_front_start_time": current_data.get("Front Cover Start", None),
            "cover_front_end_date": current_data.get("Front Cover End", None),
            "cover_front_end_time": current_data.get("Front Cover End", None),
            "cover_back_start_date": current_data.get("Back Cover Start", None),
            "cover_back_start_time": current_data.get("Back Cover Start", None),
            "cover_back_end_date": current_data.get("Back Cover End", None),
            "cover_back_end_time": current_data.get("Back Cover End", None)
        }
    else:
        keys = [
            f"{section}_by",
            f"{section}_start_date", f"{section}_start_time",
            f"{section}_end_date", f"{section}_end_time",
            f"book_pages"  # Add book_pages to keys for all non-cover sections
        ]
        defaults = {
            f"{section}_by": current_data.get(config["by"], ""),
            f"{section}_start_date": current_data.get(config["start"], None),
            f"{section}_start_time": current_data.get(config["start"], None),
            f"{section}_end_date": current_data.get(config["end"], None),
            f"{section}_end_time": current_data.get(config["end"], None),
            f"book_pages": current_data.get("book_pages", current_book_pages)
        }
    
    for key in keys:
        if f"{key}_{book_id}" not in st.session_state:
            st.session_state[f"{key}_{book_id}"] = defaults[key]
    
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
            worker = new_worker.strip() if new_worker.strip() else None
        else:
            worker = selected_worker if selected_worker != "Select Worker" else None
        if worker:
            st.session_state[f"{section}_by_{book_id}"] = worker

        if section == "cover":
            # Front Cover
            st.markdown('<div class="field-label">Front Cover</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2, gap="medium")
            with col1:
                front_start_date = st.date_input(
                    "Start Date",
                    value=st.session_state[f"{section}_front_start_date_{book_id}"],
                    key=f"{section}_front_start_date_{book_id}",
                    label_visibility="collapsed",
                    help="When front cover design began"
                )
                front_start_time = st.time_input(
                    "Start Time",
                    value=st.session_state[f"{section}_front_start_time_{book_id}"],
                    key=f"{section}_front_start_time_{book_id}",
                    label_visibility="collapsed"
                )
            with col2:
                front_end_date = st.date_input(
                    "End Date",
                    value=st.session_state[f"{section}_front_end_date_{book_id}"],
                    key=f"{section}_front_end_date_{book_id}",
                    label_visibility="collapsed",
                    help="When front cover design was completed"
                )
                front_end_time = st.time_input(
                    "End Time",
                    value=st.session_state[f"{section}_front_end_time_{book_id}"],
                    key=f"{section}_front_end_time_{book_id}",
                    label_visibility="collapsed"
                )
            
            # Back Cover
            st.markdown('<div class="field-label">Back Cover</div>', unsafe_allow_html=True)
            col3, col4 = st.columns(2, gap="medium")
            with col3:
                back_start_date = st.date_input(
                    "Start Date",
                    value=st.session_state[f"{section}_back_start_date_{book_id}"],
                    key=f"{section}_back_start_date_{book_id}",
                    label_visibility="collapsed",
                    help="When back cover design began"
                )
                back_start_time = st.time_input(
                    "Start Time",
                    value=st.session_state[f"{section}_back_start_time_{book_id}"],
                    key=f"{section}_back_start_time_{book_id}",
                    label_visibility="collapsed"
                )
            with col4:
                back_end_date = st.date_input(
                    "End Date",
                    value=st.session_state[f"{section}_back_end_date_{book_id}"],
                    key=f"{section}_back_end_date_{book_id}",
                    label_visibility="collapsed",
                    help="When back cover design was completed"
                )
                back_end_time = st.time_input(
                    "End Time",
                    value=st.session_state[f"{section}_back_end_time_{book_id}"],
                    key=f"{section}_back_end_time_{book_id}",
                    label_visibility="collapsed"
                )
            
            front_start = f"{front_start_date} {front_start_time}" if front_start_date and front_start_time else None
            front_end = f"{front_end_date} {front_end_time}" if front_end_date and front_end_time else None
            back_start = f"{back_start_date} {back_start_time}" if back_start_date and back_start_time else None
            back_end = f"{back_end_date} {back_end_time}" if back_end_date and back_end_time else None
        else:
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
            start = f"{start_date} {start_time}" if start_date and start_time else None
            end = f"{end_date} {end_time}" if end_date and end_time else None

        # Add Total Book Pages field for writing, proofreading, and formatting
        if section in ["writing", "proofreading", "formatting"]:
            st.markdown('<div class="field-label">Total Book Pages</div>', unsafe_allow_html=True)
            book_pages = st.number_input(
                "Total Book Pages",
                min_value=0,
                value=st.session_state[f"book_pages_{book_id}"],
                key=f"book_pages_{book_id}",
                label_visibility="collapsed",
                help="Enter the total number of pages in the book"
            )

        col_save, col_cancel = st.columns([1, 1])
        with col_save:
            submit = st.form_submit_button("💾 Save and Close", use_container_width=True)
        with col_cancel:
            cancel = st.form_submit_button("Cancel", use_container_width=True, type="secondary")

        if submit:
            if section == "cover":
                if (front_start and front_end and front_start > front_end) or (back_start and back_end and back_start > back_end):
                    st.error("Start must be before End for both front and back covers.")
                else:
                    with st.spinner(f"Saving {display_name} details..."):
                        sleep(2)
                        updates = {
                            "front_cover_start": front_start,
                            "front_cover_end": front_end,
                            "back_cover_start": back_start,
                            "back_cover_end": back_end,
                            "cover_by": worker
                        }
                        with conn.session as session:
                            set_clause = ", ".join([f"{key} = :{key}" for key in updates.keys()])
                            query = f"UPDATE books SET {set_clause} WHERE book_id = :id"
                            params = updates.copy()
                            params["id"] = int(book_id)
                            session.execute(text(query), params)
                            session.commit()
                        st.success(f"✔️ Updated {display_name} details")
                        sleep(1)
                        st.rerun()
            else:
                if start and end and start > end:
                    st.error("Start must be before End.")
                else:
                    with st.spinner(f"Saving {display_name} details..."):
                        sleep(2)
                        updates = {
                            config["start"]: start,
                            config["end"]: end,
                            config["by"]: worker,
                            "book_pages": book_pages if section in ["writing", "proofreading", "formatting"] else None
                        }
                        # Remove None values from updates
                        updates = {k: v for k, v in updates.items() if v is not None}
                        with conn.session as session:
                            set_clause = ", ".join([f"{key} = :{key}" for key in updates.keys()])
                            query = f"UPDATE books SET {set_clause} WHERE book_id = :id"
                            params = updates.copy()
                            params["id"] = int(book_id)
                            session.execute(text(query), params)
                            session.commit()
                        st.success(f"✔️ Updated {display_name} details")
                        sleep(1)
                        st.rerun()

        elif cancel:
            for key in keys:
                st.session_state.pop(f"{key}_{book_id}", None)
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
        font-size: 14px; 
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
    .apply-isbn-yes {
    background-color: #C8E6C9; /* Light green */
    color: #2E7D32; /* Dark green text */
    }
    .apply-isbn-no {
        background-color: #E0E0E0; /* Light gray */
        color: #616161; /* Dark gray text */
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
    /* ... existing styles ... */
    .status-completed {
        background-color: #E8F5E9;
        color: #4CAF50;
        font-weight: bold;
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
    </style>
""", unsafe_allow_html=True)

import os

import os

def render_table(books_df, title, column_sizes, color, section, role, is_running=False):
    if books_df.empty:
        st.warning(f"No {title.lower()} books available from the last 3 months.")
        return
    
    cont = st.container(border=True)
    with cont:
        count = len(books_df)
        badge_color = 'yellow' if "Running" in title else 'red' if "Pending" in title else 'green'
        st.markdown(f"<h5><span class='status-badge-{badge_color}'>{title} Books <span class='badge-count'>{count}</span></span></h5>", 
                    unsafe_allow_html=True)
        st.markdown('<div class="header-row">', unsafe_allow_html=True)
        
        # Base columns
        columns = ["Book ID", "Title", "Date", "Status"]
        # Role-specific additional columns
        if role == "proofreader":
            columns.append("Writing By")
            if not is_running:
                columns.append("Writing End")
            if "Pending" in title or "Completed" in title:
                columns.append("Book Pages")
            if "Pending" in title:
                columns.append("Rating")
        elif role == "formatter":
            if not is_running:
                columns.append("Proofreading End")
            if "Pending" in title or "Completed" in title:
                columns.append("Book Pages")
        elif role == "cover_designer":
            if "Pending" in title or is_running:
                columns.extend(["Apply ISBN", "Photo", "Details"])
        elif role == "writer":
            if "Completed" in title:
                columns.append("Book Pages")
            if "Pending" in title:
                columns.append("Syllabus")
        # Adjust columns based on table type
        if is_running:
            if role == "cover_designer":
                columns.extend(["Cover Status", "Cover By", "Action", "Details"])
            elif role == "proofreader":
                columns.extend(["Proofreading Start", "Proofreading By", "Rating", "Action"])
            elif role == "writer":
                columns.extend(["Writing Start", "Writing By", "Syllabus", "Action"])
            else:
                columns.extend([f"{section.capitalize()} Start", f"{section.capitalize()} By", "Action"])
        elif "Pending" in title:
            if role == "cover_designer":
                columns.extend(["Action", "Details"])
            else:
                columns.append("Action")
        elif "Completed" in title:
            if role == "cover_designer":
                columns.extend(["Front Cover End", "Back Cover End"])
            else:
                columns.append(f"{section.capitalize()} End")
        
        # Validate column sizes
        if len(column_sizes) < len(columns):
            st.error(f"Column size mismatch in {title}: {len(columns)} columns but only {len(column_sizes)} sizes provided.")
            return
        
        col_configs = st.columns(column_sizes[:len(columns)])
        for i, col in enumerate(columns):
            with col_configs[i]:
                st.markdown(f'<span class="header">{col}</span>', unsafe_allow_html=True)
        st.markdown('</div><div class="header-line"></div>', unsafe_allow_html=True)

        current_date = datetime.now().date()
        # Worker maps
        if user_role == role:
            unique_workers = [w for w in books_df[f'{section.capitalize()} By'].unique() if pd.notnull(w)]
            worker_map = {worker: idx % 10 for idx, worker in enumerate(unique_workers)}
            if role == "proofreader":
                unique_writing_workers = [w for w in books_df['Writing By'].unique() if pd.notnull(w)]
                writing_worker_map = {worker: idx % 10 for idx, worker in enumerate(unique_writing_workers)}
            else:
                writing_worker_map = None
        else:
            worker_map = None
            writing_worker_map = None

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
                if role == "cover_designer":
                    if pd.notnull(row['Front Cover End']) and pd.notnull(row['Back Cover End']):
                        status = "Completed"
                        days_ago = (current_date - row['Front Cover End'].date()).days if pd.notnull(row['Front Cover End']) else None
                        status_html = f'<span class="pill status-completed">{status}'
                        if days_ago is not None:
                            status_html += f' ({days_ago}d ago)'
                        status_html += '</span>'
                    elif pd.notnull(row['Front Cover Start']) or pd.notnull(row['Back Cover Start']):
                        status = "Running"
                        start_date = row['Front Cover Start'] if pd.notnull(row['Front Cover Start']) else row['Back Cover Start']
                        days = (current_date - start_date.date()).days if pd.notnull(start_date) else None
                        status_html = f'<span class="pill status-running">{status}'
                        if days is not None:
                            status_html += f' {days}d'
                        status_html += '</span>'
                    else:
                        status = "Pending"
                        days_since = get_days_since_enrolled(row['Date'], current_date)
                        status_html = f'<span class="pill status-pending">{status}'
                        if days_since is not None:
                            status_html += f'<span class="since-enrolled">{days_since}d</span>'
                        status_html += '</span>'
                else:
                    status, days = get_status(row[f'{section.capitalize()} Start'], row[f'{section.capitalize()} End'], current_date)
                    days_since = get_days_since_enrolled(row['Date'], current_date)
                    status_html = f'<span class="pill status-{"pending" if status == "Pending" else "running" if status == "Running" else "completed"}">{status}'
                    if days is not None and status == "Running":
                        status_html += f' {days}d'
                    elif "Completed" in title:
                        end_date = row[f'{section.capitalize()} End']
                        days_ago = (current_date - end_date.date()).days if pd.notnull(end_date) else None
                        if days_ago is not None:
                            status_html += f' ({days_ago}d ago)'
                    elif not is_running and days_since is not None:
                        status_html += f'<span class="since-enrolled">{days_since}d</span>'
                    status_html += '</span>'
                st.markdown(status_html, unsafe_allow_html=True)
            col_idx += 1
            
            # Role-specific columns
            if role == "proofreader":
                with col_configs[col_idx]:
                    writing_by = row['Writing By']
                    value = writing_by if pd.notnull(writing_by) and writing_by else "-"
                    if writing_worker_map and value != "-":
                        writing_idx = writing_worker_map.get(writing_by)
                        class_name = f"worker-by-{writing_idx}" if writing_idx is not None else "worker-by-not"
                        st.markdown(f'<span class="pill {class_name}">{value}</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span>{value}</span>', unsafe_allow_html=True)
                col_idx += 1
                if "Writing End" in columns:
                    with col_configs[col_idx]:
                        writing_end = row['Writing End']
                        value = writing_end.strftime('%Y-%m-%d') if not pd.isna(writing_end) and writing_end != '0000-00-00 00:00:00' else "-"
                        st.markdown(f'<span>{value}</span>', unsafe_allow_html=True)
                    col_idx += 1
                if "Book Pages" in columns:
                    with col_configs[col_idx]:
                        book_pages = row['Number of Book Pages']
                        value = str(book_pages) if pd.notnull(book_pages) and book_pages != 0 else "-"
                        st.markdown(f'<span>{value}</span>', unsafe_allow_html=True)
                    col_idx += 1
                if "Rating" in columns and not is_running:  # Rating for Pending only
                    with col_configs[col_idx]:
                        if st.button("Rate", key=f"rate_{section}_{row['Book ID']}"):
                            rate_user_dialog(row['Book ID'], conn)
                    col_idx += 1
            elif role == "formatter":
                if "Proofreading End" in columns:
                    with col_configs[col_idx]:
                        proofreading_end = row['Proofreading End']
                        value = proofreading_end.strftime('%Y-%m-%d') if not pd.isna(proofreading_end) and proofreading_end != '0000-00-00 00:00:00' else "-"
                        st.markdown(f'<span>{value}</span>', unsafe_allow_html=True)
                    col_idx += 1
                if "Book Pages" in columns:
                    with col_configs[col_idx]:
                        book_pages = row['Number of Book Pages']
                        value = str(book_pages) if pd.notnull(book_pages) and book_pages != 0 else "-"
                        st.markdown(f'<span>{value}</span>', unsafe_allow_html=True)
                    col_idx += 1
            elif role == "cover_designer":
                if "Apply ISBN" in columns:
                    with col_configs[col_idx]:
                        apply_isbn = row['Apply ISBN']
                        value = "Yes" if pd.notnull(apply_isbn) and apply_isbn else "No"
                        class_name = "pill apply-isbn-yes" if value == "Yes" else "pill apply-isbn-no"
                        st.markdown(f'<span class="{class_name}">{value}</span>', unsafe_allow_html=True)
                    col_idx += 1
                if "Photo" in columns:
                    with col_configs[col_idx]:
                        photo_received = row['All Photos Received']
                        value = "Yes" if pd.notnull(photo_received) and photo_received else "No"
                        class_name = "pill apply-isbn-yes" if value == "Yes" else "pill apply-isbn-no"
                        st.markdown(f'<span class="{class_name}">{value}</span>', unsafe_allow_html=True)
                    col_idx += 1
                if "Details" in columns:
                    with col_configs[col_idx]:
                        details_sent = row['All Details Sent']
                        value = "Yes" if pd.notnull(details_sent) and details_sent else "No"
                        class_name = "pill apply-isbn-yes" if value == "Yes" else "pill apply-isbn-no"
                        st.markdown(f'<span class="{class_name}">{value}</span>', unsafe_allow_html=True)
                    col_idx += 1
            elif role == "writer":
                if "Book Pages" in columns:
                    with col_configs[col_idx]:
                        book_pages = row['Number of Book Pages']
                        value = str(book_pages) if pd.notnull(book_pages) and book_pages != 0 else "-"
                        st.markdown(f'<span>{value}</span>', unsafe_allow_html=True)
                    col_idx += 1
                if "Syllabus" in columns and not is_running:  # Syllabus for Pending only
                    with col_configs[col_idx]:
                        syllabus_path = row['Syllabus Path']
                        disabled = pd.isna(syllabus_path) or not syllabus_path or not os.path.exists(syllabus_path)
                        if not disabled:
                            with open(syllabus_path, "rb") as file:
                                st.download_button(
                                    label=":material/download:",
                                    data=file,
                                    file_name=syllabus_path.split("/")[-1],
                                    mime="application/pdf",
                                    key=f"download_syllabus_{section}_{row['Book ID']}",
                                    disabled=disabled,
                                    help = "Download Syllabus"
                                )
                        else:
                            st.download_button(
                                label=":material/download:",
                                data="",
                                file_name="no_syllabus.pdf",
                                mime="application/pdf",
                                key=f"download_syllabus_{section}_{row['Book ID']}",
                                disabled=disabled,
                                help = "No Syllabus Available"
                            )
                    col_idx += 1
            
            # Running-specific columns
            if is_running and user_role == role:
                if role == "cover_designer":
                    with col_configs[col_idx]:
                        front_start = row['Front Cover Start']
                        back_start = row['Back Cover Start']
                        if pd.notnull(front_start) and pd.notnull(back_start):
                            status = "Both Running"
                            class_name = "status-running"
                        elif pd.notnull(front_start):
                            status = "Back Pending"
                            class_name = "status-pending"
                        elif pd.notnull(back_start):
                            status = "Front Pending"
                            class_name = "status-pending"
                        else:
                            status = "Both Pending"
                            class_name = "status-pending"
                        st.markdown(f'<span class="pill {class_name}">{status}</span>', unsafe_allow_html=True)
                    col_idx += 1
                    with col_configs[col_idx]:
                        worker = row['Cover By']
                        value = worker if pd.notnull(worker) else "Not Assigned"
                        st.markdown(f'<span class="pill worker-by-not">{value}</span>', unsafe_allow_html=True)
                    col_idx += 1
                    with col_configs[col_idx]:
                        if st.button("Edit", key=f"edit_{section}_{row['Book ID']}"):
                            edit_section_dialog(row['Book ID'], conn, section)
                    col_idx += 1
                    with col_configs[col_idx]:
                        if st.button("Details", key=f"details_{section}_{row['Book ID']}"):
                            show_author_details_dialog(row['Book ID'])
                elif role == "proofreader":
                    with col_configs[col_idx]:
                        start = row['Proofreading Start']
                        if pd.notnull(start) and start != '0000-00-00 00:00:00':
                            st.markdown(f'<span class="pill section-start-date">{start.strftime("%d %B %Y")}</span>', unsafe_allow_html=True)
                        else:
                            st.markdown('<span class="pill section-start-not">Not started</span>', unsafe_allow_html=True)
                    col_idx += 1
                    with col_configs[col_idx]:
                        worker, worker_idx = get_worker_by(row['Proofreading Start'], 
                                                          row['Proofreading By'], worker_map)
                        class_name = f"worker-by-{worker_idx}" if worker_idx is not None else "worker-by-not"
                        st.markdown(f'<span class="pill {class_name}">{worker}</span>', unsafe_allow_html=True)
                    col_idx += 1
                    with col_configs[col_idx]:
                        if st.button("Rate", key=f"rate_{section}_{row['Book ID']}"):
                            rate_user_dialog(row['Book ID'], conn)
                    col_idx += 1
                    with col_configs[col_idx]:
                        if st.button("Edit", key=f"edit_{section}_{row['Book ID']}"):
                            edit_section_dialog(row['Book ID'], conn, section)
                elif role == "writer":
                    with col_configs[col_idx]:
                        start = row['Writing Start']
                        if pd.notnull(start) and start != '0000-00-00 00:00:00':
                            st.markdown(f'<span class="pill section-start-date">{start.strftime("%d %B %Y")}</span>', unsafe_allow_html=True)
                        else:
                            st.markdown('<span class="pill section-start-not">Not started</span>', unsafe_allow_html=True)
                    col_idx += 1
                    with col_configs[col_idx]:
                        worker, worker_idx = get_worker_by(row['Writing Start'], 
                                                          row['Writing By'], worker_map)
                        class_name = f"worker-by-{worker_idx}" if worker_idx is not None else "worker-by-not"
                        st.markdown(f'<span class="pill {class_name}">{worker}</span>', unsafe_allow_html=True)
                    col_idx += 1
                    with col_configs[col_idx]:
                        syllabus_path = row['Syllabus Path']
                        disabled = pd.isna(syllabus_path) or not syllabus_path or not os.path.exists(syllabus_path)
                        if not disabled:
                            with open(syllabus_path, "rb") as file:
                                st.download_button(
                                    label=":material/download:",
                                    data=file,
                                    file_name=syllabus_path.split("/")[-1],
                                    mime="application/pdf",
                                    key=f"download_syllabus_{section}_{row['Book ID']}_running",
                                    disabled=disabled
                                )
                        else:
                            st.download_button(
                                label=":material/download:",
                                data="",
                                file_name="no_syllabus.pdf",
                                mime="application/pdf",
                                key=f"download_syllabus_{section}_{row['Book ID']}_running",
                                disabled=disabled
                            )
                    col_idx += 1
                    with col_configs[col_idx]:
                        if st.button("Edit", key=f"edit_{section}_{row['Book ID']}"):
                            edit_section_dialog(row['Book ID'], conn, section)
                else:
                    with col_configs[col_idx]:
                        start = row[f'{section.capitalize()} Start']
                        if pd.notnull(start) and start != '0000-00-00 00:00:00':
                            st.markdown(f'<span class="pill section-start-date">{start.strftime("%d %B %Y")}</span>', unsafe_allow_html=True)
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
            # Pending-specific column
            elif "Pending" in title and user_role == role:
                if role == "cover_designer":
                    with col_configs[col_idx]:
                        if st.button("Edit", key=f"edit_{section}_{row['Book ID']}"):
                            edit_section_dialog(row['Book ID'], conn, section)
                    col_idx += 1
                    with col_configs[col_idx]:
                        if st.button("Details", key=f"details_{section}_{row['Book ID']}"):
                            show_author_details_dialog(row['Book ID'])
                else:
                    with col_configs[col_idx]:
                        if st.button("Edit", key=f"edit_{section}_{row['Book ID']}"):
                            edit_section_dialog(row['Book ID'], conn, section)
            # Completed-specific column
            elif "Completed" in title:
                if role == "cover_designer":
                    with col_configs[col_idx]:
                        front_end = row['Front Cover End']
                        value = front_end.strftime('%Y-%m-%d') if not pd.isna(front_end) and front_end != '0000-00-00 00:00:00' else "-"
                        st.markdown(f'<span>{value}</span>', unsafe_allow_html=True)
                    col_idx += 1
                    with col_configs[col_idx]:
                        back_end = row['Back Cover End']
                        value = back_end.strftime('%Y-%m-%d') if not pd.isna(back_end) and back_end != '0000-00-00 00:00:00' else "-"
                        st.markdown(f'<span>{value}</span>', unsafe_allow_html=True)
                    col_idx += 1
                else:
                    with col_configs[col_idx]:
                        end_date = row[f'{section.capitalize()} End']
                        value = end_date.strftime('%Y-%m-%d') if not pd.isna(end_date) and end_date != '0000-00-00 00:00:00' else "-"
                        st.markdown(f'<span>{value}</span>', unsafe_allow_html=True)
                    col_idx += 1

# --- Section Configuration ---
sections = {
    "writing": {"role": "writer", "color": "unused"},
    "proofreading": {"role": "proofreader", "color": "unused"},
    "formatting": {"role": "formatter", "color": "unused"},
    "cover": {"role": "cover_designer", "color": "unused"}
}


for section, config in sections.items():
    if user_role == config["role"] or user_role == "admin":
        st.session_state['section'] = section
        books_df = fetch_books(months_back=4 , section=section)
        
        if section == "writing":
            running_books = books_df[
                books_df['Writing Start'].notnull() & 
                books_df['Writing End'].isnull()
            ]
            pending_books = books_df[
                books_df['Writing Start'].isnull()
            ]
            completed_books = books_df[
                books_df['Writing End'].notnull()
            ]
        elif section == "proofreading":
            running_books = books_df[
                books_df['Proofreading Start'].notnull() & 
                books_df['Proofreading End'].isnull()
            ]
            pending_books = books_df[
                (books_df['Writing End'].notnull() | (books_df['Is Publish Only'] == 1)) & 
                books_df['Proofreading Start'].isnull()
            ]
            completed_books = books_df[
                books_df['Proofreading End'].notnull()
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
            completed_books = books_df[
                books_df['Formatting End'].notnull()
            ]
        elif section == "cover":
            running_books = books_df[
                (
                    (books_df['Front Cover Start'].notnull() | books_df['Back Cover Start'].notnull())
                    & 
                    (books_df['Front Cover End'].isnull() | books_df['Back Cover End'].isnull())
                )
            ]
            pending_books = books_df[
                books_df['Front Cover Start'].isnull() & 
                books_df['Back Cover Start'].isnull()
            ]
            completed_books = books_df[
                books_df['Front Cover End'].notnull() & 
                books_df['Back Cover End'].notnull()
            ]

        # Sort Pending table by Date (oldest first)
        pending_books = pending_books.sort_values(by='Date', ascending=True)

        # Column sizes (adjusted to match previous updates)
        if section == "writing":
            column_sizes_running = [0.7, 5.5, 1, 1, 1.2, 1.2, 1, 1]  
            column_sizes_pending = [0.7, 5.5, 1, 1, 0.8, 1]            
            column_sizes_completed = [0.7, 5.5, 1, 1, 1, 1]         
        elif section == "proofreading":
            column_sizes_running = [0.8, 5.5, 1, 1.2, 1, 1.2, 1.2, 1, 1] 
            column_sizes_pending = [0.8, 5.5, 1, 1.2, 1, 1, 1, 0.8, 0.8] 
            column_sizes_completed = [0.7, 5.5, 1, 1, 1.2, 1, 1,1]    
        elif section == "formatting":
            column_sizes_running = [0.7, 5.5, 1, 1, 1.2, 1.2, 1]      
            column_sizes_pending = [0.7, 5.5, 1, 1, 1.2, 1, 1]         
            column_sizes_completed = [0.7, 5.5, 1, 1, 1.2, 1, 1]       
        elif section == "cover":
            column_sizes_running = [0.8, 5, 1.2, 1.2, 0.7, 0.7, 0.75, 1.3, 1, 1, 1.2]  
            column_sizes_pending = [0.8, 5.5, 1, 1.2, 1, 1, 1, 0.8, 1]            
            column_sizes_completed = [0.7, 5.5, 1, 1.5, 1.3, 1.3]                   
        
        selected_month = render_month_selector(books_df)
        render_metrics(books_df, selected_month, section)
        render_table(running_books, f"{section.capitalize()} Running", column_sizes_running, config["color"], section, config["role"], is_running=True)
        render_table(pending_books, f"{section.capitalize()} Pending", column_sizes_pending, config["color"], section, config["role"], is_running=False)
        if st.button(f"Show {section.capitalize()} Completed Books", key=f"show_{section}_completed"):
            render_table(completed_books, f"{section.capitalize()} Completed", column_sizes_completed, config["color"], section, config["role"], is_running=False)






