import time
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import openai
import spacy
import os
from PyPDF2 import PdfReader
from docx import Document
from difflib import SequenceMatcher

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['SECRET_KEY'] = 'your_secret_key'  # Change this to a secret key for your application
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['USERS_FILE'] = 'users.txt'  # File to store username and password credentials
app.config['REGISTERED_FILE'] = 'registered.txt'  # File to store registered user details

# Create the uploads folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Check if the users file exists, and create it if not
if not os.path.exists(app.config['USERS_FILE']):
    with open(app.config['USERS_FILE'], 'w'):
        pass

# Check if the registered file exists, and create it if not
if not os.path.exists(app.config['REGISTERED_FILE']):
    with open(app.config['REGISTERED_FILE'], 'w'):
        pass

# Set your OpenAI API key
openai.api_key = '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'

# Load the spaCy English model
nlp = spacy.load("en_core_web_md")

# Placeholder for storing the document content
document_content = ""

# Define different engines for user queries and document context
user_query_engine = "gpt-3.5-turbo-0613"
document_context_engine = "gpt-3.5-turbo-0613"

def load_user_credentials():
    users = {}
    with open(app.config['USERS_FILE'], 'r') as file:
        lines = file.readlines()
        for line in lines:
            username, password = line.strip().split(':')
            users[username] = password
    return users

def check_user_credentials(username, password, users):
    return username in users and users[username] == password

def add_user_credentials(username, password):
    with open(app.config['USERS_FILE'], 'a') as file:
        file.write(f"{username}:{password}\n")

def add_registered_user_details(username, name, email, phone):
    with open(app.config['REGISTERED_FILE'], 'a') as file:
        file.write(f"Username: {username}\nName: {name}\nEmail: {email}\nPhone: {phone}\n\n")

# Initialize the users file with default credentials
if os.path.getsize(app.config['USERS_FILE']) == 0:
    add_user_credentials('Tilak', 'SK')
    add_user_credentials('user2', 'pass2')
    add_user_credentials('user3', 'pass3')

def get_exact_match_answer(query):
    # Find an exact match for the query in the document content
    # You may want to customize this based on your document structure
    if query in document_content:
        # Extract the answer from the document based on the query
        # This logic might need improvement based on the structure of your document
        start_index = document_content.find(query)
        end_index = start_index + len(query) + 100  # Adjust the range as needed
        exact_match_answer = document_content[start_index:end_index].strip()
        return exact_match_answer
    else:
        return None

def extract_relevant_info(document_content, query):
    # Check if either document content or query is empty
    if not document_content or not query:
        return None

    # Use SequenceMatcher to find the best matching substring
    matcher = SequenceMatcher(None, document_content.lower(), query.lower())
    match = matcher.find_longest_match(0, len(document_content), 0, len(query))

    # Extract the matching substring
    start_index = match.a
    end_index = match.a + match.size
    matched_substring = document_content[start_index:end_index].strip()

    return matched_substring

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    # Load user credentials
    users = load_user_credentials()

    # Check credentials
    if check_user_credentials(username, password, users):
        flash('Login successful', 'success')
        return redirect(url_for('upload_page'))
    else:
        flash('Login failed. Check your username and password.', 'danger')
        return redirect(url_for('index'))

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register_user():
    username = request.form['Username']
    password = request.form['password']
    name = request.form['Name']
    email = request.form['email']
    phone = request.form['phone']

    # Load user credentials
    users = load_user_credentials()

    # Check if the username already exists
    if username in users:
        flash('Username already exists. Choose a different username.', 'danger')
        return redirect(url_for('register'))

    # Add user credentials
    add_user_credentials(username, password)
    add_registered_user_details(username, name, email, phone)
    flash('Registration successful. You can now log in.', 'success')

    # Redirect to the login page after successful registration
    return redirect(url_for('index'))

@app.route('/upload-page')
def upload_page():
    return render_template('upload.html')

@app.route('/upload-document', methods=['POST'])
def upload_document():
    global document_content

    try:
        file = request.files['file']

        # Generate a secure filename and save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Read the content based on file type
        if filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as txt_file:
                document_content = txt_file.read()
        elif filename.endswith('.pdf'):
            # Use PyPDF2 to extract text from PDF
            pdf_reader = PdfReader(file_path)
            document_content = ""
            for page in pdf_reader.pages:
                document_content += page.extract_text()
        elif filename.endswith('.docx'):
            # Use python-docx to extract text from DOCX
            doc = Document(file_path)
            document_content = ""
            for paragraph in doc.paragraphs:
                document_content += paragraph.text

        print("Document Content:", document_content)  # Print to the IDE console

        # Introduce a delay of 1 second (import time if not already imported)
        time.sleep(1)

        # Render a template with the content on the web page
        return render_template('uploaded_content.html', filename=filename, content=document_content)
    except Exception as e:
        app.logger.error(f"Error in /upload-document: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/editfile/<filename>')
def editfile(filename):
    # Read the content from the file (replace this with your actual content retrieval logic)
    content = ""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
    except FileNotFoundError:
        # Handle file not found error as needed
        pass
    except UnicodeDecodeError as e:
        # Handle Unicode decode error
        app.logger.error(f"Error decoding file {filename}: {str(e)}")
        flash(f"Error decoding file {filename}. Please check the file format.", 'danger')
        # Redirect to an error page or handle it as appropriate for your application
        return redirect(url_for('index'))

    return render_template('editfile.html', title=filename, content=content)

@app.route('/uploadd')
def uploadd_page():
    return render_template('uploadd.html')

@app.route('/ask-chatgpt', methods=['POST'])
def ask_chatgpt():
    global document_content

    try:
        query = request.form['query']

        # Log the received query
        app.logger.info(f"Received query: {query}")

        # If no relevant information, use ChatGPT
        # Combine user query with document context
        prompt = f"User: {query}\nDoc: {document_content}"
        engine = document_context_engine

        # Use ChatGPT API for user query and document context
        response = openai.ChatCompletion.create(
            model=engine,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )

        chatgpt_response = response['choices'][0]['message']['content'].strip()

        # Log the ChatGPT response
        app.logger.info(f"ChatGPT response: {chatgpt_response}")

        return jsonify({'response': chatgpt_response})

    except Exception as e:
        app.logger.error(f"Error in /ask-chatgpt: {str(e)}")
        return jsonify({'error': str(e)})


@app.route('/ask')
def ask():
    return render_template('asking.html')

def extract_relevant_info(document_content, query):
    # Check if either document content or query is empty
    if not document_content or not query:
        return None

    # Tokenize the document and user query
    doc_tokens = nlp(document_content)
    query_tokens = nlp(query)

    # Calculate similarity based on token overlap
    similarity_scores = [token.similarity(query_tokens) for token in doc_tokens]

    # Find the index of the token with the highest similarity
    most_similar_index = similarity_scores.index(max(similarity_scores))

    # Extract the sentence containing the most similar token
    sentences = list(doc_tokens.sents)
    most_similar_sentence = sentences[most_similar_index]

    return most_similar_sentence.text

def is_related_to_document(query):
    # Check if any words in the query are present in the document content
    query_words = set(query.lower().split())
    document_words = set(document_content.lower().split())

    return bool(query_words.intersection(document_words))

def extract_relevant_info(document_content, query):
    # Tokenize the document and user query
    doc_tokens = nlp(document_content.lower())
    query_tokens = nlp(query.lower())

    # Calculate similarity based on token overlap
    similarity_scores = [token.similarity(query_tokens) for token in doc_tokens]

    # Get the index of the most similar token
    most_similar_index = similarity_scores.index(max(similarity_scores))

    # Get the corresponding sentence
    most_similar_sentence = list(doc_tokens.sents)[most_similar_index]

    return most_similar_sentence.text

@app.route('/search-in-document', methods=['POST'])
def search_in_document():
    global document_content

    try:
        # Use ChatGPT API for document context
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",  # Change to the appropriate model if needed
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Search in document: {document_content}"},
            ],
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )

        chatgpt_response = response['choices'][0]['message']['content'].strip()

        return jsonify({'response': chatgpt_response})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
