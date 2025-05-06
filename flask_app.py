import os
import json
import traceback
import markdown
import time
import re
import sqlite3
import zipfile
import tempfile
import uuid
from datetime import datetime
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, send_file

# Import specific functions from flashcard_logic
try:
    from flashcard_logic import (
        process_new_content,  # Keep for content processing
    )
except ImportError as e:
    print(
        f"Error importing from flashcard_logic: {e}. Make sure flashcard_logic.py is in the correct path."
    )
    exit()

# Anki converter functions are now included in this file!
def parse_flashcards_from_text(text):
    """Parse flashcards from the manually-edited text document."""
    flashcards = []

    # Remove the header
    text = re.sub(r'^## .*?\n+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^#### .*?\n+', '', text, flags=re.MULTILINE)

    # Split into sections by empty lines
    sections = text.split('\n\n')

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Check if it's a cloze deletion card
        if '{{c' in section:
            # It's a cloze card
            # Clean up any extra lines within the cloze
            cloze_text = re.sub(r'\n+', ' ', section)
            # Remove LaTeX formatting for Anki compatibility
            cloze_text = re.sub(r'\\large\\boxed\{(.*?)\}', r'\1', cloze_text)
            cloze_text = re.sub(r'\\text\{(.*?)\}', r'\1', cloze_text)
            cloze_text = re.sub(r'\\boxed\{(.*?)\}', r'\1', cloze_text)
            cloze_text = re.sub(r'\\frac\{(.*?)\}\{(.*?)\}', r'(\1)/(\2)', cloze_text)
            cloze_text = re.sub(r'\$\$(.*?)\$\$', r'\1', cloze_text)
            cloze_text = re.sub(r'\$(.*?)\$', r'\1', cloze_text)
            flashcards.append({
                'type': 'cloze',
                'content': cloze_text.strip()
            })
        else:
            # Check if it's a basic card (question/answer format)
            lines = section.split('\n')
            if len(lines) >= 2:
                # Remove trailing punctuation from question if present
                question = lines[0].strip().rstrip('?')
                # Combine remaining lines as answer
                answer = ' '.join([line.strip() for line in lines[1:] if line.strip()])

                # Remove any markdown formatting from answer
                answer = re.sub(r'\*\*(.*?)\*\*', r'\1', answer)  # Bold
                answer = re.sub(r'\*(.*?)\*', r'\1', answer)    # Italic
                # Remove LaTeX formatting
                answer = re.sub(r'\\large\\boxed\{(.*?)\}', r'\1', answer)
                answer = re.sub(r'\\text\{(.*?)\}', r'\1', answer)
                answer = re.sub(r'\\boxed\{(.*?)\}', r'\1', answer)
                answer = re.sub(r'\\frac\{(.*?)\}\{(.*?)\}', r'(\1)/(\2)', answer)
                answer = re.sub(r'\$\$(.*?)\$\$', r'\1', answer)
                answer = re.sub(r'\$(.*?)\$', r'\1', answer)

                flashcards.append({
                    'type': 'basic',
                    'question': question,
                    'answer': answer
                })

    return flashcards

def create_anki_package(flashcards, output_path, deck_name="Generated Flashcards"):
    """Create an Anki package (.apkg) file from the flashcards."""

    # Create a temporary directory for our files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create the collection database
        db_path = os.path.join(temp_dir, "collection.anki2")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute('''
            CREATE TABLE col (
                id              integer PRIMARY KEY,
                crt             integer NOT NULL,
                mod             integer NOT NULL,
                scm             integer NOT NULL,
                ver             integer NOT NULL,
                dty             integer NOT NULL,
                usn             integer NOT NULL,
                ls              integer NOT NULL,
                conf            text NOT NULL,
                models          text NOT NULL,
                decks           text NOT NULL,
                dconf           text NOT NULL,
                tags            text NOT NULL
            )
        ''')

        cursor.execute('''
            CREATE TABLE notes (
                id              integer PRIMARY KEY,
                guid            text NOT NULL,
                mid             integer NOT NULL,
                mod             integer NOT NULL,
                usn             integer NOT NULL,
                tags            text NOT NULL,
                flds            text NOT NULL,
                sfld            text NOT NULL,
                csum            integer NOT NULL,
                flags           integer NOT NULL,
                data            text NOT NULL
            )
        ''')

        cursor.execute('''
            CREATE TABLE cards (
                id              integer PRIMARY KEY,
                nid             integer NOT NULL,
                did             integer NOT NULL,
                ord             integer NOT NULL,
                mod             integer NOT NULL,
                usn             integer NOT NULL,
                type            integer NOT NULL,
                queue           integer NOT NULL,
                due             integer NOT NULL,
                ivl             integer NOT NULL,
                factor          integer NOT NULL,
                reps            integer NOT NULL,
                lapses          integer NOT NULL,
                left            integer NOT NULL,
                odue            integer NOT NULL,
                odid            integer NOT NULL,
                flags           integer NOT NULL,
                data            text NOT NULL
            )
        ''')

        cursor.execute('''
            CREATE TABLE revlog (
                id              integer PRIMARY KEY,
                cid             integer NOT NULL,
                usn             integer NOT NULL,
                ease            integer NOT NULL,
                ivl             integer NOT NULL,
                lastIvl         integer NOT NULL,
                factor          integer NOT NULL,
                time            integer NOT NULL,
                type            integer NOT NULL
            )
        ''')

        cursor.execute('''
            CREATE TABLE graves (
                usn             integer NOT NULL,
                oid             integer NOT NULL,
                type            integer NOT NULL
            )
        ''')

        # Create indexes
        cursor.execute("CREATE INDEX idx_notes_usn ON notes (usn)")
        cursor.execute("CREATE INDEX idx_cards_usn ON cards (usn)")
        cursor.execute("CREATE INDEX idx_revlog_usn ON revlog (usn)")
        cursor.execute("CREATE INDEX idx_cards_nid ON cards (nid)")
        cursor.execute("CREATE INDEX idx_cards_sched ON cards (did, queue, due)")
        cursor.execute("CREATE INDEX idx_revlog_cid ON revlog (cid)")
        cursor.execute("CREATE INDEX idx_notes_csum ON notes (csum)")

        # Current timestamp (proper milliseconds)
        crt = int(datetime.now().timestamp() * 1000)
        mod = crt
        scm = crt

        # Create deck
        deck_id = 1000000000 + int(datetime.now().timestamp())
        decks = {
            str(deck_id): {
                "id": deck_id,
                "mid": 0,
                "name": deck_name,
                "usn": -1,
                "lrnToday": [0, 0],
                "revToday": [0, 0],
                "newToday": [0, 0],
                "timeToday": [0, 0],
                "collapsed": False,
                "browserCollapsed": False,
                "desc": "",
                "dyn": 0,
                "conf": 1,
                "extendNew": 10,
                "extendRev": 50
            }
        }

        # Create models
        model_id = 1000000000 + int(datetime.now().timestamp())
        basic_model = {
            "id": model_id,
            "name": "Basic",
            "type": 0,
            "mod": mod,
            "usn": -1,
            "sortf": 0,
            "did": deck_id,
            "tmpls": [
                {
                    "name": "Card 1",
                    "ord": 0,
                    "qfmt": "{{Question}}",
                    "afmt": "{{FrontSide}}<hr id=answer>{{Answer}}",
                    "did": None,
                    "bqfmt": "",
                    "bafmt": ""
                }
            ],
            "flds": [
                {
                    "name": "Question",
                    "ord": 0,
                    "sticky": False,
                    "rtl": False,
                    "font": "Arial",
                    "size": 20,
                    "media": []
                },
                {
                    "name": "Answer",
                    "ord": 1,
                    "sticky": False,
                    "rtl": False,
                    "font": "Arial",
                    "size": 20,
                    "media": []
                }
            ],
            "css": ".card {\n font-family: arial;\n font-size: 20px;\n text-align: center;\n color: black;\n background-color: white;\n}\n",
            "latexPre": "\\documentclass[12pt]{article}\n\\special{papersize=3in,5in}\n\\usepackage[utf8]{inputenc}\n\\usepackage{amssymb,amsmath}\n\\pagestyle{empty}\n\\setlength{\\parindent}{0in}\n\\begin{document}\n",
            "latexPost": "\\end{document}",
            "req": [[0, "any", [0, 1]]]
        }

        cloze_model_id = model_id + 1
        cloze_model = {
            "id": cloze_model_id,
            "name": "Cloze",
            "type": 1,
            "mod": mod,
            "usn": -1,
            "sortf": 0,
            "did": deck_id,
            "tmpls": [
                {
                    "name": "Cloze",
                    "ord": 0,
                    "qfmt": "{{cloze:Text}}",
                    "afmt": "{{cloze:Text}}",
                    "did": None,
                    "bqfmt": "",
                    "bafmt": ""
                }
            ],
            "flds": [
                {
                    "name": "Text",
                    "ord": 0,
                    "sticky": False,
                    "rtl": False,
                    "font": "Arial",
                    "size": 20,
                    "media": []
                }
            ],
            "css": ".card {\n font-family: arial;\n font-size: 20px;\n text-align: center;\n color: black;\n background-color: white;\n}\n",
            "latexPre": "\\documentclass[12pt]{article}\n\\special{papersize=3in,5in}\n\\usepackage[utf8]{inputenc}\n\\usepackage{amssymb,amsmath}\n\\pagestyle{empty}\n\\setlength{\\parindent}{0in}\n\\begin{document}\n",
            "latexPost": "\\end{document}",
            "req": [[0, "all", [0]]]
        }

        models = {
            str(model_id): basic_model,
            str(cloze_model_id): cloze_model
        }

        # Configuration
        conf = {
            "curDeck": deck_id,
            "curModel": str(model_id),
            "nextPos": 1,
            "estTimes": True,
            "activeDecks": [deck_id],
            "sortType": "noteFld",
            "timeLim": 0,
            "sortBackwards": False,
            "addToCur": True,
            "newBury": True,
            "newSpread": 0,
            "dueCounts": True,
            "collapseTime": 1200
        }

        dconf = {
            "1": {
                "id": 1,
                "mod": 0,
                "name": "Default",
                "usn": 0,
                "maxTaken": 60,
                "autoplay": True,
                "timer": 0,
                "replayq": True,
                "new": {
                    "bury": True,
                    "delays": [1, 10],
                    "initialFactor": 2500,
                    "ints": [1, 4, 7],
                    "order": 1,
                    "perDay": 20,
                    "separate": True
                },
                "lapse": {
                    "delays": [10],
                    "leechAction": 0,
                    "leechFails": 8,
                    "minInt": 1,
                    "mult": 0
                },
                "rev": {
                    "bury": True,
                    "ease4": 1.3,
                    "fuzz": 0.05,
                    "ivlFct": 1,
                    "maxIvl": 36500,
                    "minSpace": 1,
                    "perDay": 200
                },
                "dyn": False
            }
        }

        # Ensure all JSON strings are properly formatted
        try:
            conf_json = json.dumps(conf, ensure_ascii=False)
            models_json = json.dumps(models, ensure_ascii=False)
            decks_json = json.dumps(decks, ensure_ascii=False)
            dconf_json = json.dumps(dconf, ensure_ascii=False)
        except Exception as e:
            print(f"JSON encoding error: {e}")
            return False

        # Insert collection data
        cursor.execute('''
            INSERT INTO col (id, crt, mod, scm, ver, dty, usn, ls, conf, models, decks, dconf, tags)
            VALUES (1, ?, ?, ?, 11, 0, -1, 0, ?, ?, ?, ?, ?)
        ''', (crt, mod, scm, conf_json, models_json, decks_json, dconf_json, "{}"))

        # Add flashcards with proper unique IDs
        card_id = 1000000000  # Start with a large number for uniqueness
        note_id = 1000000000

        for i, card in enumerate(flashcards):
            note_id += 1
            guid = str(uuid.uuid4().hex)

            if card['type'] == 'basic':
                # Basic card
                question = card['question']
                answer = card['answer']

                # Insert note
                flds = question + "\x1f" + answer
                sfld = question
                csum = hash(question) % (10**10)

                cursor.execute('''
                    INSERT INTO notes (id, guid, mid, mod, usn, tags, flds, sfld, csum, flags, data)
                    VALUES (?, ?, ?, ?, -1, "", ?, ?, ?, 0, "")
                ''', (note_id, guid, model_id, mod, flds, sfld, csum))

                # Insert card with incremented ID
                card_id += 1
                cursor.execute('''
                    INSERT INTO cards (id, nid, did, ord, mod, usn, type, queue, due, ivl, factor, reps, lapses, left, odue, odid, flags, data)
                    VALUES (?, ?, ?, 0, ?, -1, 0, 0, ?, 0, 0, 0, 0, 0, 0, 0, 0, "")
                ''', (card_id, note_id, deck_id, mod, card_id))

            else:  # cloze
                # Cloze card
                content = card['content']

                # Insert note
                flds = content
                sfld = content
                csum = hash(content) % (10**10)

                cursor.execute('''
                    INSERT INTO notes (id, guid, mid, mod, usn, tags, flds, sfld, csum, flags, data)
                    VALUES (?, ?, ?, ?, -1, "", ?, ?, ?, 0, "")
                ''', (note_id, guid, cloze_model_id, mod, flds, sfld, csum))

                # For cloze cards, create one card per cloze deletion
                cloze_numbers = re.findall(r'{{c(\d+)::', content)
                unique_cloze_numbers = sorted(set(int(n) for n in cloze_numbers))

                for j, cloze_num in enumerate(unique_cloze_numbers):
                    card_id += 1  # Always increment for each new card
                    cursor.execute('''
                        INSERT INTO cards (id, nid, did, ord, mod, usn, type, queue, due, ivl, factor, reps, lapses, left, odue, odid, flags, data)
                        VALUES (?, ?, ?, ?, ?, -1, 0, 0, ?, 0, 0, 0, 0, 0, 0, 0, 0, "")
                    ''', (card_id, note_id, deck_id, j, mod, card_id))

        conn.commit()
        conn.close()

        # Create media file (proper UTF-8 encoded JSON)
        media_path = os.path.join(temp_dir, "media")
        media_info = {}
        with open(media_path, "w", encoding='utf-8') as f:
            json.dump(media_info, f, ensure_ascii=False)

        # Create the apkg file
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(db_path, "collection.anki2")
            zipf.write(media_path, "media")

        print(f"Successfully created Anki package: {output_path}")
        return True

app = Flask(__name__)

# A simple secret key is still good for flashing messages, etc.
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))

# Configuration
SIMILARITY_THRESHOLD = 0.8  # Keep this constant, even if not used for actual similarity

# Directory for temporary Anki packages
TEMP_DIR = os.path.join(os.path.dirname(__file__), 'temp')
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# --- Helper Functions (Simplified) ---

def process_cards_for_template(card_list):
    """Processes a list of card dictionaries for template rendering."""
    if not card_list:
        return []

    processed_list = []
    markdown_extensions = ['extra', 'sane_lists', 'fenced_code', 'codehilite']

    for card_row in card_list:
        card_dict = card_row.copy()
        card_id = card_dict.get('id', 'N/A')

        # Remove similarity-related processing since embeddings are removed
        card_dict['similar_existing_cards'] = []
        card_dict['highest_similarity_score'] = None
        card_dict['highest_similarity_id'] = None

        # Convert source chunk to HTML
        if not card_dict.get('source_chunk_html'):
            raw_chunk = card_dict.get('source_chunk', '')
            try:
                card_dict['source_chunk_html'] = markdown.markdown(
                    raw_chunk or '', extensions=markdown_extensions)
            except Exception as md_err:
                print(f"Error converting source_chunk markdown for card {card_id}: {md_err}")
                card_dict['source_chunk_html'] = markdown.escape(raw_chunk)

        # Convert main card content fields to HTML
        fields_to_convert = ['front', 'back', 'full_text']
        for field in fields_to_convert:
            raw_content = card_dict.get(field, '')

            try:
                if raw_content and not (
                        '<' in raw_content and '>' in raw_content
                        and any(tag in raw_content for tag in [
                            '<p>', '<div>', '<h1>', '<h2>', '<h3>', '<h4>',
                            '<h5>', '<h6>', '<li>', '<ul>', '<ol>',
                            '<blockquote>', '<pre>'
                        ])):
                    card_dict[field] = markdown.markdown(
                        raw_content or '', extensions=markdown_extensions)
            except Exception as md_err:
                print(f"Error converting {field} markdown for card {card_id}: {md_err}")
                card_dict[field] = markdown.escape(raw_content)

        processed_list.append(card_dict)
    return processed_list

CLEANUP_AGE_MINUTES = 60  # Clean up files older than 1 hour

# Add this function
def cleanup_old_files():
    """Remove temp files older than CLEANUP_AGE_MINUTES minutes"""
    if not os.path.exists(TEMP_DIR):
        return

    current_time = datetime.now()
    cutoff_time = current_time - timedelta(minutes=CLEANUP_AGE_MINUTES)

    for filename in os.listdir(TEMP_DIR):
        if filename.endswith('.apkg'):
            file_path = os.path.join(TEMP_DIR, filename)
            file_modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))

            if file_modified_time < cutoff_time:
                try:
                    os.remove(file_path)
                    print(f"Cleaned up old file: {filename}")
                except Exception as e:
                    print(f"Error cleaning up {filename}: {e}")


# --- Routes ---

@app.route('/')
def root():
    """Root URL redirects to index"""
    return redirect(url_for('index'))

@app.route('/index', methods=['GET'])
def index():
    """Renders the main page for generating new flashcards."""
    return render_template(
        'index.html',
        generated_cards=None,
        similarity_threshold=SIMILARITY_THRESHOLD
    )

@app.route('/generate', methods=['POST'])
def generate():
    """Handles the flashcard generation request."""
    source_text = request.form.get('source_text', '')
    generated_cards_raw = []
    error_message = None
    parsing_warnings = []

    if not source_text.strip():
        flash("Please enter some text.", "warning")
        return render_template('index.html',
                               source_text=source_text,
                               similarity_threshold=SIMILARITY_THRESHOLD)

    try:
        print("Processing request...")
        start_time = time.time()

        generated_cards_raw, error_message, parsing_warnings = process_new_content(
            source_text)

        end_time = time.time()
        print(
            f"Processing complete in {end_time - start_time:.2f} seconds. "
            f"Generated: {len(generated_cards_raw)} cards total for this session."
        )

        generated_cards_final = process_cards_for_template(generated_cards_raw)

        if error_message:
            flash(f"Error during processing: {error_message}", "danger")
        elif not generated_cards_final and not parsing_warnings:
            flash("No flashcards were generated from the provided text.",
                  "warning")
        elif generated_cards_final:
            flash(
                f"Generated {len(generated_cards_final)} cards for this session.",
                "success")

        if parsing_warnings:
            warnings_str = "; ".join(parsing_warnings)
            flash(f"Processing Warnings: {warnings_str}", "warning")

    except Exception as e:
        print(f"Unhandled exception in /generate route: {e}")
        traceback.print_exc()
        flash(
            f"An unexpected server error occurred during generation. Please try again.",
            "danger")
        error_message = "Unexpected server error."
        generated_cards_final = []

    return render_template(
        'index.html',
        source_text=source_text,
        generated_cards=generated_cards_final,
        error=error_message,
        warnings=parsing_warnings,
        similarity_threshold=SIMILARITY_THRESHOLD)

@app.route('/convert', methods=['GET', 'POST'])
def convert_to_anki():
    """Convert uploaded flashcards text file to Anki package."""
    print(f"DEBUG: convert route called, method={request.method}")

    if request.method == 'GET':
        return render_template('convert.html')

    # Get content from either file upload or text input
    content = None
    deck_name = request.form.get('deck_name', 'Generated Flashcards')
    cleanup_old_files()

    # Check for text content first
    text_content = request.form.get('text_content', '').strip()
    if text_content:
        content = text_content
        print("DEBUG: Using text input")
    else:
        # Check for file upload
        if 'file' not in request.files or request.files['file'].filename == '':
            print("DEBUG: No file or text provided")
            flash('Please provide either a file or text content', 'warning')
            return render_template('convert.html')

        file = request.files['file']
        try:
            content = file.read().decode('utf-8')
            print("DEBUG: Using file upload")
        except Exception as e:
            print(f"DEBUG: Error reading file: {e}")
            flash(f'Error reading file: {str(e)}', 'danger')
            return render_template('convert.html')

    if not content:
        flash('No content provided', 'warning')
        return render_template('convert.html')

    try:
        print(f"DEBUG: Content length: {len(content)}")

        # Parse flashcards
        flashcards = parse_flashcards_from_text(content)
        print(f"DEBUG: Parsed {len(flashcards)} flashcards")

        if not flashcards:
            flash('No flashcards found in the provided content.', 'warning')
            return render_template('convert.html', deck_name=deck_name)

        # Create output filename with timestamp to avoid conflicts
        timestamp = int(time.time())
        safe_filename = "".join(c for c in deck_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        output_filename = f"{safe_filename}_{timestamp}.apkg"
        output_path = os.path.join(TEMP_DIR, output_filename)

        print(f"DEBUG: Creating Anki package at: {output_path}")

        # Create Anki package
        success = create_anki_package(flashcards, output_path, deck_name)

        print(f"DEBUG: Package creation success: {success}")
        print(f"DEBUG: File exists: {os.path.exists(output_path)}")

        if success and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"DEBUG: Package size: {file_size} bytes")
            return render_template('convert.html', 
                                   success_file=output_filename,
                                   deck_name=deck_name)
        else:
            print("DEBUG: Failed to create package")
            flash('Error creating Anki package.', 'danger')
            return render_template('convert.html', deck_name=deck_name)

    except Exception as e:
        print(f"DEBUG: Error in /convert route: {e}")
        traceback.print_exc()
        flash(f'Error processing content: {str(e)}', 'danger')
        return render_template('convert.html', deck_name=deck_name)

@app.route('/download/<filename>')
def download_anki_package(filename):
    """Download generated Anki package."""
    try:
        file_path = os.path.join(TEMP_DIR, filename)
        print(f"DEBUG: Attempting to download file: {file_path}")

        if os.path.exists(file_path):
            print(f"DEBUG: File exists, sending it")
            return send_file(file_path, as_attachment=True)
        else:
            print(f"DEBUG: File not found")
            flash('File not found.', 'danger')
            return redirect(url_for('convert_to_anki'))
    except Exception as e:
        print(f"DEBUG: Error downloading file: {e}")
        traceback.print_exc()
        flash(f'Error downloading file: {str(e)}', 'danger')
        return redirect(url_for('convert_to_anki'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

    print(f"Starting Flask server on {host}:{port} (debug={debug})...")
    app.run(host=host, port=port, debug=debug)

application = app
