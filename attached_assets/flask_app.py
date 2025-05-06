import os
import json
import traceback # For detailed error logging
import markdown # Import markdown library
import time
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
# Removed: secrets, wraps, ProxyFix (if not needed for other reasons beyond user sessions)
# Removed: database, user_management imports

# Import specific functions from flashcard_logic
try:
    from flashcard_logic import (
        process_new_content, 
        load_embedding_model,
        SIMILARITY_THRESHOLD  # Keep if still used for in-session similarity
    )
except ImportError as e:
     print(f"Error importing from flashcard_logic: {e}. Make sure flashcard_logic.py is in the correct path.")
     exit()

app = Flask(__name__)
# app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1) # Keep if relevant for your deployment beyond user sessions

# A simple secret key is still good for flashing messages, etc.
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))


# --- Helper Functions (Simplified) ---

def process_cards_for_template(card_list):
    """Processes a list of card dictionaries for template rendering.
       - Converts source_chunk and card content fields to HTML using Markdown.
       - Ensures similarity info (if any from in-session processing) is handled.
    """
    if not card_list:
        return []

    processed_list = []
    markdown_extensions = ['extra', 'sane_lists', 'fenced_code', 'codehilite']

    for card_row in card_list:
        card_dict = card_row.copy() # Assuming card_row is already a dict from flashcard_logic
        card_id = card_dict.get('id', 'N/A') # ID might be temporary/session-based now

        # Process similar_existing_cards (which would be from the current batch)
        # This part might be simplified or removed if in-batch similarity is not complex
        if 'similar_existing_cards' not in card_dict:
            card_dict['similar_existing_cards'] = [] # Default to empty list

        if 'highest_similarity_score' not in card_dict:
            if card_dict.get('similar_existing_cards'):
                try:
                    max_sim_card = max(card_dict['similar_existing_cards'],
                                      key=lambda x: x.get('similarity', 0))
                    card_dict['highest_similarity_score'] = max_sim_card.get('similarity')
                    card_dict['highest_similarity_id'] = max_sim_card.get('id') # This ID would be temporary
                except (ValueError, TypeError): # Handle empty list or items without 'similarity'
                    card_dict['highest_similarity_score'] = None
                    card_dict['highest_similarity_id'] = None
            else:
                card_dict['highest_similarity_score'] = None
                card_dict['highest_similarity_id'] = None
        
        # Convert source chunk to HTML
        if not card_dict.get('source_chunk_html'): # Check if already processed by flashcard_logic
            raw_chunk = card_dict.get('source_chunk', '')
            try:
                card_dict['source_chunk_html'] = markdown.markdown(raw_chunk or '', extensions=markdown_extensions)
            except Exception as md_err:
                print(f"Error converting source_chunk markdown for card {card_id}: {md_err}")
                card_dict['source_chunk_html'] = markdown.escape(raw_chunk)


        # Convert main card content fields to HTML
        fields_to_convert = ['front', 'back', 'full_text']
        for field in fields_to_convert:
            raw_content = card_dict.get(field, '')
            html_field_key = f"{field}_html" # e.g. front_html

            # Store raw and HTML versions if useful, or just replace
            # For simplicity, let's assume template expects HTML in 'front', 'back', 'full_text'
            try:
                # Only convert if it doesn't appear to be HTML already
                if raw_content and not ('<' in raw_content and '>' in raw_content and any(tag in raw_content for tag in ['<p>','<div>','<h1>','<h2>','<h3>','<h4>','<h5>','<h6>','<li>','<ul>','<ol>','<blockquote>','<pre>'])):
                    card_dict[field] = markdown.markdown(raw_content or '', extensions=markdown_extensions)
                # else it's already HTML or empty
            except Exception as md_err:
                print(f"Error converting {field} markdown for card {card_id}: {md_err}")
                card_dict[field] = markdown.escape(raw_content) # Fallback to escaped raw text

        processed_list.append(card_dict)
    return processed_list

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
        similarity_threshold=SIMILARITY_THRESHOLD # Pass if used in template
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
        return render_template('index.html', source_text=source_text,
                             similarity_threshold=SIMILARITY_THRESHOLD)

    try:
        print("Processing request...")
        start_time = time.time()
        
        # process_new_content no longer takes user_id and doesn't save to DB
        generated_cards_raw, error_message, parsing_warnings = process_new_content(source_text)
        
        end_time = time.time()
        print(f"Processing complete in {end_time - start_time:.2f} seconds. "
              f"Generated: {len(generated_cards_raw)} cards total for this session.")

        generated_cards_final = process_cards_for_template(generated_cards_raw)

        if error_message:
            flash(f"Error during processing: {error_message}", "danger")
        elif not generated_cards_final and not parsing_warnings:
             flash("No flashcards were generated from the provided text.", "warning")
        elif generated_cards_final:
            high_sim_count = sum(1 for c in generated_cards_final
                               if c.get('highest_similarity_score') is not None and
                               c['highest_similarity_score'] >= SIMILARITY_THRESHOLD)
            threshold_val = SIMILARITY_THRESHOLD
            flash(f"Generated {len(generated_cards_final)} cards for this session. "
                 f"Found {high_sim_count} with high similarity (within this batch) >= {threshold_val:.1f}.",
                 "success")

        if parsing_warnings:
            warnings_str = "; ".join(parsing_warnings)
            flash(f"Processing Warnings: {warnings_str}", "warning")

    except Exception as e:
        print(f"Unhandled exception in /generate route: {e}")
        traceback.print_exc()
        flash(f"An unexpected server error occurred during generation. Please try again.", "danger")
        error_message = "Unexpected server error."
        generated_cards_final = []

    return render_template(
        'index.html',
        source_text=source_text,
        generated_cards=generated_cards_final, # These are for the current request only
        error=error_message,
        warnings=parsing_warnings,
        similarity_threshold=SIMILARITY_THRESHOLD
    )

# Removed routes: /login, /logout, /admin, /api/cards, /update_status, /get_full_source, 
# /remake_card, /kept, /flagged, /history

if __name__ == '__main__':
    print("Attempting to preload embedding model...")
    try:
        load_embedding_model() # Keep if embeddings are used for in-session processing
        print("Embedding model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not preload embedding model: {e}")
        traceback.print_exc()

    port = int(os.environ.get('PORT', 8080))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

    print(f"Starting Flask server on {host}:{port} (debug={debug})...")
    app.run(host=host, port=port, debug=debug)

application = app