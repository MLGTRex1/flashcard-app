import os
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd # Keep if used for other data manipulation, or remove if not
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ast
# import csv # Removed as flashcards.csv is no longer a primary source
import re
import time
# import sqlite3 # Removed
import datetime # Keep if timestamps are used for non-DB purposes
import uuid # Keep for session_id or temporary card IDs
import json
import nltk
import markdown # For adding source_chunk_html directly

# --- NLTK Check ---
try:
    nltk.data.find('tokenizers/punkt')
    print("NLTK 'punkt' tokenizer found.")
except LookupError:
    print("NLTK 'punkt' tokenizer not found. Attempting download...")
    try:
        nltk.download('punkt')
        print("'punkt' downloaded successfully.")
        nltk.data.find('tokenizers/punkt')
        print("'punkt' verified.")
    except Exception as download_e:
        print(f"Error downloading 'punkt': {download_e}")
except Exception as e:
    print(f"Error checking NLTK data: {e}")

load_dotenv()

# --- Configuration ---
# EXISTING_FLASHCARDS_CSV = "flashcards.csv" # Removed
# DATABASE_FILE = 'flashcards_log.db' # Removed
PROMPT_EXAMPLE_FILE = "prompt_example.md"
EMBEDDING_MODEL_NAME = 'multi-qa-MiniLM-L6-cos-v1'
SIMILARITY_THRESHOLD = 0.8 
SIMILAR_CARDS_TO_SHOW = 15 # For in-batch similarity
GEMINI_MODEL_NAME = "gemini-2.0-flash"

CHUNK_METHOD = 'paragraph'
TARGET_CHUNK_SIZE_WORDS = 50

embedding_model = None # Singleton embedding model
print(f"Using Gemini Model: {GEMINI_MODEL_NAME}")
print(f"Chunking Method: {CHUNK_METHOD}" + (f", Size: {TARGET_CHUNK_SIZE_WORDS} words" if CHUNK_METHOD == 'word_count' else ""))

try:
    with open(PROMPT_EXAMPLE_FILE, 'r', encoding='utf-8') as f:
        prompt_example_content = f.read()
    print(f"Successfully loaded prompt example from {PROMPT_EXAMPLE_FILE}")
except Exception as e:
    print(f"Warning: Could not load {PROMPT_EXAMPLE_FILE}: {e}")
    prompt_example_content = ""

# --- Gemini Prompt Templates (GEMINI_PROMPT_BASE, GEMINI_PROMPT_TASK_SUFFIX_CHUNKED, GEMINI_PROMPT_REMAKE_CARD) ---
# These remain the same as they define the core LLM interaction.
# GEMINI_PROMPT_REMAKE_CARD might be less relevant if "remaking" refers to editing a persisted card.
# For now, keeping them. If card editing is implemented in-session, it might be adapted.

GEMINI_PROMPT_BASE = """
To ensure the generation of Anki flashcards that exhibit consistency, contextual independence, and adherence to established mnemonic principles, the subsequent guidelines must be strictly observed:

**General Principles**
* **Self-Contained Cards:** Each card, whether Basic or Cloze, **must** be fully self-contained with ALL necessary context included directly in the card itself. All questions must be specific and comprehensible without requiring information from other cards.
* **Subject Reference:** The specific subject, theory, concept, or topic name MUST be explicitly included in EVERY question or cloze statement to provide immediate context.
* **Verify Self-Containment:** Before finalizing any card, verify that someone without prior knowledge could understand exactly what is being asked and in what context.
* **Source Adherence:** One hundred percent of the provided material must be incorporated; no omissions are permissible. **Strictly** avoid paraphrasing the source text. The exact wording, as it appears in the original text, must be employed whenever possible.
* **Australian English:** Australian English spelling conventions (e.g., 'analyse', 'colour', 'prioritise') are to be adhered to.
* **Simplicity:** Complex information is to be simplified by partitioning it into multiple, discrete cards.
* **Brevity and Clarity:** Wording should be optimized for brevity and clarity, free from unnecessary complexity, introductory phrases, or preambles in both questions and answers.
* **Avoid Interference:** Wording must be precise to prevent confusion with similar facts.
* **Redundancy:** Avoid testing the exact same piece of information multiple times *within a single card* unless essential for comprehension. However, difficult concepts may be reinforced through multiple cards presenting different perspectives or facets.
* **Value Check:** Ensure each card provides substantive value and tests meaningful content rather than vague assertions or overly general statements.
* **Statistical and Numerical Accuracy:** ALL statistics, numerical data, percentages, dates, and quantitative information MUST be included VERBATIM from the source material with 100% accuracy. No rounding, approximating, or simplifying of numerical values is permitted.
* **Quote Preservation:** All quotes from the source material MUST be preserved exactly as written, maintaining the precise wording, punctuation, and attribution. Quotes must never be paraphrased or abbreviated.
* **Complete Data Inclusion:** Every single piece of data, statistic, measurement, figure, and quantitative finding in the source content MUST be incorporated into flashcards. No numerical information can be omitted.

**Card Types**

**1. Basic Cards (Front/Back):**
* **Structure:** Must consist of exactly two lines: a Question line followed immediately by an Answer line.
* **Complete Context:** Questions MUST include the specific topic, theory or concept name as well as sufficient context to be fully understood in isolation.
* **Specificity:** Questions must be precise and unambiguous. Avoid generic questions like "What is the importance of X?" unless the answer is substantive and specific.
* **Focus:** Each card assesses precisely one specific fact or concept.
* **Answer (Back):** Must be **extremely succinct**, containing *only* the specific fact or concept being tested. Avoid repeating context from the question. It should typically be a single word, phrase, number, or very short sentence fragment.
* **Labels:** The labels "Q:" and "A:" are **not** to be used.
* **Compound Answers:** Avoid compound answers; split into multiple cards if necessary.
* **Statistical Integrity:** When a question tests for statistical data, the answer MUST contain the EXACT figure(s) as presented in the source material, including all decimal places, units, and contextual qualifiers.
* **Example:**
    According to Phillips curve theory, what economic problem was an economy believed to face in the 1960s?
    Either an inflation problem or an unemployment problem.

**2. Cloze Deletion Cards:**
* **Structure:** Must consist of a single line containing the full sentence with one or more cloze deletions.
* **Complete Context:** Each cloze sentence MUST include the specific topic or concept name and sufficient context to be fully understood in isolation.
* **CRITICAL FORMAT:** Use the **exact** `{{c<number>::text}}` format (double curly braces REQUIRED, where `<number>` is the cloze index like 1, 2, 3...). Use unique c-numbers starting from c1 for each *new* card. Single braces like `{c1::text}` are **incorrect and must not be used**.
* **Nesting:** Group related terms under the same c-number using nesting. Example: `{{c3::{{c1::option A}} / {{c2::option B}}}}`.
* **Deletion Scope:** Avoid deleting an excessive number of words at once. Focus on key terms/data.
* **Labels:** The labels "Q:" and "A:" are **not** to be used.
* **Context:** Ensure the sentence provides enough context to be understood independently.
* **Precise Data Clozing:** When creating cloze deletions for statistical data, the ENTIRE statistical value must be included in the cloze, including all decimal places, units, and contextual qualifiers.

**Content Rules**
* Use 100% of the provided material from the 'Current Chunk to Process'.
* Verify COMPLETE coverage by systematically checking off each fact, concept, and data point as it is incorporated into flashcards.
* Ensure key terminology and important concepts receive priority coverage.
* Use exact wording from the source whenever possible.
* Use cloze deletions for lists, not basic cards with long enumerations.
* Integrate context (headings) directly into each card for clarity.
* Partition tables into individual cards per row (usually Basic Q/A).
* All content must be covered by BOTH basic cards and cloze cards (i.e. all content MUST be covered twice)
* EVERY statistic, piece of numerical data, percentage, measurement, date, and quantitative finding MUST be included in multiple cards with 100% accuracy to the source.
* ALL direct quotes from the source material MUST be preserved exactly as written in dedicated flashcards.

**Formatting Rules**
* Use plain text exclusively (no markdown formatting like **, ##, ###, etc. in the output cards).
* No hierarchical bullet points in cloze deletions.
* **CRITICAL SEPARATOR:** You **MUST** place `---` (three hyphens) on its own line between **EVERY** generated flashcard (both Basic and Cloze).

**Additional Guidelines**
* Each output card MUST be BOTH a Basic card (Question line followed by Answer line) AND a Cloze card (a single line containing `{{c...}}` deletions). Do not output standalone statements. Do not mix Basic question format with Cloze answer format.
* Optimize wording for brevity and clarity.
* Avoid interference; use precise wording. Omit unnecessary preambles.
* Before finalizing, review each card for:
  1. Self-containment (includes all needed context)
  2. Specificity (asks a precise question with a concrete answer)
  3. Value (provides meaningful knowledge)
  4. Accuracy (correctly represents the source material)
  5. Statistical precision (maintains exact values from source)
  6. Quote accuracy (preserves exact wording of quotes)

---
**Example of Input and Expected Output:**
"""

GEMINI_PROMPT_TASK_SUFFIX_CHUNKED = """
---
**Task:**

Use the 'Full Original Text' below for essential context (like understanding headings, concepts, and the overall topic).
Generate flashcards that cover 100% of the information **exclusively** from the 'Current Chunk to Process' section below, following all the rules defined above.

**Critical Requirements:**
1. Every question or cloze statement MUST explicitly name the overall topic/theory being discussed
2. Every card MUST be completely self-contained and understandable without other cards
3. Verify that each card tests substantive knowledge (avoid vague questions with generic answers)
4. Ensure 100% coverage of all facts, concepts and data points in the chunk
5. Output each flashcard clearly separated by '---' (three hyphens) on a line by itself
6. Ensure Cloze cards use the mandatory `{{c<number>::text}}` format with double curly braces
7. DO NOT include any markdown headings (like ## or ###) in the flashcard output itself
8. EVERY statistic, numerical value, percentage, measurement, and quantitative data point MUST be included with 100% precision as it appears in the source text
9. ALL quotes must be preserved EXACTLY as written in the source material, with no alteration to wording or punctuation
10. **OUTPUT ONLY THE FLASHCARDS** - no explanations, comments, verifications, or any other text

**Full Original Text (Context Essential):**
{full_original_text}

**Current Chunk to Process:**
{current_chunk_to_process}

**IMPORTANT: Your output must contain ONLY the flashcards themselves. Do not include any explanations, verifications, or commentary in your output. Only include the cards separated by '---' (three hyphens).**

**Internal Verification (Do NOT output this):**
Before submitting, internally check that:
1. All key facts and concepts from the chunk are covered
2. Each card explicitly names the overall topic/theory being discussed
3. Each card is completely self-contained with all necessary context
4. No card contains vague questions with generic answers
5. Every single statistic, measurement, percentage, date, and numerical value is preserved with 100% accuracy
6. All direct quotes are maintained verbatim, with exact wording and punctuation
"""

GEMINI_PROMPT_REMAKE_CARD = """**Full Original Text (Context Only):**
{full_original_text}

**Original Source Chunk:**
{source_chunk}

**Original Flashcard to Remake:**
{original_card_content}

**YOUR OUTPUT MUST CONTAIN ONLY THE REMADE FLASHCARD AND NOTHING ELSE.**
You are an expert Anki flashcard editor with precise instructions to revise the provided flashcard.

**Essential Context:**
Your task is to remake an existing flashcard while preserving its *exact* same knowledge point, but with minimal improvements to its format, clarity, or contextual independence. The source chunk contains material for MULTIPLE flashcards, but you're only responsible for improving THIS ONE specific flashcard without expanding its scope.

**CRITICAL UNDERSTANDING:**
- The source chunk likely contains information for several different flashcards
- You must identify ONLY the specific knowledge point the original flashcard was testing
- Your rewrite should focus EXCLUSIVELY on that specific knowledge point
- DO NOT expand the flashcard to cover additional material from the source chunk

**Systematic Improvement Process (Follow This Decision Tree):**
1. **FIRST: Identify Format Issues**
   - Is the card using the correct format (Basic = 2 lines, Cloze = 1 line with {{{{c1::text}}}})?
   - Are there formatting errors (missing double curly braces, improper cloze syntax, etc.)?
   - If format issues exist → FIX THESE FIRST before moving to other improvements

2. **SECOND: Assess Question/Prompt Quality**
   - Is the question/prompt awkwardly phrased, unclear, or unnatural sounding?
   - Does it use unnecessarily complex language or indirect phrasing?
   - Would a student immediately understand what's being asked?
   - If question quality issues exist → REPHRASE FOR CLARITY AND NATURALNESS

3. **THIRD: Optimize Cloze Deletions (For Cloze Cards Only)**
   - Are the right concepts being tested? (Look for important terms/concepts that should be deleted but aren't)
   - Are there key terms left undeleted that would be more valuable to test?
   - Is there a better balance of what's being tested vs. what's providing context?
   - Would testing additional or different concepts in the sentence improve learning value?
   - If cloze selection issues exist → ADJUST WHAT IS BEING TESTED APPROPRIATELY

4. **FOURTH: Check Content Completeness**
   - Is the specific subject/topic explicitly identified in the card?
   - Is all essential context present to make the card fully self-contained?
   - Are there any missing terms or information critical to understanding?
   - If completeness issues exist → ADD MINIMAL NECESSARY CONTEXT

5. **FIFTH: Verify Accuracy**
   - Does the card contain any factual errors compared to the source chunk?
   - Are all statistics, data points, and numerical values accurate?
   - If accuracy issues exist → CORRECT THE INFORMATION

**Question Improvement Guidelines:**
* **Natural Language:** Questions should sound like natural inquiries, not artificial constructs
* **Direct Phrasing:** Use straightforward language that clearly indicates what is being asked
* **Precision:** Questions should have unambiguous answers that directly match what's being tested
* **Avoid Awkwardness:** Eliminate stilted or unnatural phrasings that could confuse learners
* **Contextual Smoothness:** Integrate topic context naturally into the question flow rather than appending it mechanically

**Cloze Deletion Optimization Guidelines:**
* **Strategic Selection:** Identify and delete key terms, concepts, and data points that represent important knowledge
* **Comprehensive Testing:** Look for opportunities to test ALL significant concepts in the sentence
* **Balanced Testing:** Ensure a good mix of what's being tested and what's providing context
* **Meaningful Deletions:** Focus on terms that represent substantive knowledge worth testing
* **Content Evaluation:** Assess what's currently being tested against what COULD be tested
* **Examples of Optimization:**
  - **Before:** If economic growth is less than {{{{c1::3%}}}}, it would increase {{{{c2::unemployment}}}} since demand for labour is a derived demand.
  - **After:** If {{{{c3::economic growth}}}} is less than {{{{c1::3%}}}}, it would increase {{{{c2::unemployment}}}} since {{{{c4::demand for labour}}}} is a {{{{c5::derived demand}}}}.

  - **Before:** Today, the sustainable rate or potential economic growth (EG) rate is considered to be lower at {{{{c1::2.5-2.5%}}}}.
  - **After:** Today, the {{{{c2::sustainable economic growth rate}}}} is considered to be lower at {{{{c1::2.5-2.5%}}}}.
* **Key Questions When Optimizing:**
  - Are all testable concepts in this sentence being tested?
  - What important terms are currently not being tested that should be?
  - Does the current selection of deletions adequately test understanding of the full concept?

**Context Integration Techniques:**
* **Organic Incorporation:** Weave context naturally into the question rather than tagging it on
* **Introductory Framing:** Begin with "In [subject/topic]," or "According to [theory]," to establish context
* **Implied Subject:** When possible, make the subject/topic the grammatical subject of the question
* **Seamless Flow:** Ensure added context doesn't disrupt the natural flow of the question
* **Minimal Addition:** Add only essential context needed for self-containment, not excessive details

**Final Quality Verification (Internal Check):**
Before submitting your remake, verify that:
1. The card format is 100% correct (Basic or Cloze with proper syntax)
2. The question/prompt reads naturally and is immediately clear to a student
3. Essential context is integrated smoothly, not awkwardly appended
4. The specific knowledge point matches exactly what the original card intended to test
5. No unnecessary information has been added that would expand the card's scope
6. The language flows naturally and sounds like something a teacher would ask
7. For Cloze cards, ALL key concepts in the sentence are being tested with appropriate deletions

**Format Requirements (Must Follow Exactly):**
* **Basic Cards:**
  - Exactly TWO lines total: Question line followed by Answer line
  - No "Q:"/"A:" labels
  - Question must contain sufficient context
  - Answer must be succinct - single word, phrase, number, or short fragment

* **Cloze Cards:**
  - Exactly ONE line total with one or more cloze deletions
  - Use EXACT format: `{{{{c<number>::text}}}}` (surrounded by double curley brackets)(TWO curley brackets on its left and TWO curley brackets on its right)
  - Start numbering from c1
  - Ensure sentence contains sufficient context

**Critical Rules:**
* **Preserve Original Scope:** The flashcard MUST test EXACTLY the same specific knowledge point as the original - do NOT expand its scope
* **Maintain Boundaries:** Do NOT incorporate additional facts/concepts from the source chunk that would be better as separate flashcards
* **Focus on Original Intent:** Identify what specific fact/concept the original card was testing and ONLY improve that specific card
* **Minimal Context Addition:** If adding context, add ONLY what's necessary for that specific card (usually just topic identifiers)
* **Source Wording:** Use exact wording from Source Chunk where possible for the specific fact being tested
* **Self-Containment:** Make the card understandable without external context through minimal additions
* **No Style Changes:** Keep the card in its original format (Basic→Basic, Cloze→Cloze)
* **Comprehensive Cloze Testing:** For Cloze cards, ensure ALL key concepts and terms in the sentence are being tested through appropriate deletions
* **No Commentary:** Provide ONLY the remade flashcard in your output
"""

# --- Database Functions (Removed or Simplified) ---

def load_embedding_model():
    global embedding_model
    if embedding_model is None:
        print(f"Loading embedding model ({EMBEDDING_MODEL_NAME})...")
        try:
            embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print("Embedding model loaded successfully.")
        except Exception as e:
            print(f"Fatal Error loading embedding model: {e}")
            raise
    return embedding_model

def get_card_text_for_embedding(card_row):
    card_type = card_row.get('type', '').lower()
    if card_type == 'basic':
        return f"{card_row.get('front', '')} [SEP] {card_row.get('back', '')}".strip()
    elif card_type == 'cloze':
        text = re.sub(r"\{\{?c\d+::(.*?)\}\}?", r"\1", card_row.get('full_text', ''))
        return text.strip()
    else:
        return f"{card_row.get('front', '')} {card_row.get('full_text', '')} {card_row.get('back', '')}".strip()

def generate_embeddings(texts, model=None):
    if not texts: return np.array([])
    if model is None: model = load_embedding_model()
    try:
        print(f"Generating embeddings for {len(texts)} texts...")
        start_time = time.time()
        embeddings = model.encode(texts, show_progress_bar=False)
        end_time = time.time()
        print(f"Embedding generation finished in {end_time - start_time:.2f} seconds.")
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return np.array([])

def serialize_embedding(embedding): # Kept if embeddings are stored in session/client-side
    if embedding is None: return None
    try: return json.dumps(embedding.tolist())
    except Exception as e:
        print(f"Error serializing embedding: {e}")
        return None

def deserialize_embedding(embedding_json): # Kept if embeddings are stored in session/client-side
    if not embedding_json: return None
    try: return np.array(json.loads(embedding_json))
    except Exception as e:
        print(f"Error deserializing embedding: {e}")
        return None

def find_similar_cards(new_embedding, existing_embeddings, existing_indices=None, top_n=15):
    if not isinstance(existing_embeddings, np.ndarray) or existing_embeddings.size == 0 or not isinstance(new_embedding, np.ndarray):
        return []
    if new_embedding.ndim == 1: new_embedding = new_embedding.reshape(1, -1)
    if existing_embeddings.ndim == 1:
        if existing_embeddings.size > 0: existing_embeddings = existing_embeddings.reshape(1, -1)
        else: return []
    try:
        similarities = cosine_similarity(new_embedding, existing_embeddings)[0]
        sorted_indices = np.argsort(similarities)[::-1]
        top_n_indices = sorted_indices[:top_n]
        results = []
        for index in [idx for idx in top_n_indices if idx < len(similarities)]:
            if similarities[index] > -1: 
                card_id = existing_indices[index] if existing_indices is not None and index < len(existing_indices) else index
                results.append((card_id, similarities[index]))
        results.sort(key=lambda item: item[1], reverse=True)
        return results
    except Exception as e:
        print(f"Error calculating cosine similarity: {e}")
        import traceback
        traceback.print_exc()
        return []

# Removed: load_all_embeddings (as it relied on DB/CSV)
# Removed: save_generated_cards_db, save_processing_history_db, update_card_status_db
# Removed: get_card_by_id_db, update_card_content_db, get_cards_by_status_db
# Removed: get_all_generated_cards_db, get_full_source_text_db

# process_card_similarities is no longer needed here as similarity processing will be
# more localized within process_new_content if done for in-batch cards.

def split_into_chunks(text, method='paragraph', target_size=300):
    chunks = []
    if not text or not text.strip(): return chunks
    if method == 'paragraph':
        chunks = [chunk.strip() for chunk in re.split(r'\n\s*\n', text) if chunk.strip()]
        print(f"Split into {len(chunks)} paragraph chunks.")
    elif method == 'word_count':
        print(f"Attempting to split into chunks near {target_size} words...")
        try: sentences = nltk.sent_tokenize(text)
        except Exception as e:
            print(f"NLTK failed: {e}. Falling back to paragraph.")
            return split_into_chunks(text, 'paragraph')
        current_chunk, current_word_count = [], 0
        for sentence in sentences:
            sentence_word_count = len(sentence.split())
            if current_word_count > 0 and current_word_count + sentence_word_count > target_size:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_word_count = [sentence], sentence_word_count
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_word_count
        if current_chunk: chunks.append(" ".join(current_chunk))
        print(f"Split into {len(chunks)} word count chunks.")
    else:
        print("Warning: Unknown chunk method.")
        chunks = [text]
    return [chunk for chunk in chunks if chunk and chunk.strip()]

def get_gemini_response_chunked(base_prompt, example_content, task_suffix, full_text, current_chunk):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return None, "API Key not configured."
    full_prompt = (base_prompt + "\n" + example_content + "\n" +
                  task_suffix.format(full_original_text=full_text, current_chunk_to_process=current_chunk))
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        print(f"Sending request to Gemini model: {GEMINI_MODEL_NAME}...")
        start_time = time.time()
        response = model.generate_content(full_prompt) # Removed safety_settings for simplicity here, add back if needed
        end_time = time.time()
        print(f"Gemini response received in {end_time - start_time:.2f} seconds.")
        raw_text_output = ""
        try:
            raw_text_output = response.text
            print(f"\n--- RAW GEMINI OUTPUT (Chunk) ---\n{raw_text_output}\n-------------------------------\n")
        except Exception as e: # More specific error handling for response.text
            print(f"Could not get text from Gemini response: {e}")
            block_reason = "Unknown (empty/no text)"
            try: # Try to get block reason
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    block_reason = response.prompt_feedback.block_reason or block_reason
                error_message = f"Gemini response empty/blocked. Reason: {block_reason}."
            except: error_message = "Gemini response empty/blocked."
            print(f"Warning: {error_message}")
            return None, error_message
        return raw_text_output, None
    except Exception as e:
        error_message = f"Error calling Gemini API: {e}"
        print(error_message)
        # Add more specific error details if available
        feedback = getattr(e, 'prompt_feedback', None) or getattr(getattr(e, 'response', None), 'prompt_feedback', None)
        if feedback: error_message += f" | Feedback: {feedback}"; print(f"Prompt Feedback: {feedback}")
        if "API key not valid" in str(e): error_message += " (Check .env file)"
        if "PermissionDenied" in str(e) or "403" in str(e): error_message += f" (Check access to model '{GEMINI_MODEL_NAME}')"
        if "model not found" in str(e).lower() or "404" in str(e): error_message += f" (Model '{GEMINI_MODEL_NAME}' not found?)"
        return None, error_message

# get_gemini_response_remake is removed as remake_flashcard_api is removed.
# If in-session card editing is desired, a new approach would be needed.

def parse_gemini_output(text):
    candidates, parsing_warnings = [], []
    if not text: return candidates, parsing_warnings
    card_blocks = re.split(r'\n+---\n+', text.strip())
    for block_num, block in enumerate(card_blocks):
        block = block.strip()
        if not block: continue
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if not lines: continue
        
        lines[0] = re.sub(r"^[#\s]+", "", lines[0]) # Strip leading #
        if not lines[0] and len(lines) > 1: lines.pop(0) # Handle case where stripping # leaves an empty line
        if not lines: continue
        elif not lines[0] and len(lines) == 1: continue # Only an empty line left

        card = {'type': None, 'front': '', 'back': '', 'full_text': '', 'id': f"temp_{block_num}_{uuid.uuid4().hex[:6]}"} # Temporary ID
        is_correct_cloze = re.search(r"\{\{c\d+::", block)
        is_malformed_cloze = re.search(r"\{c\d+::", block) and not is_correct_cloze

        if is_correct_cloze:
            card['type'] = 'cloze'
            card['full_text'] = "\n".join(lines)
            if not re.search(r"\}\}", block): parsing_warnings.append(f"Cloze Issue (missing '}}'?): {block[:100]}...")
        elif is_malformed_cloze:
            card['type'] = 'cloze'
            card['full_text'] = "\n".join(lines)
            warning_msg = f"Cloze detected with INCORRECT single braces: {block[:100]}..."
            print(f"Warning: {warning_msg}"); parsing_warnings.append(warning_msg)
        elif len(lines) >= 2:
            card['type'] = 'basic'
            card['front'] = lines[0]
            card['back'] = lines[1]
            if len(lines) > 2: card['back'] += " " + " ".join(lines[2:]) # Concatenate remaining lines to back
        elif len(lines) == 1: # Treat as basic if not cloze
            warning_msg = f"Single line (not Cloze) treated as Basic Front: '{lines[0]}'"
            print(f"Warning: {warning_msg}"); parsing_warnings.append(warning_msg)
            card['type'] = 'basic'
            card['front'] = lines[0]
        else: # Should not happen if lines is not empty
            warning_msg = f"Could not parse block: {block}"
            print(f"Warning: {warning_msg}"); parsing_warnings.append(warning_msg)
            continue

        # Ensure content exists
        if card['type'] == 'basic' and not (card['front'] or card['back']):
            parsing_warnings.append(f"Parsed Basic empty: Block='{block[:100]}...'")
            continue
        if card['type'] == 'cloze' and not card['full_text']:
            parsing_warnings.append(f"Parsed Cloze empty: Block='{block[:100]}...'")
            continue
            
        if card['type']: candidates.append(card)
            
    if not candidates:
        warning_msg = "Could not parse any cards from Gemini output."
        print(f"Warning: {warning_msg}"); parsing_warnings.append(warning_msg)
    return candidates, parsing_warnings

def process_new_content(full_input_text): # user_id removed
    global embedding_model, prompt_example_content # Ensure these are accessible
    error_message = None
    all_parsing_warnings = []
    all_generated_cards_this_session = []
    # session_id = str(uuid.uuid4()) # Can be kept for logging if needed, but not saved to DB

    markdown_extensions = ['extra', 'sane_lists', 'fenced_code', 'codehilite']


    try:
        embedding_model = load_embedding_model() # Ensure model is loaded

        # Removed: save_processing_history_db
        # Removed: load_all_embeddings (no persistent dataset to compare against)
        # Similarity will now be only against cards generated IN THIS BATCH.

        chunks = split_into_chunks(full_input_text, method=CHUNK_METHOD, target_size=TARGET_CHUNK_SIZE_WORDS)
        if not chunks:
            return [], "Input text resulted in empty chunks.", []

        print(f"Processing {len(chunks)} chunks...")
        
        # Store embeddings and card info for this batch only for intra-batch similarity
        current_batch_embeddings_list = []
        current_batch_card_info = [] # Will store dicts of the cards themselves
        chunk_error_occurred = False

        for i, chunk in enumerate(chunks):
            print(f"\n--- Processing Chunk {i+1}/{len(chunks)} ---")
            if not chunk.strip():
                print("Skipping empty chunk."); continue

            gemini_output, gemini_error = get_gemini_response_chunked(
                GEMINI_PROMPT_BASE, prompt_example_content, GEMINI_PROMPT_TASK_SUFFIX_CHUNKED,
                full_input_text, chunk
            )

            if gemini_error:
                error_message = f"API Error on chunk {i+1}: {gemini_error}"
                chunk_error_occurred = True; break 
            if not gemini_output:
                print(f"Warning: No output for chunk {i+1}."); continue

            chunk_candidate_cards, chunk_parsing_warnings = parse_gemini_output(gemini_output)
            all_parsing_warnings.extend(chunk_parsing_warnings)
            if not chunk_candidate_cards:
                print(f"Warning: Could not parse cards for chunk {i+1}."); continue

            # Prepare for in-batch similarity
            chunk_card_texts_for_embedding = [get_card_text_for_embedding(card) for card in chunk_candidate_cards]
            
            # Generate embeddings for this chunk's cards
            chunk_card_embeddings = np.array([])
            if any(chunk_card_texts_for_embedding): # Check if there's anything to embed
                chunk_card_embeddings = generate_embeddings(
                    [text for text in chunk_card_texts_for_embedding if text and text.strip()], 
                    embedding_model
                )

            processed_cards_for_this_chunk = []
            embedding_idx_counter = 0 # To map back from filtered embeddings to original chunk_candidate_cards

            for card_idx_in_chunk, base_card in enumerate(chunk_candidate_cards):
                current_card = base_card.copy()
                current_card['similar_existing_cards'] = [] # For similarity within this batch
                current_card['has_high_similarity_duplicate'] = False
                current_card['source_chunk'] = chunk # Keep raw source chunk
                current_card['source_chunk_html'] = markdown.markdown(chunk or '', extensions=markdown_extensions) # Add HTML version of chunk

                card_text_for_embedding = chunk_card_texts_for_embedding[card_idx_in_chunk]

                if card_text_for_embedding and card_text_for_embedding.strip() and chunk_card_embeddings.size > 0 and embedding_idx_counter < chunk_card_embeddings.shape[0]:
                    candidate_embedding = chunk_card_embeddings[embedding_idx_counter]
                    embedding_idx_counter += 1
                    
                    # Compare against cards ALREADY processed AND EMBEDDED in this batch
                    if current_batch_embeddings_list: 
                        existing_batch_embeddings_np = np.vstack(current_batch_embeddings_list)
                        existing_batch_card_ids = [c['id'] for c in current_batch_card_info] # Use temporary IDs

                        similar_results = find_similar_cards(
                            candidate_embedding,
                            existing_batch_embeddings_np,
                            existing_batch_card_ids, # Pass the temporary IDs
                            top_n=SIMILAR_CARDS_TO_SHOW 
                        )

                        if similar_results:
                            similar_cards_list_for_storage = []
                            for similar_card_id, score in similar_results:
                                # Find the actual card data from current_batch_card_info
                                similar_card_data = next((c for c in current_batch_card_info if c['id'] == similar_card_id), None)
                                if similar_card_data:
                                    # Create a simplified dict for storage
                                    sim_card_display = {
                                        'id': similar_card_data.get('id'), # temp ID
                                        'type': similar_card_data.get('type'),
                                        'front': similar_card_data.get('front', '')[:100] + "...", # Snippet
                                        'full_text': similar_card_data.get('full_text', '')[:100] + "...", # Snippet
                                        'similarity': score
                                    }
                                    similar_cards_list_for_storage.append(sim_card_display)
                                    if score >= SIMILARITY_THRESHOLD:
                                        current_card['has_high_similarity_duplicate'] = True
                            current_card['similar_existing_cards'] = similar_cards_list_for_storage
                    
                    # Add this card's embedding and info to the batch list for subsequent comparisons
                    current_batch_embeddings_list.append(candidate_embedding)
                    current_batch_card_info.append(current_card.copy()) # Store a copy of the card info
                
                processed_cards_for_this_chunk.append(current_card)
            
            all_generated_cards_this_session.extend(processed_cards_for_this_chunk)
            if chunk_error_occurred: break # Stop if an error occurred in a chunk

        if chunk_error_occurred: print("Processing stopped due to API error in a chunk.")
        else: print("\nChunk processing finished for this request.")
        
        # No DB saving, cards are returned directly
        return all_generated_cards_this_session, error_message, all_parsing_warnings

    except Exception as e:
        print(f"Error in process_new_content: {e}")
        import traceback
        traceback.print_exc()
        return [], f"Internal error: {e}", all_parsing_warnings

# Removed: remake_flashcard_api (as it relied on DB)