import os
from dotenv import load_dotenv
import google.generativeai as genai
import re
import time
import datetime
import uuid
import json
import nltk
import markdown

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
PROMPT_EXAMPLE_FILE = "prompt_example.md"
GEMINI_MODEL_NAME = "gemini-2.0-flash"

CHUNK_METHOD = 'paragraph'
TARGET_CHUNK_SIZE_WORDS = 50

print(f"Using Gemini Model: {GEMINI_MODEL_NAME}")
print(f"Chunking Method: {CHUNK_METHOD}" +
      (f", Size: {TARGET_CHUNK_SIZE_WORDS} words" if CHUNK_METHOD ==
       'word_count' else ""))

try:
    with open(PROMPT_EXAMPLE_FILE, 'r', encoding='utf-8') as f:
        prompt_example_content = f.read()
    print(f"Successfully loaded prompt example from {PROMPT_EXAMPLE_FILE}")
except Exception as e:
    print(f"Warning: Could not load {PROMPT_EXAMPLE_FILE}: {e}")
    prompt_example_content = ""

# --- Gemini Prompt Templates ---
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

def split_into_chunks(text, method='paragraph', target_size=300):
    chunks = []
    if not text or not text.strip(): return chunks
    if method == 'paragraph':
        chunks = [
            chunk.strip() for chunk in re.split(r'\n\s*\n', text)
            if chunk.strip()
        ]
        print(f"Split into {len(chunks)} paragraph chunks.")
    elif method == 'word_count':
        print(f"Attempting to split into chunks near {target_size} words...")
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception as e:
            print(f"NLTK failed: {e}. Falling back to paragraph.")
            return split_into_chunks(text, 'paragraph')
        current_chunk, current_word_count = [], 0
        for sentence in sentences:
            sentence_word_count = len(sentence.split())
            if current_word_count > 0 and current_word_count + sentence_word_count > target_size:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_word_count = [sentence
                                                     ], sentence_word_count
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_word_count
        if current_chunk: chunks.append(" ".join(current_chunk))
        print(f"Split into {len(chunks)} word count chunks.")
    else:
        print("Warning: Unknown chunk method.")
        chunks = [text]
    return [chunk for chunk in chunks if chunk and chunk.strip()]


def get_gemini_response_chunked(base_prompt, example_content, task_suffix,
                                full_text, current_chunk):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return None, "API Key not configured."
    full_prompt = (base_prompt + "\n" + example_content + "\n" +
                   task_suffix.format(full_original_text=full_text,
                                      current_chunk_to_process=current_chunk))
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        print(f"Sending request to Gemini model: {GEMINI_MODEL_NAME}...")
        start_time = time.time()
        response = model.generate_content(full_prompt)
        end_time = time.time()
        print(
            f"Gemini response received in {end_time - start_time:.2f} seconds."
        )
        raw_text_output = ""
        try:
            raw_text_output = response.text
            print(
                f"\n--- RAW GEMINI OUTPUT (Chunk) ---\n{raw_text_output}\n-------------------------------\n"
            )
        except Exception as e:
            print(f"Could not get text from Gemini response: {e}")
            block_reason = "Unknown (empty/no text)"
            try:
                if hasattr(response,
                           'prompt_feedback') and response.prompt_feedback:
                    block_reason = response.prompt_feedback.block_reason or block_reason
                error_message = f"Gemini response empty/blocked. Reason: {block_reason}."
            except:
                error_message = "Gemini response empty/blocked."
            print(f"Warning: {error_message}")
            return None, error_message
        return raw_text_output, None
    except Exception as e:
        error_message = f"Error calling Gemini API: {e}"
        print(error_message)
        feedback = getattr(e, 'prompt_feedback', None) or getattr(
            getattr(e, 'response', None), 'prompt_feedback', None)
        if feedback:
            error_message += f" | Feedback: {feedback}"
            print(f"Prompt Feedback: {feedback}")
        if "API key not valid" in str(e): error_message += " (Check .env file)"
        if "PermissionDenied" in str(e) or "403" in str(e):
            error_message += f" (Check access to model '{GEMINI_MODEL_NAME}')"
        if "model not found" in str(e).lower() or "404" in str(e):
            error_message += f" (Model '{GEMINI_MODEL_NAME}' not found?)"
        return None, error_message


def parse_gemini_output(text):
    candidates, parsing_warnings = [], []
    if not text: return candidates, parsing_warnings
    card_blocks = re.split(r'\n+---\n+', text.strip())
    for block_num, block in enumerate(card_blocks):
        block = block.strip()
        if not block: continue
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if not lines: continue

        lines[0] = re.sub(r"^[#\s]+", "", lines[0])  # Strip leading #
        if not lines[0] and len(lines) > 1:
            lines.pop(0)  # Handle case where stripping # leaves an empty line
        if not lines: continue
        elif not lines[0] and len(lines) == 1:
            continue  # Only an empty line left

        card = {
            'type': None,
            'front': '',
            'back': '',
            'full_text': '',
            'id': f"temp_{block_num}_{uuid.uuid4().hex[:6]}"
        }  # Temporary ID
        is_correct_cloze = re.search(r"\{\{c\d+::", block)
        is_malformed_cloze = re.search(r"\{c\d+::",
                                       block) and not is_correct_cloze

        if is_correct_cloze:
            card['type'] = 'cloze'
            card['full_text'] = "\n".join(lines)
            if not re.search(r"\}\}", block):
                parsing_warnings.append(
                    f"Cloze Issue (missing '}}'?): {block[:100]}...")
        elif is_malformed_cloze:
            card['type'] = 'cloze'
            card['full_text'] = "\n".join(lines)
            warning_msg = f"Cloze detected with INCORRECT single braces: {block[:100]}..."
            print(f"Warning: {warning_msg}")
            parsing_warnings.append(warning_msg)
        elif len(lines) >= 2:
            card['type'] = 'basic'
            card['front'] = lines[0]
            card['back'] = lines[1]
            if len(lines) > 2:
                card['back'] += " " + " ".join(
                    lines[2:])  # Concatenate remaining lines to back
        elif len(lines) == 1:  # Treat as basic if not cloze
            warning_msg = f"Single line (not Cloze) treated as Basic Front: '{lines[0]}'"
            print(f"Warning: {warning_msg}")
            parsing_warnings.append(warning_msg)
            card['type'] = 'basic'
            card['front'] = lines[0]
        else:  # Should not happen if lines is not empty
            warning_msg = f"Could not parse block: {block}"
            print(f"Warning: {warning_msg}")
            parsing_warnings.append(warning_msg)
            continue

        # Ensure content exists
        if card['type'] == 'basic' and not (card['front'] or card['back']):
            parsing_warnings.append(
                f"Parsed Basic empty: Block='{block[:100]}...'")
            continue
        if card['type'] == 'cloze' and not card['full_text']:
            parsing_warnings.append(
                f"Parsed Cloze empty: Block='{block[:100]}...'")
            continue

        if card['type']: candidates.append(card)

    if not candidates:
        warning_msg = "Could not parse any cards from Gemini output."
        print(f"Warning: {warning_msg}")
        parsing_warnings.append(warning_msg)
    return candidates, parsing_warnings


def process_new_content(full_input_text):
    """Process new content and generate flashcards."""
    global prompt_example_content
    error_message = None
    all_parsing_warnings = []
    all_generated_cards_this_session = []

    markdown_extensions = ['extra', 'sane_lists', 'fenced_code', 'codehilite']

    chunks = split_into_chunks(full_input_text,
                               method=CHUNK_METHOD,
                               target_size=TARGET_CHUNK_SIZE_WORDS)
    if not chunks:
        return [], "Input text resulted in empty chunks.", []

    print(f"Processing {len(chunks)} chunks...")

    chunk_error_occurred = False

    for i, chunk in enumerate(chunks):
        print(f"\n--- Processing Chunk {i+1}/{len(chunks)} ---")
        if not chunk.strip():
            print("Skipping empty chunk.")
            continue

        gemini_output, gemini_error = get_gemini_response_chunked(
            GEMINI_PROMPT_BASE, prompt_example_content,
            GEMINI_PROMPT_TASK_SUFFIX_CHUNKED, full_input_text, chunk)

        if gemini_error:
            error_message = f"API Error on chunk {i+1}: {gemini_error}"
            chunk_error_occurred = True
            break
        if not gemini_output:
            print(f"Warning: No output for chunk {i+1}.")
            continue

        chunk_candidate_cards, chunk_parsing_warnings = parse_gemini_output(
            gemini_output)
        all_parsing_warnings.extend(chunk_parsing_warnings)
        if not chunk_candidate_cards:
            print(f"Warning: Could not parse cards for chunk {i+1}.")
            continue

        processed_cards_for_this_chunk = []

        for base_card in chunk_candidate_cards:
            current_card = base_card.copy()
            current_card['source_chunk'] = chunk  # Keep raw source chunk
            current_card['source_chunk_html'] = markdown.markdown(
                chunk or '', extensions=markdown_extensions
            )  # Add HTML version of chunk

            processed_cards_for_this_chunk.append(current_card)

        all_generated_cards_this_session.extend(
            processed_cards_for_this_chunk)

        if chunk_error_occurred:
            break  # Stop if an error occurred in a chunk

    if chunk_error_occurred:
        print("Processing stopped due to API error in a chunk.")
    else:
        print("\nChunk processing finished for this request.")

    return all_generated_cards_this_session, error_message, all_parsing_warnings

# Example usage
if __name__ == "__main__":
    # Example text to process
    sample_text = """
    The Python programming language was created by Guido van Rossum. 
    It was first released in 1991 and is known for its simple, easy-to-learn syntax.

    Python is interpreted rather than compiled, which makes development faster but can 
    make execution slower compared to compiled languages like C or C++.

    The language emphasizes code readability and its syntax allows programmers to express 
    concepts in fewer lines of code than would be possible in languages such as C++ or Java.
    """

    # Process the content
    result_cards, error, warnings = process_new_content(sample_text)

    if error:
        print(f"Error occurred: {error}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")

    print(f"\nGenerated {len(result_cards)} cards:")
    for i, card in enumerate(result_cards):
        print(f"\n--- Card {i+1} ---")
        print(f"Type: {card['type']}")
        if card['type'] == 'basic':
            print(f"Front: {card['front']}")
            print(f"Back: {card['back']}")
        else:
            print(f"Full text: {card['full_text']}")
