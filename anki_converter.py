import os
import re
import json
import sqlite3
import zipfile
import tempfile
import uuid
from datetime import datetime

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

        # Add flashcards
        card_id_start = 1000000000
        note_id_start = 1000000000

        for i, card in enumerate(flashcards):
            note_id = note_id_start + i
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

                # Insert card
                card_id = card_id_start + i
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
                    card_id = card_id_start + i + j
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

# Test the fix
if __name__ == "__main__":
    test_content = '''What is inflation?
An increase in the general price level of goods and services.

{{c1::Inflation}} is a sustained increase in the {{c2::general price level}} of goods and services.'''

    flashcards = parse_flashcards_from_text(test_content)
    print(f"Parsed {len(flashcards)} flashcards")

    if flashcards:
        create_anki_package(flashcards, "test_package.apkg", "Test Deck")
