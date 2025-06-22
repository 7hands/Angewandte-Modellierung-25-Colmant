import os
import re
import pandas as pd
import numpy as np
import spacy
from spacy.matcher import PhraseMatcher
from rapidfuzz import process, fuzz

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 1800000

# Helper: load books
def load_books(folder: str, ext: str = '.txt'):
    return [entry.path for entry in os.scandir(folder) if entry.name.endswith(ext)]

# Alias map for character name variants
alias_map = {
    'Ronald': 'Ron',
    'Ronald Weasley': 'Ron',
    # add more aliases here
}

# Load and clean character lists from multiple CSVs
def build_character_lists(csv_paths):
    names = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df['character'] = df['character'].str.replace(r"\(.*?\)", "", regex=True).str.strip()
        df['firstname'] = df['character'].str.split(pat=' ', n=1).str[0]
        names.extend(df['character'].tolist())
        names.extend(df['firstname'].tolist())
    # Include alias values
    names.extend(alias_map.values())
    # dedupe & sort by length
    all_names = sorted(set(names), key=lambda x: -len(x))
    stop_titles = {'Mr.', 'Mrs.', 'Miss', 'Dr.', 'Sir', 'Madam'}
    return [n for n in all_names if n and n not in stop_titles]

# Add matchers: Exact, Ruler & Fuzzy
def add_matchers(nlp, names, fuzzy_threshold: int = 85):
    # Exact matching before NER
    matcher_exact = PhraseMatcher(nlp.vocab, attr='LOWER')
    matcher_exact.add("CHAR_EXACT", [nlp.make_doc(n) for n in names])
    @spacy.language.Language.component("exact_matcher")
    def exact_matcher(doc):
        spans = [spacy.tokens.Span(doc, start, end, label="CHAR") for _, start, end in matcher_exact(doc)]
        doc.ents = spacy.util.filter_spans(list(doc.ents) + spans)
        return doc
    if "exact_matcher" not in nlp.pipe_names:
        nlp.add_pipe("exact_matcher", before="ner")

    # EntityRuler for multi-token patterns
    ruler = nlp.add_pipe("entity_ruler", after="exact_matcher", config={"overwrite_ents": False})
    patterns = [{"label": "CHAR", "pattern": n} for n in names]
    ruler.add_patterns(patterns)

    # Fuzzy matching after NER
    matcher_fuzzy = PhraseMatcher(nlp.vocab, attr='LOWER')
    matcher_fuzzy.add("CHAR_FUZZY", [nlp.make_doc(n) for n in names])
    @spacy.language.Language.component("fuzzy_matcher")
    def fuzzy_matcher(doc):
        spans = []
        for _, start, end in matcher_fuzzy(doc):
            span = doc[start:end]
            best = process.extractOne(span.text, names, scorer=fuzz.token_sort_ratio)
            if best and best[1] >= fuzzy_threshold:
                canonical = alias_map.get(best[0], best[0])
                spans.append(spacy.tokens.Span(doc, span.start, span.end, label="CHAR"))
        doc.ents = spacy.util.filter_spans(list(doc.ents) + spans)
        return doc
    if "fuzzy_matcher" not in nlp.pipe_names:
        nlp.add_pipe("fuzzy_matcher", after="ner")

# Extract entities over 3-sentence segments
def extract_segments_entities(doc, seg_size: int = 3):
    rows = []
    sents = list(doc.sents)
    for i in range(len(sents)):
        seg = sents[i:i+seg_size]
        if not seg: break
        start_char = seg[0].start_char
        end_char = seg[-1].end_char
        ents = [ent.text for ent in doc.ents if ent.label_ == 'CHAR' and ent.start_char >= start_char and ent.end_char <= end_char]
        # normalize aliases
        ents = [alias_map.get(e, e) for e in ents]
        rows.append({'start_sent': i,
                     'sentences': " ".join([s.text for s in seg]),
                     'entities': list(set(ents))})
    return pd.DataFrame(rows)

# Build relationships across segments
def build_relationships(df, window: int = 5):
    rels = []
    for i in df.index:
        idxs = [j for j in range(i, i + window + 1) if j in df.index]
        group = sum(df.loc[idxs, 'entities'].tolist(), [])
        unique = []
        for name in group:
            if not unique or unique[-1] != name:
                unique.append(name)
        rels += [(s, t) for a, b in zip(unique, unique[1:]) for s, t in [(alias_map.get(a, a), alias_map.get(b, b))]]
    rel_df = pd.DataFrame(rels, columns=['source','target'])
    rel_df['value'] = 1
    return rel_df.groupby(['source','target'], as_index=False).sum()

if __name__ == '__main__':
    data_folder = 'data'
    char_csvs = ['extra_chars.csv', 'data_chars_sort.csv']

    names = build_character_lists(char_csvs)
    add_matchers(nlp, names)

    for path in load_books(data_folder):
        text = open(path, encoding='utf8').read()
        doc = nlp(text)

        seg_df = extract_segments_entities(doc, seg_size=3)
        seg_df = seg_df[seg_df['entities'].str.len() > 0]
        seg_df.to_csv(path.replace('.txt', '_seg_ents.csv'), index=False)

        rel_df = build_relationships(seg_df, window=3)
        rel_df.to_csv(path.replace('.txt', '_relationships.csv'), index=False)

        print(f"Processed {os.path.basename(path)}: {len(seg_df)} segments, {len(rel_df)} relations")
