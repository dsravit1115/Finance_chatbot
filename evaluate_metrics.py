from difflib import SequenceMatcher

def calculate_context_recall(sources, answer):
    match_score = max([SequenceMatcher(None, src, answer).ratio() for src in sources])
    return match_score

def calculate_exact_match(answer, query):
    return int(query.lower().strip('?') in answer.lower())
