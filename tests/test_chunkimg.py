from app import semantic_chunk

def test_chunking():
    assert isinstance(semantic_chunk('Sentence one. Sentence two.'), list)
