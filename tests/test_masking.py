from app import mask_pii

def test_masking():
    assert '[MASKED_EMAIL]' in mask_pii('Email: john@example.com')
