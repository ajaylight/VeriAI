from app.services.claim_splitter import split_into_claims

def test_splits_three_sentences():
    result = split_into_claims("The sky is blue. Water is wet! Is this correct?")
    assert [item.text for item in result] == ["The sky is blue.", "Water is wet!", "Is this correct?"]

def test_empty_input_returns_no_claims():
    assert split_into_claims("   ") == []

def test_offsets_point_to_original_text():
    answer = "  First claim. Second claim."
    result = split_into_claims(answer)
    assert answer[result[0].start_char:result[0].end_char] == "First claim."
    assert answer[result[1].start_char:result[1].end_char] == "Second claim."
