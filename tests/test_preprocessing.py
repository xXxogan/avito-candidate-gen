from avito_candidate_gen.preprocessing import (
    normalize_text,
    lemmatize_text,
    build_item_text,
)


def test_normalize_text_lowercases_and_strips_punctuation():
    assert normalize_text("Баня, на Дровах!") == "баня на дровах"


def test_lemmatize_text_reduces_word_forms():
    assert lemmatize_text("бани") == lemmatize_text("баня")


def test_build_item_text_repeats_title():
    result = build_item_text(
        "Баня", "Строим бани", "Вид услуги Строительство", title_repeats=2
    )
    assert result.count("баня") >= 2
