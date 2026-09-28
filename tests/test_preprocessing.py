from avito_candidate_gen.preprocessing import (
    normalize_text,
    lemmatize_text,
    build_item_text,
    fix_keyboard_layout,
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


def test_fix_keyboard_layout_decodes_en_layout():
    # «ktrnhbrf ghjdjlrf» набрано в английской раскладке
    assert fix_keyboard_layout("ktrnhbrf ghjdjlrf") == "лектрика проводка"


def test_fix_keyboard_layout_keeps_cyrillic_and_digits():
    assert fix_keyboard_layout("маникюр") == "маникюр"
    assert fix_keyboard_layout("0445110298") == "0445110298"


def test_fix_keyboard_layout_keeps_real_english():
    # «iphone» декодируется в бессмыслицу — словарь такое не подтверждает
    assert fix_keyboard_layout("iphone") == "iphone"


def test_fix_keyboard_layout_requires_all_words_known():
    # «high heels» -> «ршпр рууды»: «рууды» есть в словаре редких слов,
    # но второе слово нет — запрос остаётся нетронутым
    assert fix_keyboard_layout("high heels") == "high heels"
    assert fix_keyboard_layout("bi led") == "bi led"
