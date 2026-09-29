from tools.archive_quant_research import normalize_text, redact_text


def test_redact_text_cleans_credentials_and_internal_locations():
    source = (
        'export QUANTX_MAIL_PASSWORD="secret-value"\n'
        "Authorization: Bearer abc.def.ghi\n"
        "url=http://10.36.14.23/path\n"
        "host=model.aidi.hobot.cc\n"
        "/home/users/mingxiao.li/git/quantization\n"
    )

    clean, counts = redact_text(source)

    assert "secret-value" not in clean
    assert "abc.def.ghi" not in clean
    assert "10.36.14.23" not in clean
    assert "model.aidi.hobot.cc" not in clean
    assert "/home/users/mingxiao.li" not in clean
    assert "<REDACTED:CREDENTIAL>" in clean
    assert "<REDACTED:BEARER_TOKEN>" in clean
    assert "<PRIVATE_IP>" in clean
    assert "<INTERNAL_HOST>" in clean
    assert "${HOME}/git/quantization" in clean
    assert sum(counts.values()) == 4


def test_normalize_text_uses_lf_spaces_and_no_trailing_whitespace():
    source = "a  \r\n \tline\t \r\n"

    assert normalize_text(source) == "a\n    line\n"
