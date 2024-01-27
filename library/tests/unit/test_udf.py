from de_identification_library.udf import de_identify_fn
import pytest


def test_return_de_identify_one_word(mocker):
    send_deidentification_request_mock = mocker.patch(
        "de_identification_library.client.send_deidentification_request"
    )
    send_deidentification_request_mock.return_value = ["foobar"]

    actual = de_identify_fn("endpoint", "foobar text")
    expected = "xxx text"

    assert expected == actual
    send_deidentification_request_mock.assert_called_with("endpoint", "foobar text")


def test_return_de_identify_n_words(mocker):
    send_deidentification_request_mock = mocker.patch(
        "de_identification_library.client.send_deidentification_request"
    )
    send_deidentification_request_mock.return_value = ["foobar", "Bar Foo", "barfoo"]

    actual = de_identify_fn(
        "http://another-endpoint.com",
        "foobar text and another bar foo together with barfoo.",
    )
    expected = "xxx text and another xxx together with xxx."

    assert expected == actual
    send_deidentification_request_mock.assert_called_with(
        "http://another-endpoint.com",
        "foobar text and another bar foo together with barfoo.",
    )


def test_raise_error(mocker):
    send_deidentification_request_mock = mocker.patch(
        "de_identification_library.client.send_deidentification_request"
    )
    send_deidentification_request_mock.side_effect = RuntimeError("error")

    with pytest.raises(RuntimeError):
        de_identify_fn("broken_endpoint", "regardless")
