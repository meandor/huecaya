from unittest.mock import call
from requests.exceptions import HTTPError
from de_identification_library.client import send_deidentification_request
import pytest


def test_send_request(mocker):
    post_mock = mocker.patch("requests.post")
    post_mock.return_value.json.return_value = {"answer": "foobar"}

    actual = send_deidentification_request("endpoint", "foo")
    expected = ["foobar"]

    assert expected == actual


def test_send_another_request(mocker):
    post_mock = mocker.patch("requests.post")
    post_mock.return_value.json.return_value = {"answer": "foobar, Foo Bar, barfoo"}

    actual = send_deidentification_request("http//another.endpoint", "foo bar foobar")
    expected = ["foobar", "Foo Bar", "barfoo"]

    assert expected == actual


def test_with_retry_success(mocker):
    post_mock = mocker.patch("requests.post")

    def side_effect():
        side_effect.call_count = getattr(side_effect, "call_count", 0) + 1
        if side_effect.call_count <= 4:
            raise HTTPError()
        else:
            return None

    post_mock.return_value.raise_for_status.side_effect = side_effect
    post_mock.return_value.json.return_value = {"answer": "foobar, Foo Bar, barfoo"}

    actual = send_deidentification_request("http//another.endpoint", "foo bar foobar")
    expected = ["foobar", "Foo Bar", "barfoo"]

    assert expected == actual


def test_with_retry_fail(mocker):
    post_mock = mocker.patch("requests.post")
    post_mock.return_value.raise_for_status.side_effect = HTTPError()

    with pytest.raises(RuntimeError):
        send_deidentification_request("http//another.endpoint", "foo bar foobar")
        post_mock.assert_has_calls([call()])
