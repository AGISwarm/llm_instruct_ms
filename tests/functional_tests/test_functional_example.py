"""Example test file for pytest."""

# pylint: disable=import-error
import pytest  # type: ignore


def test_example():
    """Example test function."""
    if False:  # pylint: disable=using-constant-test
        pytest.fail("This test should pass.")
