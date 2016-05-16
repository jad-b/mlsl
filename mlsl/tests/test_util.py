import textwrap

from mlsl.util import format_metadata


def test_format_metadata():
    test_meta = {
        'cost': 123.456,
        'iterations': 1000,
        'time': .203
    }
    string = format_metadata(test_meta) % test_meta
    expected = textwrap.dedent("""
    Metadata
    ========
    cost = 123.456
    iterations = 1000
    time = 0.203
    """)
    assert string == expected
