from typing import Protocol

from stridebook.domain import ParsedRun


class RunSource(Protocol):
    """Turns the bytes of an uploaded file into a ParsedRun.

    The toy source fakes this; a Stridelab source will be `stridelab.read(data)` plus its
    records_df(), laps_df() and summary().
    """

    def parse(self, data: bytes, name: str) -> ParsedRun: ...
