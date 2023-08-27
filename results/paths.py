from dataclasses import dataclass


@dataclass
class ResultsPaths:
    raw: str
    error_files_raw: str
    processed: str
    evaluation: str
