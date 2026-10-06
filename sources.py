"""Input sources for the pipeline.

The original project read Hadoop Sequence Files from HDFS on a university
cluster. That corpus is not publicly redistributable, so this repository also
supports a local directory of plain-text messages, which is what the included
generator produces and what the tests use. Both paths produce the same thing:
an RDD of raw message strings.
"""

from __future__ import annotations

from pathlib import Path


def utf8_decode_and_filter(rdd):
    """Decode sequence-file values to UTF-8, dropping records that fail.

    Provided by the original course harness and left unchanged. Real corpora
    contain mis-encoded messages; they are dropped rather than killing the task.
    """

    def utf_decode(s):
        try:
            return str(s, "utf-8")
        except Exception:
            return None

    return rdd.map(lambda x: utf_decode(x[1])).filter(lambda x: x is not None)


def read_sequence_files(sc, path: str):
    """Read raw messages from a Hadoop Sequence File path (HDFS or local)."""
    return utf8_decode_and_filter(sc.sequenceFile(path))


def read_text_directory(sc, path: str, min_partitions: int = 8):
    """Read raw messages from a directory of one-message-per-file text files.

    ``wholeTextFiles`` keeps each message intact, which matters because messages
    are multi-line records and ``textFile`` would split them by line.
    """
    directory = Path(path)
    if not directory.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run `python -m src.generate_data` first."
        )
    return sc.wholeTextFiles(str(directory), minPartitions=min_partitions).map(
        lambda kv: kv[1]
    )


def load(sc, path: str, fmt: str = "auto", min_partitions: int = 8):
    """Load raw messages, choosing the reader from ``fmt`` or the path shape."""
    if fmt == "sequence":
        return read_sequence_files(sc, path)
    if fmt == "text":
        return read_text_directory(sc, path, min_partitions)

    if path.endswith(".seq") or path.startswith("hdfs://") or path.startswith("/user/"):
        return read_sequence_files(sc, path)
    return read_text_directory(sc, path, min_partitions)
