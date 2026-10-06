"""Shared pytest fixtures: one local SparkContext for the whole session."""

from __future__ import annotations

import pytest
from pyspark import SparkConf, SparkContext


@pytest.fixture(scope="session")
def sc():
    conf = (
        SparkConf()
        .setAppName("enron-tests")
        .setMaster("local[2]")
        .set("spark.ui.enabled", "false")
        .set("spark.sql.shuffle.partitions", "2")
    )
    context = SparkContext.getOrCreate(conf=conf)
    context.setLogLevel("ERROR")
    yield context
    context.stop()


def message(sender: str, to=(), cc=(), bcc=(), date: str = "Mon, 31 Jul 2000 05:48:00 -0700 (PDT)") -> str:
    """Build a minimal but realistically shaped RFC-822 message."""
    headers = [f"Date: {date}", f"From: {sender}"]
    if to:
        headers.append(f"To: {', '.join(to)}")
    if cc:
        headers.append(f"Cc: {', '.join(cc)}")
    if bcc:
        headers.append(f"Bcc: {', '.join(bcc)}")
    headers.append("Subject: test")
    return "\n".join(headers) + "\n\nbody\n"
