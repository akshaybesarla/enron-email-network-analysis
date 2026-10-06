"""
Enron email network pipeline.

Transforms raw RFC-822 email messages into a weighted directed communication
graph, then derives degree statistics and degree distributions from it.

Pipeline stages
---------------
    raw messages            ->  extract_email_network()
    (sender, recipient, ts) ->  convert_to_weighted_network()
    (sender, recipient, w)  ->  get_out_degrees() / get_in_degrees()
    (degree, node)          ->  get_out_degree_dist() / get_in_degree_dist()

Everything uses the Spark RDD API rather than DataFrames or Spark SQL. That was
a constraint of the original brief, and it is kept here deliberately: the point
of the project is to reason about transformations, partitioning and shuffle
boundaries directly, without a query optimiser in the way.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone
from email.parser import Parser

# An address is accepted when:
#   - the local part contains no whitespace and no '@'
#   - the domain is zero or more dot-terminated labels followed by "enron.com"
#
# Anchoring the tail as a full label pair is what rejects near-misses such as
# "joe@senron.com" or "joe@notenron.com", while still accepting legitimate
# subdomains such as "joe@sales.enron.com".
ENRON_ADDRESS = re.compile(r"^[^\s@]+@([a-zA-Z0-9-]+\.)*enron\.com$")

RECIPIENT_FIELDS = ("To", "Cc", "Bcc")


def date_to_dt(date: str) -> datetime:
    """Convert an RFC-822 Date header into a timezone-aware datetime.

    Enron headers carry a trailing timezone abbreviation in parentheses, e.g.
    "Mon, 31 Jul 2000 05:48:00 -0700 (PDT)". The last six characters are
    stripped so the numeric offset is the final token for %z.

    Provided by the original course template and left unchanged.
    """

    def to_dt(tms):
        def tz():
            return timezone(timedelta(seconds=tms.tm_gmtoff))

        return datetime(
            tms.tm_year,
            tms.tm_mon,
            tms.tm_mday,
            tms.tm_hour,
            tms.tm_min,
            tms.tm_sec,
            tzinfo=tz(),
        )

    return to_dt(time.strptime(date[:-6], "%a, %d %b %Y %H:%M:%S %z"))


def _parse_message(raw: str):
    """Parse one raw message into (sender, [recipients], raw_date_string).

    Returns None when the message has no usable Date header, since every output
    triple is required to carry a timestamp.
    """
    message = Parser().parsestr(raw)

    sender = (message.get("From") or "").strip()
    raw_date = (message.get("Date") or "").strip()
    if not sender or not raw_date:
        return None

    recipients = [
        address.strip()
        for field in RECIPIENT_FIELDS
        for address in (message.get(field) or "").split(",")
        if address.strip()
    ]
    if not recipients:
        return None

    return sender, recipients, raw_date


def _safe_date(raw_date: str):
    """Parse a Date header, returning None rather than raising on malformed input.

    Real corpora contain unparseable headers. On a cluster an exception here
    kills the whole task, so bad records are dropped instead.
    """
    try:
        return date_to_dt(raw_date)
    except (ValueError, TypeError, AttributeError):
        return None


def extract_email_network(rdd):
    """Raw messages -> distinct (sender, recipient, timestamp) triples.

    One message with N recipients becomes N triples: ``flatMap`` is what turns a
    collection of messages into a collection of transmissions.

    Constraints applied:
      - both addresses must be valid and in the enron.com domain
      - self-loops are removed
      - the output is distinct

    ``distinct()`` matters because the same person can appear in both To and Cc
    of a single message, which would otherwise double-count one transmission.
    """
    parsed = rdd.map(_parse_message).filter(lambda x: x is not None)

    dated = (
        parsed
        .map(lambda x: (x[0], x[1], _safe_date(x[2])))
        .filter(lambda x: x[2] is not None)
    )

    edges = dated.flatMap(
        lambda x: [(x[0], recipient, x[2]) for recipient in x[1]]
    )

    return edges.filter(
        lambda e: (
            e[0] != e[1]
            and ENRON_ADDRESS.match(e[0]) is not None
            and ENRON_ADDRESS.match(e[1]) is not None
        )
    ).distinct()


def convert_to_weighted_network(rdd, drange=None):
    """Timestamped edges -> (sender, recipient, weight) triples.

    Collapses repeated transmissions between the same pair into a single edge
    whose weight is the message count. Mechanically this is word count with a
    pair as the key instead of a word.

    ``drange`` is an optional (start, end) pair of datetimes. It is applied
    before counting, which is what makes time-sliced analysis possible. The
    bounds are normalised so a reversed pair behaves the same as an ordered one.
    """
    edges = rdd
    if drange:
        lo, hi = (drange[0], drange[1]) if drange[0] <= drange[1] else (drange[1], drange[0])
        edges = edges.filter(lambda e: lo <= e[2] <= hi)

    return (
        edges
        .map(lambda e: ((e[0], e[1]), 1))
        .reduceByKey(lambda a, b: a + b)
        .map(lambda kv: (kv[0][0], kv[0][1], kv[1]))
    )


def _degrees(rdd, key_index: int, zero_index: int):
    """Shared implementation for weighted in/out degree.

    The subtlety is nodes of degree zero. A node that only ever receives mail
    never appears in the sender column, so a naive aggregation silently drops
    it. Every node is therefore seeded with a zero from the opposite column and
    unioned in before the reduce: the zeros contribute nothing arithmetically
    but force the key to exist.

    Output is (degree, address) sorted in descending lexicographic order of the
    pair, so ties on degree break on address descending.
    """
    contributions = rdd.map(lambda e: (e[key_index], e[2]))
    zero_seeds = rdd.map(lambda e: (e[zero_index], 0))

    return (
        zero_seeds
        .union(contributions)
        .reduceByKey(lambda a, b: a + b)
        .map(lambda kv: (kv[1], kv[0]))
        .sortBy(lambda pair: pair, ascending=False)
    )


def get_out_degrees(rdd):
    """Weighted out-degree per node: total messages sent."""
    return _degrees(rdd, key_index=0, zero_index=1)


def get_in_degrees(rdd):
    """Weighted in-degree per node: total messages received."""
    return _degrees(rdd, key_index=1, zero_index=0)


def _degree_dist(degree_rdd):
    """(degree, node) pairs -> (degree, number of nodes), ascending by degree."""
    return (
        degree_rdd
        .map(lambda pair: (pair[0], 1))
        .reduceByKey(lambda a, b: a + b)
        .sortByKey()
    )


def get_out_degree_dist(rdd):
    """Histogram of weighted out-degrees. Input to the power-law analysis."""
    return _degree_dist(get_out_degrees(rdd))


def get_in_degree_dist(rdd):
    """Histogram of weighted in-degrees. Input to the power-law analysis."""
    return _degree_dist(get_in_degrees(rdd))


def get_monthly_contacts(rdd):
    """Per sender, the month in which they contacted the most distinct people.

    Ties are kept: if a sender peaks at the same count in several months, every
    one of those months is emitted.

    Two aggregation passes:
      1. key by (sender, month), count distinct recipients
      2. re-key by sender, keep the months achieving the maximum

    Implementation note. The original version used ``groupByKey`` for both
    passes. ``groupByKey`` ships every value across the network before
    combining, whereas ``aggregateByKey`` and ``reduceByKey`` combine map-side
    first and shuffle far less. Pass 1 now builds the recipient set with
    ``aggregateByKey``; pass 2 computes the maximum with ``reduceByKey`` and
    joins back, instead of materialising every month for a sender on one
    executor. The original also recomputed ``max(...)`` inside a comprehension,
    making pass 2 quadratic in the number of months per sender.
    """

    def month_key(e):
        return (e[0], f"{e[2].month:02d}/{e[2].year}"), e[1]

    def seq_op(acc, recipient):
        acc.add(recipient)
        return acc

    def comb_op(a, b):
        a |= b
        return a

    # (sender, month) -> number of distinct recipients
    per_month = (
        rdd
        .map(month_key)
        .aggregateByKey(set(), seq_op, comb_op)
        .mapValues(len)
        .map(lambda kv: (kv[0][0], (kv[0][1], kv[1])))
    )

    # sender -> best count
    best = per_month.mapValues(lambda mv: mv[1]).reduceByKey(max)

    return (
        per_month
        .join(best)
        .filter(lambda kv: kv[1][0][1] == kv[1][1])
        .map(lambda kv: (kv[0], kv[1][0][0], kv[1][0][1]))
        .sortBy(lambda t: (t[2], t[0]), ascending=False)
    )
