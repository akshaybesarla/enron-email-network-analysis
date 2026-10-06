"""Tests for the pipeline stages.

The first test reproduces the exact expected output published with the original
course brief, which is the strongest available check that the extraction logic
still behaves as the specification required. The remaining tests cover the
constraints that the single sample message does not exercise: domain filtering,
self-loops, de-duplication, date-range slicing, zero-degree nodes and ordering.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from conftest import message

from src.pipeline import (
    convert_to_weighted_network,
    extract_email_network,
    get_in_degree_dist,
    get_in_degrees,
    get_monthly_contacts,
    get_out_degree_dist,
    get_out_degrees,
)

PDT = timezone(timedelta(hours=-7))
SAMPLE_TS = datetime(2000, 7, 31, 5, 48, tzinfo=PDT)

GEORGE = "george.mcclellan@enron.com"
SAMPLE_RECIPIENTS = [
    "mike.mcconnell@enron.com",
    "jeffrey.shankman@enron.com",
    "stuart.staley@enron.com",
    "daniel.reck@enron.com",
    "michael.beyer@enron.com",
    "kevin.mcgowan@enron.com",
]


# ---------------------------------------------------------------- extraction

def test_matches_published_expected_output(sc):
    """Reproduces the six triples published as expected output for Q1."""
    rdd = sc.parallelize([message(GEORGE, to=SAMPLE_RECIPIENTS)])
    result = set(extract_email_network(rdd).collect())

    assert result == {(GEORGE, r, SAMPLE_TS) for r in SAMPLE_RECIPIENTS}


def test_recipients_are_unioned_across_to_cc_bcc(sc):
    rdd = sc.parallelize([
        message(GEORGE,
                to=["a.one@enron.com"],
                cc=["b.two@enron.com"],
                bcc=["c.three@enron.com"])
    ])
    recipients = {r for _, r, _ in extract_email_network(rdd).collect()}

    assert recipients == {"a.one@enron.com", "b.two@enron.com", "c.three@enron.com"}


def test_subdomains_kept_and_lookalike_domains_rejected(sc):
    """sales.enron.com is internal; senron.com and notenron.com are not."""
    rdd = sc.parallelize([
        message(GEORGE, to=[
            "legit.user@sales.enron.com",
            "imposter.user@senron.com",
            "other.user@notenron.com",
            "external.user@ibm.com",
        ])
    ])
    recipients = {r for _, r, _ in extract_email_network(rdd).collect()}

    assert recipients == {"legit.user@sales.enron.com"}


def test_self_loops_removed(sc):
    rdd = sc.parallelize([message(GEORGE, to=[GEORGE, "a.one@enron.com"])])
    recipients = {r for _, r, _ in extract_email_network(rdd).collect()}

    assert recipients == {"a.one@enron.com"}


def test_duplicate_recipient_across_fields_counted_once(sc):
    """The same address in To and Cc is one transmission, not two."""
    rdd = sc.parallelize([message(GEORGE, to=["a.one@enron.com"], cc=["a.one@enron.com"])])

    assert extract_email_network(rdd).count() == 1


def test_messages_without_date_are_dropped(sc):
    rdd = sc.parallelize([
        "From: " + GEORGE + "\nTo: a.one@enron.com\nSubject: no date\n\nbody\n",
        message(GEORGE, to=["b.two@enron.com"]),
    ])
    recipients = {r for _, r, _ in extract_email_network(rdd).collect()}

    assert recipients == {"b.two@enron.com"}


def test_unparseable_date_does_not_kill_the_job(sc):
    rdd = sc.parallelize([
        message(GEORGE, to=["a.one@enron.com"], date="not a real date at all"),
        message(GEORGE, to=["b.two@enron.com"]),
    ])
    recipients = {r for _, r, _ in extract_email_network(rdd).collect()}

    assert recipients == {"b.two@enron.com"}


# ------------------------------------------------------------------ weighting

def _edges(sc):
    """Three senders, known multiplicities, two months."""
    jan = "Mon, 10 Jan 2000 09:00:00 -0700 (PDT)"
    feb = "Thu, 10 Feb 2000 09:00:00 -0700 (PDT)"
    return sc.parallelize([
        message("a@enron.com", to=["b@enron.com"], date=jan),
        message("a@enron.com", to=["b@enron.com"], date=feb),
        message("a@enron.com", to=["c@enron.com"], date=feb),
        message("b@enron.com", to=["c@enron.com"], date=feb),
    ])


def test_weights_count_messages_between_a_pair(sc):
    network = extract_email_network(_edges(sc))
    weighted = dict(
        ((s, r), w) for s, r, w in convert_to_weighted_network(network).collect()
    )

    assert weighted == {
        ("a@enron.com", "b@enron.com"): 2,
        ("a@enron.com", "c@enron.com"): 1,
        ("b@enron.com", "c@enron.com"): 1,
    }


def test_date_range_filters_before_counting(sc):
    network = extract_email_network(_edges(sc))
    february = (
        datetime(2000, 2, 1, tzinfo=timezone.utc),
        datetime(2000, 3, 1, tzinfo=timezone.utc),
    )
    weighted = dict(
        ((s, r), w) for s, r, w in convert_to_weighted_network(network, february).collect()
    )

    assert weighted == {
        ("a@enron.com", "b@enron.com"): 1,
        ("a@enron.com", "c@enron.com"): 1,
        ("b@enron.com", "c@enron.com"): 1,
    }


def test_reversed_date_range_behaves_like_an_ordered_one(sc):
    network = extract_email_network(_edges(sc))
    lo = datetime(2000, 2, 1, tzinfo=timezone.utc)
    hi = datetime(2000, 3, 1, tzinfo=timezone.utc)

    assert (sorted(convert_to_weighted_network(network, (lo, hi)).collect())
            == sorted(convert_to_weighted_network(network, (hi, lo)).collect()))


# -------------------------------------------------------------------- degrees

def test_every_node_appears_including_zero_degree(sc):
    """c only receives mail, so it must still appear with out-degree 0."""
    weighted = convert_to_weighted_network(extract_email_network(_edges(sc)))
    out_degrees = dict((node, degree) for degree, node in get_out_degrees(weighted).collect())

    assert out_degrees == {"a@enron.com": 3, "b@enron.com": 1, "c@enron.com": 0}


def test_in_degrees_mirror_out_degrees(sc):
    weighted = convert_to_weighted_network(extract_email_network(_edges(sc)))
    in_degrees = dict((node, degree) for degree, node in get_in_degrees(weighted).collect())

    assert in_degrees == {"a@enron.com": 0, "b@enron.com": 2, "c@enron.com": 2}


def test_total_in_degree_equals_total_out_degree(sc):
    weighted = convert_to_weighted_network(extract_email_network(_edges(sc))).cache()
    total_out = sum(d for d, _ in get_out_degrees(weighted).collect())
    total_in = sum(d for d, _ in get_in_degrees(weighted).collect())

    assert total_out == total_in


def test_degrees_sorted_descending_with_address_tiebreak(sc):
    """Ties on degree break on address descending, per the specification."""
    jan = "Mon, 10 Jan 2000 09:00:00 -0700 (PDT)"
    rdd = sc.parallelize([
        message("alpha@enron.com", to=["z.target@enron.com"], date=jan),
        message("bravo@enron.com", to=["z.target@enron.com"], date=jan),
    ])
    weighted = convert_to_weighted_network(extract_email_network(rdd))
    result = get_out_degrees(weighted).collect()

    assert result == sorted(result, reverse=True)
    assert result[0] == (1, "bravo@enron.com")
    assert result[1] == (1, "alpha@enron.com")


def test_degree_distribution_is_a_histogram_ascending_by_degree(sc):
    weighted = convert_to_weighted_network(extract_email_network(_edges(sc))).cache()

    assert get_out_degree_dist(weighted).collect() == [(0, 1), (1, 1), (3, 1)]
    assert get_in_degree_dist(weighted).collect() == [(0, 1), (2, 2)]


# ----------------------------------------------------------- monthly contacts

def test_monthly_contacts_matches_published_expected_output(sc):
    """Published expected output for Q5 on the single-message sample."""
    rdd = sc.parallelize([message(GEORGE, to=SAMPLE_RECIPIENTS)])
    result = get_monthly_contacts(extract_email_network(rdd)).collect()

    assert result == [(GEORGE, "07/2000", 6)]


def test_monthly_contacts_counts_distinct_people_not_messages(sc):
    """Three messages to the same person in one month is one contact, not three."""
    dates = [
        "Mon, 03 Jan 2000 09:00:00 -0700 (PDT)",
        "Tue, 04 Jan 2000 09:00:00 -0700 (PDT)",
        "Wed, 05 Jan 2000 09:00:00 -0700 (PDT)",
    ]
    rdd = sc.parallelize([message("a@enron.com", to=["b@enron.com"], date=d) for d in dates])
    result = get_monthly_contacts(extract_email_network(rdd)).collect()

    assert result == [("a@enron.com", "01/2000", 1)]


def test_monthly_contacts_picks_the_peak_month(sc):
    rdd = sc.parallelize([
        message("a@enron.com", to=["b@enron.com"], date="Mon, 03 Jan 2000 09:00:00 -0700 (PDT)"),
        message("a@enron.com", to=["b@enron.com", "c@enron.com", "d@enron.com"],
                date="Thu, 03 Feb 2000 09:00:00 -0700 (PDT)"),
    ])
    result = get_monthly_contacts(extract_email_network(rdd)).collect()

    assert result == [("a@enron.com", "02/2000", 3)]


def test_monthly_contacts_keeps_tied_months(sc):
    rdd = sc.parallelize([
        message("a@enron.com", to=["b@enron.com"], date="Mon, 03 Jan 2000 09:00:00 -0700 (PDT)"),
        message("a@enron.com", to=["c@enron.com"], date="Thu, 03 Feb 2000 09:00:00 -0700 (PDT)"),
    ])
    months = {m for _, m, _ in get_monthly_contacts(extract_email_network(rdd)).collect()}

    assert months == {"01/2000", "02/2000"}
