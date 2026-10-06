"""Generate a synthetic Enron-shaped corpus so the pipeline can be run locally.

The real Enron corpus used in the original project lived on a university HDFS
cluster and is not redistributable here. This generator produces messages in the
same RFC-822 shape, with a deliberately heavy-tailed sender distribution, so the
pipeline and the network analysis both exercise realistically.

This is synthetic data. Every address and name is invented. Numbers produced
from it will not match the figures reported in the README, which came from the
real 0.5M-message corpus.

Usage
-----
    python -m src.generate_data --messages 20000 --out data/raw
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

FIRST_NAMES = [
    "james", "mary", "robert", "patricia", "john", "jennifer", "michael",
    "linda", "david", "elizabeth", "william", "barbara", "richard", "susan",
    "joseph", "jessica", "thomas", "sarah", "charles", "karen", "daniel",
    "nancy", "matthew", "lisa", "anthony", "betty", "mark", "margaret",
    "donald", "sandra", "steven", "ashley", "paul", "kimberly", "andrew",
]

LAST_NAMES = [
    "smith", "johnson", "williams", "brown", "jones", "garcia", "miller",
    "davis", "rodriguez", "martinez", "hernandez", "lopez", "gonzalez",
    "wilson", "anderson", "thomas", "taylor", "moore", "jackson", "martin",
    "lee", "perez", "thompson", "white", "harris", "sanchez", "clark",
    "ramirez", "lewis", "robinson", "walker", "young", "allen", "king",
]

SUBDOMAINS = ["", "", "", "", "sales.", "trading.", "legal.", "corp."]

SUBJECTS = [
    "Re: Weekly position report",
    "Gas nominations for tomorrow",
    "FW: Contract review",
    "Meeting moved to 3pm",
    "Q3 numbers",
    "Re: Counterparty credit limits",
    "Staffing for the Houston desk",
    "Please review attached schedule",
    "Re: Pipeline maintenance window",
    "Approval needed on deal 4471",
]

BODY_LINES = [
    "Please see the attached and let me know if anything looks off.",
    "I have updated the sheet with the latest numbers.",
    "Can we push this to next week?",
    "Confirmed on my end.",
    "Looping in the rest of the desk.",
    "Let me know if you need anything else from me on this.",
    "Following up on the below.",
    "Thanks for turning this around so quickly.",
]

# Timezone offsets seen in the real corpus (US business hours, DST both ways).
OFFSETS = [("-0700", "PDT"), ("-0800", "PST"), ("-0500", "CDT"), ("-0600", "CST")]


def make_address(rng: random.Random) -> str:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    return f"{first}.{last}@{rng.choice(SUBDOMAINS)}enron.com"


def build_population(rng: random.Random, n_people: int) -> list[str]:
    """Build a de-duplicated address book of the requested size."""
    people: set[str] = set()
    guard = 0
    while len(people) < n_people and guard < n_people * 50:
        people.add(make_address(rng))
        guard += 1
    return sorted(people)


def zipf_weights(n: int, exponent: float) -> list[float]:
    """Rank-based weights giving a heavy-tailed activity distribution."""
    return [1.0 / ((rank + 1) ** exponent) for rank in range(n)]


def format_message(
    sender: str,
    to: list[str],
    cc: list[str],
    sent_at: datetime,
    offset: tuple[str, str],
    rng: random.Random,
) -> str:
    numeric, abbrev = offset
    date_header = sent_at.strftime("%a, %d %b %Y %H:%M:%S ") + f"{numeric} ({abbrev})"

    headers = [
        f"Message-ID: <{rng.randrange(10**7, 10**8)}.{rng.randrange(10**6, 10**7)}.JavaMail.evans@thyme>",
        f"Date: {date_header}",
        f"From: {sender}",
        f"To: {', '.join(to)}",
    ]
    if cc:
        headers.append(f"Cc: {', '.join(cc)}")
    headers.extend(
        [
            f"Subject: {rng.choice(SUBJECTS)}",
            "Mime-Version: 1.0",
            "Content-Type: text/plain; charset=us-ascii",
            "Content-Transfer-Encoding: 7bit",
        ]
    )

    body = "\n".join(rng.sample(BODY_LINES, k=rng.randint(1, 3)))
    return "\n".join(headers) + "\n\n" + body + "\n"


def generate(
    messages: int,
    people: int,
    out_dir: Path,
    seed: int,
    start: datetime,
    months: int,
) -> None:
    rng = random.Random(seed)
    population = build_population(rng, people)

    # Senders are heavily skewed (a few hubs send most mail); recipients are
    # skewed too but less sharply. This is what makes the degree distributions
    # worth analysing rather than uniform noise.
    send_weights = zipf_weights(len(population), exponent=1.15)
    recv_order = population[:]
    rng.shuffle(recv_order)
    recv_weights = zipf_weights(len(recv_order), exponent=0.85)

    # People join over time rather than all existing from day one. Without this
    # the node count saturates in month one while degrees keep climbing, and the
    # k_max-against-n growth analysis measures nothing.
    n = len(population)
    entry_month = {
        person: int(months * (rank / n) ** 1.4)
        for rank, person in enumerate(population)
    }

    span_days = months * 30
    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.glob("msg_*.txt"):
        existing.unlink()

    send_weight_of = dict(zip(population, send_weights))
    recv_weight_of = dict(zip(recv_order, recv_weights))

    width = len(str(messages))
    written = 0
    for i in range(messages):
        # Pick when first, then draw only from people already active by then.
        day = rng.randrange(span_days)
        month_index = day // 30
        active = [p for p in population if entry_month[p] <= month_index]
        if len(active) < 2:
            continue

        sender = rng.choices(active, weights=[send_weight_of[p] for p in active], k=1)[0]

        n_to = rng.choices([1, 2, 3, 5, 8, 15], weights=[55, 20, 12, 7, 4, 2], k=1)[0]
        pool = [p for p in recv_order if entry_month[p] <= month_index]
        recipients = rng.choices(pool, weights=[recv_weight_of[p] for p in pool], k=n_to)
        recipients = [r for r in dict.fromkeys(recipients) if r != sender]
        if not recipients:
            continue

        split = len(recipients) if rng.random() < 0.75 else rng.randint(1, len(recipients))
        to, cc = recipients[:split], recipients[split:]

        sent_at = start + timedelta(
            days=day,
            hours=rng.randrange(6, 21),
            minutes=rng.randrange(60),
            seconds=rng.randrange(60),
        )

        message = format_message(sender, to, cc, sent_at, rng.choice(OFFSETS), rng)
        (out_dir / f"msg_{i:0{width}d}.txt").write_text(message, encoding="utf-8")
        written += 1

    print(f"Wrote {written} messages from {len(population)} addresses to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", type=int, default=20000)
    parser.add_argument("--people", type=int, default=1200)
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--months", type=int, default=24)
    args = parser.parse_args()

    generate(
        messages=args.messages,
        people=args.people,
        out_dir=args.out,
        seed=args.seed,
        start=datetime.strptime(args.start, "%Y-%m-%d"),
        months=args.months,
    )


if __name__ == "__main__":
    main()
