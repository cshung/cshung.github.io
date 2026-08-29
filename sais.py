"""A traceable implementation of the SA-IS suffix-array algorithm."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def suffix_array(
    text: str,
    *,
    trace: bool = False,
    verify: bool = False,
    verify_limit: int = 100,
) -> list[int]:
    """Return the starting positions of text's suffixes in sorted order."""
    alphabet = {ch: rank + 1 for rank, ch in enumerate(sorted(set(text)))}
    encoded = [alphabet[ch] for ch in text] + [0]

    sa = _sais(
        encoded,
        alphabet_size=len(alphabet) + 1,
        trace=trace,
        verify=verify,
        verify_limit=verify_limit,
        depth=0,
    )

    # The sentinel-only suffix is always first and is not part of text.
    assert sa[0] == len(text)
    return sa[1:]


def _sais(
    s: list[int],
    alphabet_size: int,
    *,
    trace: bool,
    verify: bool,
    verify_limit: int,
    depth: int,
) -> list[int]:
    n = len(s)
    indent = "  " * depth

    assert n >= 1
    assert s[-1] == 0
    assert s.count(0) == 1
    assert all(0 < symbol < alphabet_size for symbol in s[:-1])

    def trace_line(text: str) -> None:
        if not trace:
            return

        print(f"\n{indent}{text}")

    def show_table(
        title: str,
        *columns: tuple[str, Sequence[object]],
    ) -> None:
        if not trace:
            return

        lengths = {len(values) for _, values in columns}
        assert len(lengths) == 1

        text_columns = [
            (heading, [str(value) for value in values])
            for heading, values in columns
        ]
        widths = [
            max(len(heading), *(len(value) for value in values))
            for heading, values in text_columns
        ]

        print(f"\n{indent}{title}")
        print(
            f"{indent}| "
            + " | ".join(
                f"{heading:<{width}}"
                for (heading, _), width in zip(text_columns, widths)
            )
            + " |"
        )
        print(
            f"{indent}|-"
            + "-|-".join("-" * width for width in widths)
            + "-|"
        )

        row_count = lengths.pop()
        for row_index in range(row_count):
            print(
                f"{indent}| "
                + " | ".join(
                    f"{values[row_index]:<{width}}"
                    for (_, values), width in zip(text_columns, widths)
                )
                + " |"
            )

    def show_sa(title: str, sa: Sequence[int]) -> None:
        suffixes = [
            "·" if pos == -1 else "".join(map(str, s[pos:]))
            for pos in sa
        ]
        show_table(
            title,
            ("slot", list(range(n))),
            ("suffix start", ["·" if pos == -1 else pos for pos in sa]),
            ("suffix", suffixes),
        )

    if n == 1:
        show_sa("Base case", [0])
        return [0]

    # S-type means this suffix is smaller than the suffix immediately after it.
    is_s = [False] * n
    is_s[-1] = True

    for i in range(n - 2, -1, -1):
        is_s[i] = s[i] < s[i + 1] or (
            s[i] == s[i + 1] and is_s[i + 1]
        )

    def is_lms(i: int) -> bool:
        return i > 0 and is_s[i] and not is_s[i - 1]

    lms_positions = [i for i in range(1, n) if is_lms(i)]
    types = ["LMS" if is_lms(i) else "S" if is_s[i] else "L" for i in range(n)]

    trace_line(f"SA-IS call at recursion depth {depth}")
    show_table(
        "Classify suffixes",
        ("position", list(range(n))),
        ("symbol", s),
        ("type", types),
    )

    counts = [0] * alphabet_size
    for symbol in s:
        counts[symbol] += 1

    def bucket_heads() -> list[int]:
        heads = [0] * alphabet_size
        total = 0
        for symbol, count in enumerate(counts):
            heads[symbol] = total
            total += count
        return heads

    def bucket_tails() -> list[int]:
        tails = [0] * alphabet_size
        total = 0
        for symbol, count in enumerate(counts):
            total += count
            tails[symbol] = total - 1
        return tails

    def induced_sort(lms_order: Sequence[int], phase: str) -> list[int]:
        sa = [-1] * n

        # LMS suffixes are S-type, so seed them at their bucket tails.
        tails = bucket_tails()
        for pos in reversed(lms_order):
            symbol = s[pos]
            sa[tails[symbol]] = pos
            tails[symbol] -= 1
        show_sa(f"{phase}: place LMS suffixes", sa)

        # Scanning left-to-right discovers L-type predecessors in sorted order.
        heads = bucket_heads()
        for slot in range(n):
            pos = sa[slot]
            if pos <= 0:
                continue

            predecessor = pos - 1
            if not is_s[predecessor]:
                symbol = s[predecessor]
                sa[heads[symbol]] = predecessor
                heads[symbol] += 1
        show_sa(f"{phase}: induce L suffixes", sa)

        # The symmetric right-to-left pass discovers S-type predecessors.
        tails = bucket_tails()
        for slot in range(n - 1, -1, -1):
            pos = sa[slot]
            if pos <= 0:
                continue

            predecessor = pos - 1
            if is_s[predecessor]:
                symbol = s[predecessor]
                sa[tails[symbol]] = predecessor
                tails[symbol] -= 1
        show_sa(f"{phase}: induce S suffixes", sa)

        return sa

    first_sa = induced_sort(lms_positions, "First induced sort")
    sorted_lms = [pos for pos in first_sa if is_lms(pos)]

    def same_lms_substring(a: int, b: int) -> bool:
        """Compare through, and including, the next LMS position."""
        offset = 0

        while True:
            a_pos = a + offset
            b_pos = b + offset

            if s[a_pos] != s[b_pos] or is_s[a_pos] != is_s[b_pos]:
                return False

            if offset > 0:
                a_at_end = is_lms(a_pos)
                b_at_end = is_lms(b_pos)

                if a_at_end or b_at_end:
                    return a_at_end and b_at_end

            offset += 1

    # Equal adjacent LMS substrings receive the same integer name.
    name_at_position = [-1] * n
    name = -1
    previous = -1

    for pos in sorted_lms:
        if previous == -1 or not same_lms_substring(previous, pos):
            name += 1
        name_at_position[pos] = name
        previous = pos

    # Reading names in text order creates the smaller recursive string.
    reduced = [name_at_position[pos] for pos in lms_positions]
    number_of_names = name + 1

    show_table(
        "Name LMS substrings in sorted order",
        ("rank", list(range(len(sorted_lms)))),
        ("LMS position", sorted_lms),
        ("name", [name_at_position[pos] for pos in sorted_lms]),
    )
    show_table(
        "Build the reduced string in text order",
        ("text index", list(range(len(lms_positions)))),
        ("LMS position", lms_positions),
        ("name", reduced),
    )

    if number_of_names == len(lms_positions):
        # Distinct names are already ranks, so inversion gives sorted LMS order.
        ordered_lms = [0] * len(lms_positions)
        for text_order_index, lms_name in enumerate(reduced):
            ordered_lms[lms_name] = lms_positions[text_order_index]
        show_table(
            "All LMS names are distinct",
            ("rank", list(range(len(ordered_lms)))),
            ("LMS position", ordered_lms),
        )
    else:
        reduced_sa = _sais(
            reduced,
            alphabet_size=number_of_names,
            trace=trace,
            verify=verify,
            verify_limit=verify_limit,
            depth=depth + 1,
        )
        ordered_lms = [lms_positions[index] for index in reduced_sa]
        show_table(
            "Recover LMS order from recursive suffix array",
            ("rank", list(range(len(ordered_lms)))),
            ("reduced suffix", reduced_sa),
            ("LMS position", ordered_lms),
        )

    final_sa = induced_sort(ordered_lms, "Final induced sort")

    if verify and n <= verify_limit:
        expected = sorted(range(n), key=lambda pos: s[pos:])
        assert final_sa == expected, (
            f"SA-IS result differs from direct sorting:\n"
            f"actual:   {final_sa}\n"
            f"expected: {expected}"
        )

    return final_sa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", default="banana")
    parser.add_argument(
        "--trace",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    sa = suffix_array(args.text, trace=args.trace, verify=args.verify)
    print(f"\nSuffix array: {sa}")
    for pos in sa:
        print(f"{pos:>4}: {args.text[pos:]}")


if __name__ == "__main__":
    main()
