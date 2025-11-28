import sys
from functools import lru_cache

from .tokenize import tokenize
from .skeleton import Skeleton, Given, Insertion, Deletion, Blank, Correction


def align_skeleton(skeleton, code, is_whitespace, allow_newline_blanks):
    """
    Aligns the given skeleton with the given code. This algorithm minimizes
        the edit distance (just insert/delete) of all the Given portions of
        the skeleton with that of the code.
    """
    code = tokenize(code)
    segments = [
        tokenize(x.code) if isinstance(x, Given) else Blank for x in skeleton.segments
    ]

    @lru_cache(None)
    def helper_align(segment_idx, within_segment_idx, code_idx):
        """
        Aligns the given skeletal segments with the code.

        Returns (match, cost)
            match: the sequence of corrections as a linked list of tuples
            cost: the cost of the corrections, in edits
        """
        if segment_idx == len(segments) and code_idx == len(code):
            return (), 0
        if segment_idx > len(segments) or code_idx > len(code):
            return None, float("inf")

        segm = segments[segment_idx] if segment_idx < len(segments) else None
        code_char = code[code_idx] if code_idx < len(code) else None

        if segm is Blank:
            possibilities = []

            # do not use blank
            possibilities.append(helper_align(segment_idx + 1, 0, code_idx))

            # use blank?
            if code_char is not None and (allow_newline_blanks or code_char != "\n"):
                s, c = helper_align(segment_idx, within_segment_idx, code_idx + 1)
                new_s = Blank(code_char), s
                possibilities.append((new_s, c))

            return min(possibilities, key=lambda x: x[1])

        if segm is not None and within_segment_idx == len(segm):
            return helper_align(segment_idx + 1, 0, code_idx)

        segm_char = segm[within_segment_idx] if segm is not None else None

        # match?
        if segm_char == code_char:
            s, c = helper_align(segment_idx, within_segment_idx + 1, code_idx + 1)
            new_s = Given(code_char), s
            # in theory, this should be added to the possibilites list and compared
            # to everything else, but this hack speeds up the calculation
            # considerably, by assuming that we can just correct later.
            # :shrug emoji:
            return new_s, c

        possibilities = []
        # insert?
        if code_char is not None:
            s, c = helper_align(segment_idx, within_segment_idx, code_idx + 1)
            if is_whitespace and code_char.isspace():
                res = (Given(code_char), s), c + 0.0001
            else:
                res = (Insertion(code_char), s), c + len(code_char)
            possibilities.append(res)

        # delete
        if segm_char is not None:
            s, c = helper_align(segment_idx, within_segment_idx + 1, code_idx)
            if is_whitespace and segm_char.isspace():
                res = s, c + 0.0001
            else:
                res = (Deletion(segm_char), s), c + len(segm_char)
            possibilities.append(res)

        return min(possibilities, key=lambda x: x[1])

    with recursionlimit(10 ** 5):
        s, _ = helper_align(0, 0, 0)
    return Skeleton(consolidate(to_list(s)))


def to_list(ll):
    result = []
    while ll:
        result.append(ll[0])
        ll = ll[1]
    return result


def consolidate(segments):
    result = []
    for x in segments:
        if not result or type(result[-1]) != type(x):
            result.append(x)
            continue
        if isinstance(x, (Given, Correction)):
            result[-1] = type(x)(result[-1].code + x.code)
        elif isinstance(x, Blank):
            result[-1] = type(x)(result[-1].solution + x.solution)
        else:
            raise AssertionError("unreachable")
    return result


class recursionlimit:
    # from https://stackoverflow.com/a/50120316/1549476
    def __init__(self, limit):
        self.limit = limit
        self.old_limit = sys.getrecursionlimit()

    def __enter__(self):
        sys.setrecursionlimit(self.limit)

    def __exit__(self, type, value, tb):
        sys.setrecursionlimit(self.old_limit)
