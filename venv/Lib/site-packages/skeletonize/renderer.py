import random
import re
from abc import ABC, abstractmethod

import attr


class SkeletonRenderer(ABC):
    @abstractmethod
    def combine(self, per_segment_outputs):
        pass

    @abstractmethod
    def render_blank(self, blank):
        pass

    @abstractmethod
    def render_given(self, given):
        pass

    @abstractmethod
    def render_correction(self, correction):
        pass


@attr.s
class UnderscoreBlankRenderer(SkeletonRenderer):
    blank_size = attr.ib(default=6)

    def combine(self, per_segment_outputs):
        return "".join(per_segment_outputs)

    def render_blank(self, blank):
        return "_" * self.blank_size

    def render_given(self, given):
        return given.code

    def render_correction(self, correction):
        raise ValueError(
            "Cannot render with blanks a skeleton that contains corrections"
        )


@attr.s
class DisplaySolutionsRenderer(SkeletonRenderer):
    start = attr.ib(default="{{")
    end = attr.ib(default="}}")

    start_correction = attr.ib(default="<<")
    end_correction = attr.ib(default=">>")

    def combine(self, per_segment_outputs):
        return "".join(per_segment_outputs)

    def render_blank(self, blank):
        return self.start + blank.solution + self.end

    def render_given(self, given):
        return given.code

    def render_correction(self, correction):
        return (
            self.start_correction
            + correction.symbol
            + correction.code
            + self.end_correction
        )


class IdentifierBlankRenderer(SkeletonRenderer):
    """
    Renders each blank as an identifier, as determined by the provided blanks_map.

    Arguments:
        blanks_map: mapping from id(blank) -> identifier
    """

    def __init__(self, blanks_map):
        self.blanks_map = blanks_map

    def combine(self, per_segment_outputs):
        return "".join(per_segment_outputs)

    def render_blank(self, blank):
        return self.blanks_map[id(blank)]

    def render_given(self, given):
        return given.code

    def render_correction(self, correction):
        raise ValueError(
            "Cannot render with blanks a skeleton that contains corrections"
        )


def render_with_identifiers(skeleton, identifier_length=30):
    """
    Render the given skeleton with random identifiers per blank. The identifiers are each
    drawn from the set {a...z}^identifier_length

    :param skeleton: a Skeleton object containing givens and blanks.
    :param identifier_length: the length of the identifiers to insert
    :return: the code with the inserted identifiers, and a mapping from each identifier to its corresponding blank
    """
    from skeletonize.skeleton import Blank

    def generate_id():
        return "".join(
            chr(ord("a") + random.randint(0, 25)) for _ in range(identifier_length)
        )

    identifiers = {
        generate_id(): blank for blank in skeleton.segments if isinstance(blank, Blank)
    }

    rendered_code = skeleton.render(
        IdentifierBlankRenderer(
            {id(blank): identifier for identifier, blank in identifiers.items()}
        )
    )

    return rendered_code, identifiers


class DuplicateIdentifierException(Exception):
    pass


def parse_identifiers(code, identifier_to_blank):
    """
    Parses the given code into a skeleton, using each identifier in identifier_to_blank and converting it to the
    corresponding blank

    :param code: the code to process and convert to a skeleton
    :param identifier_to_blank: a mapping from identifier -> Blank object
    :return: a Skeleton containing Given sections from the code and Blank sections from identifiers_for_blank
    """
    from .parser import SkeletonParser
    from .skeleton import Skeleton, Given

    if not identifier_to_blank:
        return Skeleton([Given(code)])

    pattern = re.compile(
        "|".join(re.escape(identifier) for identifier in identifier_to_blank)
    )
    used = set()

    def substitution(match):
        identifier = match.group()
        if identifier in used:
            raise DuplicateIdentifierException(identifier)
        used.add(identifier)
        solution = identifier_to_blank[identifier].solution
        return "\x00" * 2 + solution + "\x01" * 2

    code = pattern.sub(substitution, code)
    return SkeletonParser(start_char="\x00", end_char="\x01").parse(code)
