import re
from cached_property import cached_property

from skeletonize.skeleton import Given, Skeleton, Blank


class SkeletonParser:
    def __init__(self, start_char="{", end_char="}", number_to_match=2):
        self.start_char = start_char
        self.end_char = end_char
        self.number_to_match = number_to_match

    def parse(self, skeleton_code):
        self._check_too_long(skeleton_code)

        chunks = []

        def add_given(text):
            self._check_no_delimiters(text)
            if text:
                chunks.append(Given(text))

        def add_blank(text):
            self._check_no_delimiters(text)
            chunks.append(Blank(text))

        index = 0
        for match in self._blank_pattern.finditer(skeleton_code):
            add_given(skeleton_code[index : match.start()])
            add_blank(match.group(1))
            index = match.end()

        add_given(skeleton_code[index:])
        return Skeleton(chunks)

    @cached_property
    def _blank_pattern(self):
        start = re.escape(self.start_char * self.number_to_match)
        end = re.escape(self.end_char * self.number_to_match)
        return re.compile(r"{}(.*?){}".format(start, end), flags=re.M | re.S)

    def _bad_pattern(self, count):
        start = re.escape(self.start_char * count)
        end = re.escape(self.end_char * count)
        return re.compile(r"{}|{}".format(start, end), flags=re.M | re.S)

    def _check_too_long(self, skeleton_code):
        match = self._bad_pattern(self.number_to_match + 1).findall(skeleton_code)
        if match:
            raise TooLongDelimiterException(str(match))

    def _check_no_delimiters(self, text):
        match = self._bad_pattern(self.number_to_match).findall(text)
        if match:
            raise ExtraDelimiterException(str(match))


class TooLongDelimiterException(Exception):
    pass


class ExtraDelimiterException(Exception):
    pass
