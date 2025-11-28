from abc import ABC, abstractmethod

import attr

from .renderer import SkeletonRenderer


@attr.s
class Skeleton:
    segments = attr.ib()

    def render(self, renderer: SkeletonRenderer):
        return renderer.combine([segment.render(renderer) for segment in self.segments])

    def error_free(self):
        return all(not isinstance(x, Correction) for x in self.segments)


class Segment(ABC):
    @abstractmethod
    def render(self, renderer: SkeletonRenderer):
        pass

    @abstractmethod
    def matcher_regex(self, quotation):
        pass


@attr.s
class Blank(Segment):
    solution = attr.ib()

    def render(self, renderer: SkeletonRenderer):
        return renderer.render_blank(self)

    def matcher_regex(self, quotation):
        return r"(.*?)"


@attr.s
class Given(Segment):
    code = attr.ib()

    def render(self, renderer: SkeletonRenderer):
        return renderer.render_given(self)

    def matcher_regex(self, quotation):
        return "(" + quotation(self.code) + ")"


@attr.s
class Correction(Segment):
    code = attr.ib()

    def render(self, renderer: SkeletonRenderer):
        return renderer.render_correction(self)

    def matcher_regex(self, quotation):
        raise NotImplementedError

    @property
    @abstractmethod
    def symbol(self):
        pass


class Insertion(Correction):
    symbol = "+"


class Deletion(Correction):
    symbol = "-"
