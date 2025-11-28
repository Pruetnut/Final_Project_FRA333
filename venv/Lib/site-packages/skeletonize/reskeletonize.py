import re

import astunparse
import ast
import attr

from .renderer import render_with_identifiers, parse_identifiers
from .skeleton import Skeleton, Blank, Given
from .align import align_skeleton


@attr.s
class Reskeletonizer:
    ignore_whitespace = attr.ib(default=False)
    allow_newline_blanks = attr.ib(default=False)
    normalizer = attr.ib(default=None)
    """
    Reskeletonizer that uses character-level matching to match given portions the skeleton
        in the provided code, in order to find blanks.

    Arguments
        ignore_whitespace: whether to ignore whitespace when resolving blanks. Default False
        normalizer: function that parses and unparses code, to remove extraneous formatting.
            Default is None, or no normalization. If this is used, the returned reskeletonized
            code will correspond to the normalized code rather than the original code.

            For this to work, the blanks in the original skeleton must correspond to places an
            identifier could be placed, syntactically. Additionally, any sequence of [a-z]*
            must be a valid identifier for this to work.
    """

    def reskeletonize(self, skeleton: Skeleton, code: str) -> Skeleton:
        if self.normalizer is not None:
            new_code = self.normalizer(code)
            if new_code is not None:
                code = new_code
                skeleton_code, skeleton_ids = render_with_identifiers(skeleton)
                skeleton_code = self.normalizer(skeleton_code)
                skeleton = parse_identifiers(skeleton_code, skeleton_ids)

        return align_skeleton(
            skeleton, code, self.ignore_whitespace, self.allow_newline_blanks
        )

    def create_regex(self, skeleton):
        regex_chunks = ["^"]
        for segment in skeleton.segments:
            regex_chunks.append(segment.matcher_regex(self._match_given_text))
        regex_chunks.append("$")
        pattern = "".join(regex_chunks)
        return re.compile(pattern, re.DOTALL)

    def _match_given_text(self, code):
        if not self.ignore_whitespace:
            return re.escape(code)
        return r"\s+".join(re.escape(word) for word in re.split(r"\s+", code))


class RemoveDocstring(ast.NodeTransformer):
    replacement = "[ommited docstring]"

    def visit_Module(self, node):
        for sub in node.body:
            super().visit(sub)
            if not isinstance(sub, ast.Expr):
                continue
            if hasattr(ast, "Constant"):
                # python3.5 doesn't have this and we support it.
                if isinstance(sub.value, ast.Constant) and isinstance(
                    sub.value.value, str
                ):
                    sub.value.value = self.replacement
                    continue
            if isinstance(sub.value, ast.Str):
                sub.value.s = self.replacement
                continue
        return node

    visit_ClassDef = visit_FunctionDef = visit_AsyncFunctionDef = visit_Module


def normalize_python(code, remove_docstrings=True):
    """
    Parses and unparses python, while also removing docstrings and module strings
    and replacing them with '[ommited docstring]'
    """
    try:
        tree = ast.parse(code, "<<code>>")
    except SyntaxError:
        return None

    if remove_docstrings:
        tree = RemoveDocstring().visit(tree)

    return astunparse.unparse(tree)
