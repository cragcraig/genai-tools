"""AST data structures and parser for prompt templates.

A prompt template is parsed into a tree of nodes that can contain:
  - Literal text segments
  - Inline image references via @path syntax
  - Variant expansion groups via {A|B|C} syntax

The top-level result of parsing is always a SequenceNode.

Public API:
  parse(template)   -> SequenceNode
  expand(node)      -> list[list[TextNode | ImageNode]]
  format_variant(v) -> str
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Union


@dataclass
class TextNode:
    """A literal text segment."""
    text: str

    def __repr__(self):
        excerpt = self.text[:40].replace('\n', '\\n')
        suffix = '...' if len(self.text) > 40 else ''
        return f"TextNode({excerpt!r}{suffix})"


@dataclass
class ImageNode:
    """An inline image reference from @path syntax (e.g. @photo.png)."""
    path: str

    def __repr__(self):
        return f"ImageNode({self.path!r})"


@dataclass
class VariantGroupNode:
    """A {A|B|C} variant expansion block.

    Each option is a SequenceNode representing one branch. During expansion
    each option produces a separate concrete prompt.
    """
    options: list[SequenceNode]

    def __repr__(self):
        return f"VariantGroupNode({len(self.options)} options)"


@dataclass
class SequenceNode:
    """An ordered sequence of prompt components.

    This is the root node type returned by the parser as well as the type
    used for each branch inside a VariantGroupNode.
    """
    children: list[PromptNode] = field(default_factory=list)

    def __repr__(self):
        return f"SequenceNode({self.children!r})"


# Union of all concrete node types
PromptNode = Union[TextNode, ImageNode, VariantGroupNode, SequenceNode]

# Type alias for a fully-expanded variant (no VariantGroupNode remains)
Variant = list[Union[TextNode, ImageNode]]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class _ConsumableString:
    """Character-by-character cursor over a string."""

    def __init__(self, s: str):
        self.s = s
        self.idx = 0

    def next(self) -> str | None:
        if self.idx >= len(self.s):
            return None
        c = self.s[self.idx]
        self.idx += 1
        return c

    def peek(self) -> str | None:
        if self.idx >= len(self.s):
            return None
        return self.s[self.idx]


def parse(template: str) -> SequenceNode:
    """Parse a prompt template string into a SequenceNode AST."""
    cs = _ConsumableString(template)
    result = _parse_sequence(cs, inside_variant=False)
    if cs.peek() is not None:
        raise ValueError(
            'Prompt parsing failed: Encountered unpaired }. '
            'Note that literal { or } characters must be escaped like \\{ or \\}')
    return result


def _parse_sequence(cs: _ConsumableString, *, inside_variant: bool) -> SequenceNode:
    """Parse a sequence of nodes.

    Stops (without consuming the delimiter) at | or } when inside_variant is
    True, allowing the caller to inspect which delimiter ended the sequence.
    """
    children: list[PromptNode] = []
    builder: list[str] = []

    while (c := cs.peek()) is not None:
        if inside_variant and c in ('|', '}'):
            break
        cs.next()

        if c == '\\':
            escaped = cs.next()
            if escaped is not None:
                builder.append(escaped)
        elif c == '{':
            if builder:
                children.append(TextNode(''.join(builder)))
                builder = []
            children.append(_parse_variant_group(cs))
        elif c == '}':
            # Reachable only when inside_variant=False; if inside_variant=True
            # we would have peeked and broken before consuming this character.
            raise ValueError(
                'Prompt parsing failed: Encountered unpaired }. '
                'Note that literal { or } characters must be escaped like \\{ or \\}')
        elif c == '@':
            if builder:
                children.append(TextNode(''.join(builder)))
                builder = []
            ref = _parse_ref(cs)
            if isinstance(ref, SequenceNode):
                # Inline included-file content directly into this sequence
                children.extend(ref.children)
            else:
                children.append(ref)
        else:
            builder.append(c)

    if builder:
        children.append(TextNode(''.join(builder)))

    return SequenceNode(children)


def _parse_variant_group(cs: _ConsumableString) -> VariantGroupNode:
    """Parse a {A|B|C} block. The opening { has already been consumed."""
    options: list[SequenceNode] = []
    while True:
        options.append(_parse_sequence(cs, inside_variant=True))
        delim = cs.next()
        if delim == '}':
            break
        elif delim == '|':
            continue
        else:
            raise ValueError(
                'Prompt parsing failed: Encountered unpaired {. '
                'Note that literal { or } characters must be escaped like \\{ or \\}')
    return VariantGroupNode(options)


def _parse_ref(cs: _ConsumableString) -> TextNode | ImageNode | SequenceNode:
    """Parse a @path reference. The @ has already been consumed.

    Returns:
      ImageNode  for recognised image extensions (.png, .jpg, .jpeg, .webp)
      SequenceNode for text files (recursively parsed)
    """
    builder: list[str] = []
    while cs.peek() is not None and not cs.peek().isspace():
        c = cs.next()
        if c == '\\':
            escaped = cs.next()
            if escaped is not None:
                builder.append(escaped)
        else:
            builder.append(c)
    path = ''.join(builder)
    _, ext = os.path.splitext(path)
    if ext.lower() in ('.png', '.jpg', '.jpeg', '.webp'):
        return ImageNode(path)
    with open(path, 'r') as f:
        return parse(f.read())


# ---------------------------------------------------------------------------
# Expander
# ---------------------------------------------------------------------------

def expand(node: SequenceNode) -> list[Variant]:
    """Expand all variant groups into concrete sequences.

    Returns one flat list of TextNode / ImageNode per concrete variant.
    A template with no variant groups yields a single-element list.
    """
    return _expand_sequence(node)


def _expand_sequence(node: SequenceNode) -> list[Variant]:
    variants: list[Variant] = [[]]
    for child in node.children:
        if isinstance(child, (TextNode, ImageNode)):
            for v in variants:
                v.append(child)
        elif isinstance(child, VariantGroupNode):
            new_variants: list[Variant] = []
            for v in variants:
                for option in child.options:
                    for opt_expansion in _expand_sequence(option):
                        new_variants.append(v + opt_expansion)
            variants = new_variants
        elif isinstance(child, SequenceNode):
            new_variants = []
            for v in variants:
                for nested in _expand_sequence(child):
                    new_variants.append(v + nested)
            variants = new_variants
    return variants


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def format_variant(variant: Variant) -> str:
    """Return a human-readable string representation of an expanded variant.

    TextNodes are rendered as their text; ImageNodes as @path.
    """
    parts = []
    for node in variant:
        if isinstance(node, TextNode):
            parts.append(node.text)
        elif isinstance(node, ImageNode):
            parts.append(f"@{node.path}")
    return ''.join(parts)
