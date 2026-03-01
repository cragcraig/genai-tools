"""Command handlers for the Nano Banana interactive REPL."""

import asyncio
from dataclasses import dataclass
from typing import Optional, List, Any, TYPE_CHECKING
from PIL import Image

import prompt_ast
from imagewrappers import PILImageWrapper

if TYPE_CHECKING:
    from gennode import GenNode


@dataclass
class CommandContext:
    """Context passed to all command handlers."""
    client: Any
    image_config: Any
    confirm_at: Optional[int] = None


@dataclass
class CommandResult:
    """Result returned by command handlers."""
    node: Optional['GenNode']  # New active node (None = exit requested)
    new_children: Optional[List['GenNode']] = None  # Newly generated nodes


# These will be set by the main module to avoid circular imports
HELP_TEXT: str = ''
prompt_yn = None


def init(help_text: str, prompt_yn_fn):
    """Initialize module dependencies from main module."""
    global HELP_TEXT, prompt_yn
    HELP_TEXT = help_text
    prompt_yn = prompt_yn_fn


# --- Command Handlers ---
# All commands follow the signature:
#   async def cmd_name(args: List[str], node: GenNode, ctx: CommandContext) -> CommandResult


async def cmd_help(args: List[str], node: 'GenNode', ctx: CommandContext) -> CommandResult:
    """Display help text."""
    print(HELP_TEXT)
    return CommandResult(node=node)


async def cmd_exit(args: List[str], node: 'GenNode', ctx: CommandContext) -> CommandResult:
    """Exit the interactive session."""
    print('Goodbye!')
    return CommandResult(node=None)


async def cmd_show(args: List[str], node: 'GenNode', ctx: CommandContext) -> CommandResult:
    """Display content generated at the active node."""
    node.output(hide=False, write=False)
    return CommandResult(node=node)


async def cmd_context(args: List[str], node: 'GenNode', ctx: CommandContext) -> CommandResult:
    """Display the full context/history for the active node."""
    print('')
    node.print_summary(full=True)
    return CommandResult(node=node)


async def cmd_up(args: List[str], node: 'GenNode', ctx: CommandContext) -> CommandResult:
    """Navigate to parent or ancestor node."""
    if len(args) == 1 and args[0] in ['root', 'top']:
        return CommandResult(node=node.get_root(virtualroots=args[0] == 'root'))

    lvls = int(args[0]) if len(args) == 1 else 1
    if not lvls or lvls < 1 or len(args) > 1:
        print('Error: Unexpected parameter(s), expecting:  up [{top|root|LEVELS}]')
        return CommandResult(node=node)

    for _ in range(lvls):
        if node.parent is None:
            print('Warning: halted early, reached the root of the tree')
            return CommandResult(node=node)
        node = node.parent
    return CommandResult(node=node)


async def cmd_down(args: List[str], node: 'GenNode', ctx: CommandContext) -> CommandResult:
    """Navigate to a child node."""
    if len(node.children) == 0:
        print("Error: Can't go down from a leaf node")
        return CommandResult(node=node)

    if len(node.children) == 1:
        idx = 0
    elif len(args) == 1:
        idx = int(args[0])
    else:
        print(f"Missing argument, expected CHILD_INDEX to specify which of the {len(node.children)} children")
        return CommandResult(node=node)

    if idx >= len(node.children):
        print(f"Error: Out of bounds index {idx}, node has {len(node.children)} children")
        return CommandResult(node=node)

    return CommandResult(node=node.children[idx])


async def cmd_root(args: List[str], node: 'GenNode', ctx: CommandContext) -> CommandResult:
    """Create a new virtual root based on current node."""
    if node.is_root():
        print('Error: already a root')
        return CommandResult(node=node)
    return CommandResult(node=node.create_virtualroot(node.generated_imgs()))


async def cmd_tree(args: List[str], node: 'GenNode', ctx: CommandContext) -> CommandResult:
    """Display the tree structure around the active node."""
    if len(args) > 1:
        print('Error: too many arguments')
        return CommandResult(node=node)

    filter_arg = args[0].lower() if args else None

    if not filter_arg or filter_arg in ['up', 'down']:
        print('')
        node.print_tree(
            up=filter_arg != 'down',
            down=filter_arg != 'up'
        )
    elif filter_arg in ['root', 'top']:
        print('')
        node.get_root(virtualroots=filter_arg == 'root').print_tree(up=False, down=True)
    else:
        print('Error: Unsupported value for FILTER argument, optional but if set must be one of: {up, down, top, root}')

    return CommandResult(node=node)


async def cmd_generate(args: List[str], node: 'GenNode', ctx: CommandContext,
                       mode: str = 'generate') -> CommandResult:
    """Generate new image(s) as child, sibling, or retry.

    mode: 'generate' (child), 'sibling' (sibling with new prompt), 'retry' (sibling with same parts)
    """
    if len(args) > 1:
        print('Error: too many arguments')
        return CommandResult(node=node)

    variations = 1
    if len(args) == 1:
        variations = int(args[0])
        if variations <= 0:
            print(f"Error: Unsupported value `{variations}` for VARIATIONS argument, optional but if set must be a positive integer")
            return CommandResult(node=node)

    # Handle sibling/retry: move to parent first
    if mode in ['sibling', 'retry']:
        if node.is_root():
            print("Error: can't create siblings for root nodes")
            return CommandResult(node=node)
        retry_parts = node.parts if mode == 'retry' else None
        node = node.parent

    # For retry: reuse the original parts exactly, bypass template parsing
    if mode == 'retry':
        return await _run_generation([retry_parts], variations, node, ctx)

    # Determine prompt template and context images
    prompt = None
    context_imgs = []

    if node.is_root():
        context_imgs = node._images()
        if node._text():
            print('INFO: Starting a new top-level generate using root prompt')
            prompt = node._text()
        elif context_imgs:
            print(f"INFO: Starting a new top-level generate using the provided image{'s' if len(context_imgs) > 1 else ''}")
            prompt = input(f"[Turn 0] Prompt >> ").strip()
        else:
            raise ValueError('Invalid state: root node should not exist with no prompt and no reference images')
    else:
        prompt = input(f"[Turn {node.turn}] Prompt >> ").strip()

    if not prompt:
        print('Generate canceled (empty prompt)')
        return CommandResult(node=node)

    return await _execute_generation(prompt, context_imgs, variations, node, ctx)


async def _execute_generation(prompt: str, context_imgs: list, variations: int,
                               node: 'GenNode', ctx: CommandContext) -> CommandResult:
    """Parse a prompt template, build parts lists, and generate."""
    try:
        variants = prompt_ast.expand(prompt_ast.parse(prompt))
    except (ValueError, OSError) as e:
        print(e)
        print('Aborting generate, try again')
        return CommandResult(node=node)

    # Print summary of variants (shows @paths from the AST before image loading)
    if len(variants) > 1:
        print('\nPrompt variants:')
        for i, v in enumerate(variants):
            print(f" {i}  {prompt_ast.format_variant(v)}")

    # Build parts lists: convert each AST variant to [str | ImageWrapper],
    # then append context images at the end (matching current text-first ordering)
    parts_list = []
    for variant in variants:
        parts = []
        for n in variant:
            if isinstance(n, prompt_ast.TextNode):
                parts.append(n.text)
            elif isinstance(n, prompt_ast.ImageNode):
                parts.append(PILImageWrapper(Image.open(n.path)))
        parts.extend(context_imgs)
        parts_list.append(parts)

    return await _run_generation(parts_list, variations, node, ctx)


async def _run_generation(parts_list: list, variations: int,
                           node: 'GenNode', ctx: CommandContext) -> CommandResult:
    """Core generation loop: confirm, create child nodes, gather tasks."""
    variations_text = f" {variations} variations{' of each prompt variant' if len(parts_list) > 1 else ''}" if variations > 1 else ''

    gen_count = len(parts_list) * variations
    if ctx.confirm_at is not None and gen_count >= ctx.confirm_at:
        if not prompt_yn(f"Confirm approval for generating {gen_count} images?"):
            print('Aborted before generating images')
            return CommandResult(node=node)

    print(f"Generating{variations_text}...  ", end='', flush=True)
    new_children = []
    for parts in parts_list:
        new_children.extend([
            node.create_child(parts, seed=None if i == 0 else i)
            for i in range(variations)
        ])

    tasks = [child.generate(ctx.client, ctx.image_config) for child in new_children]
    elapsed = await asyncio.gather(*tasks)
    print(', '.join([f"{sec:.1f}" for sec in elapsed]) + ' seconds')

    result_node = new_children[0] if len(new_children) == 1 else node
    return CommandResult(node=result_node, new_children=new_children)


async def cmd_generate_from_text(text: str, node: 'GenNode', ctx: CommandContext) -> CommandResult:
    """Handle unrecognized input as a generation prompt."""
    if node.is_root() and node._text():
        print("Warning: Unrecognized as a command but also can't use as a prompt due to being at a root node with a prompt already. Try just `generate`.")
        return CommandResult(node=node)

    context_imgs = node._images() if node.is_root() else []
    if not prompt_yn('Unrecognized as a command. Generate as a prompt?'):
        print('ACK: Not a prompt, ignoring invalid command. Try again.')
        return CommandResult(node=node)

    return await _execute_generation(text, context_imgs, 1, node, ctx)


# Command registry: maps command names to handlers
COMMANDS = {
    'help': cmd_help,
    '?': cmd_help,
    'exit': cmd_exit,
    'quit': cmd_exit,
    'show': cmd_show,
    'context': cmd_context,
    'up': cmd_up,
    'down': cmd_down,
    'root': cmd_root,
    'tree': cmd_tree,
    'generate': cmd_generate,
    'sibling': lambda args, node, ctx: cmd_generate(args, node, ctx, mode='sibling'),
    'retry': lambda args, node, ctx: cmd_generate(args, node, ctx, mode='retry'),
}
