import asyncio
import argparse
import readline
import os
import uuid
from google import genai
from google.genai import types
from PIL import Image

import kittygraphics
import commands
from imagewrappers import PILImageWrapper
from gennode import GenNode

# TODO:
#   - Record error nodes (and allow retries?)
#   - Support supplying reference images as arguments to generate-family commands

HELP_TEXT = """
tl;dr:

    An interactive REPL loop for iterative image generation with Nano Banana.

How It Works:

    Construct a directed tree of sequential Prompt -> Generated Image turns as
    you iteratively explore and refine branching chains of generated images.

    The resulting tree is essentially an undo/redo history supporting arbitrary
    resumption from any decision point in the conversation. Each branch in the
    tree represents an alternate timeline of the conversation.

    Generating a new image is represented as a new child node in the tree. The
    context used for generating this image will be the coversation history and
    images since the most recent root node, constructed by following the
    conversation path along the chain of nodes leading from that last root node
    to this new child node.

    Commands support arbitrarily traversing the conversation tree and branching
    off new child nodes (aka kicking off new image generation) from any point in
    the conversation.

Command Glossary:

    Tree Traversal
        - tree [up|down|top|root]
            Display the slice of the tree containing the active node.

            Hint: To view the full tree, move to the top root node via 'up top'
                  before running the 'tree' command.

        - up [top|root|LEVELS]
            Move to the parent (default) or other ancestor node.

        - down [CHILD_INDEX]
            Move to a child node.

    Current Node
        - show
            Display any content generated at the active node.

        - context
            Display the full context used to generate content at the active node.

    Generate Images
        - generate [VARIATIONS]
          (or simply enter text that doesn't match to any command)
            Generate a child of the active node.

        - sibling [VARIATIONS]
            Generate a new sibling (shared parent) of the active node.

        - retry [VARIATIONS]
            Generate a new sibling (shared parent) using the exact same context
            used to generate the active node.

        - root
            Create a copy of the current node and mark it as a new conversation
            root. Descendents of this new root node will seed their context
            starting from the generated image at this node.

    General
        - help / ?
        - exit / quit"""


def prompt_yn(prompt):
    try:
        return input(f"{prompt} (Y/n)  ").strip().lower() in ['', 'y', 'yes']
    except EOFError:
        return False
    except KeyboardInterrupt:
        return False


def parse_prompt(template_prompt, outer=True):
    """Return list of all concrete prompt variants described by the template
    prompt,

    i.e., expand all {varA|varB|varC} blocks.

    """
    variants = [[]]
    parts = []
    builder = []
    level = 0
    escape_next = False
    for c in template_prompt:
        if escape_next:
            builder.append(c)
            escape_next = False
        elif c == '\\':
            escape_next = True
        elif c == '{':
            level += 1
            if level > 1:
                raise ValueError(
                    'Prompt variant templates don\'t (yet) support nested variant blocks')
            chunk = ''.join(builder)
            for v in variants:
                v.append(chunk)
            builder = []
        elif c == '}':
            level -= 1
            if level < 0:
                raise ValueError(
                    'Prompt parsing failed: Encountered unescaped and unpaired }. Note that literal { or } characters must be escaped like \\{ or \\}')
            # first finish the part
            parts.append(''.join(builder))
            builder = []
            # then construct updated set of (partial) variants
            old_variants = variants
            variants = []
            for v in old_variants:
                for p in parts:
                    variants.append(v + [p])
            parts = []
        elif c == '|':
            if level == 0 and outer:
                # don't treat | as a special character outside of {} blocks
                builder.append(c)
            else:
                # | separates variants in a {} block
                parts.append(''.join(builder))
                builder = []
        else:
            # All non-special characters are simply appended to the current chunk
            builder.append(c)
    if level != 0:
        raise ValueError(
            'Prompt parsing failed: Encountered unescaped and unpaired {. Note that literal { or } characters must be escaped like \\{ or \\}')
    chunk = ''.join(builder)
    for v in variants:
        v.append(chunk)
    builder = []
    return [''.join(v) for v in variants]

async def interactive_session(client, image_config, node, confirm_at=None):
    """Process a single interactive command and return the new state."""
    ctx = commands.CommandContext(client=client, image_config=image_config, confirm_at=confirm_at)

    try:
        line = input("> ").strip()
        if not line:
            return node, None

        splits = line.split()
        cmd = splits[0].lower()
        args = splits[1:]

        if cmd in commands.COMMANDS:
            result = await commands.COMMANDS[cmd](args, node, ctx)
        else:
            result = await commands.cmd_generate_from_text(line, node, ctx)

        return result.node, result.new_children

    except EOFError:
        print('\nExiting...')
        return None, None
    except KeyboardInterrupt:
        print('\nInterrupted by user.')
        return None, None

# Simple autocomplete function
def completer(text, state):
    options = [i for i in ['help', '?', 'exit', 'quit', 'sibling', 'retry', 'generate', 'context', 'down', 'up',
                           'up root', 'up top', 'root', 'tree', 'tree up', 'tree down', 'tree root', 'tree top', 'show'] if i.startswith(text)]
    if state < len(options):
        return options[state]
    else:
        return None


async def main():
    parser = argparse.ArgumentParser(prog='nano-banana')
    parser.add_argument('--path', default=None,
                        help='Directory for all output files; default creates a unique subdirectory')
    parser.add_argument('--prefix', default='img',
                        help='Filename prefix for all output images')
    parser.add_argument(
        '--resolution', choices=['1K', '2K', '4K'], default='1K', help='Defaults to 1K')
    parser.add_argument('--aspect_ratio', choices=[None, '1:1', '2:3', '3:2',
                        '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', '21:9'], default=None)
    parser.add_argument('--img', nargs='*', default=[],
                        help='(optional) Reference images to include in initial context')
    parser.add_argument('--show', action='store_true',
                        help='Show generate image results')
    parser.add_argument('--noinline', action='store_true',
                        help='Don\'t show image thumbnails inline in terminal using the Kitty graphics protocol')
    parser.add_argument('--prompt', default='',
                        help='Initial conversation context; optional if reference image(s) are provided with --img')
    parser.add_argument('--confirm-at', default=None,
                        help='Threshold of simulatenous generate image calls that will ask for confirmation; default disables any limit')
    args = parser.parse_args()

    # Register the completer function and set the 'Tab' key for completion
    readline.set_completer(completer)
    readline.parse_and_bind('tab: complete')

    # Initialize the commands module with dependencies
    commands.init(HELP_TEXT, parse_prompt, prompt_yn)

    # Load any reference images now so that we fail early if they're missing
    ref_imgs = [PILImageWrapper(Image.open(path)) for path in args.img]

    # Define the Gemini image config
    image_config = types.ImageConfig(
        aspect_ratio=args.aspect_ratio,
        image_size=args.resolution
    )

    # Check for Kitty graphics protocol support
    kitty_graphics_enabled = (not args.noinline) and kittygraphics.supports_kitty_graphics()
    if (not args.noinline) and not kitty_graphics_enabled:
        print('WARNING: Requested inline graphics, but Kitty graphics protocol is not supported by terminal')

    # Prompt for base prompt, if necessary
    prompt = args.prompt
    if not args.prompt and not args.img:
        print('Provide a prompt for initial context')
        print('  example: Baby t-rex with {faint scales|feathers} on a small pacific atoll. {Daytime|Moonless night with a brilliant aurora borealis}.,following its mother\'s huge indented footprints,Studio Ghibli inspired,cinematic lighting\n')
        try:
            prompt = input(f"Prompt >> ").strip()
            if not prompt:
                print(
                    'Error: Empty context (no initial prompt nor any reference images), aborting...')
                return
        except EOFError:
            # Handles Ctrl+D
            print('\nExiting...')
            return
        except KeyboardInterrupt:
            # Handles Ctrl+C
            print('\nInterrupted by user.')
            return

    # Prep output path and ensure directory exists
    path = args.path
    if path is None:
        path = 'out-' + str(uuid.uuid4())[0:4]
        print(f"Output path set to:  {path}")
    if path:
        os.makedirs(path, exist_ok=True if args.path else False)
    img_prefix = os.path.join(path, args.prefix)

    # Gemini
    with genai.Client() as client:
        root = GenNode.new_root(prompt, ref_imgs=ref_imgs)
        node = root
        prev_node = None
        # Start the interactive command loop
        print("\n--- Interactive Nano Banana Shell (type 'exit' or Ctrl+D to quit) ---\n")
        while node is not None:
            if node != prev_node:
                node.print_summary(full=False)
                prev_node = node
                print('')
            node, out = await interactive_session(client, image_config, node, confirm_at=args.confirm_at)
            if out:
                # Output all newly generated nodes
                for n in out:
                    n.output(img_prefix, hide=not args.show, inline=kitty_graphics_enabled)
            print('')
            # Confirm exit
            if not node and not prompt_yn('Confirm exit?'):
                # Cancel exit
                node = prev_node

if __name__ == '__main__':
    asyncio.run(main())
