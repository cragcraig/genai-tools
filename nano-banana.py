import asyncio
import argparse
import readline
import os
import uuid
from google import genai
from google.genai import types

import kittygraphics
import commands
import prompt_ast
from gennode import GenNode

# TODO:
#   - Record error nodes (and allow retries?)

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


async def interactive_session(client, image_config, model_id, node, confirm_at=None):
    """Process a single interactive command and return the new state."""
    ctx = commands.CommandContext(client=client, image_config=image_config, model_id=model_id, confirm_at=confirm_at)

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
                        help='Directory for output files')
    parser.add_argument('--prefix', default='img',
                        help='Filename prefix for output images')
    parser.add_argument(
        '--resolution', choices=['1K', '2K', '4K'], default='1K', help='Default 1K')
    parser.add_argument(
        '--model', choices=['flash', 'pro'], default='flash',
        help='Nano Banana model: flash (gemini-3.1-flash-image-preview) or pro (gemini-3-pro-image-preview); default: flash')
    parser.add_argument('--aspect_ratio', choices=[None, '1:1', '2:3', '3:2',
                        '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', '21:9'], default=None)
    parser.add_argument('--show', action='store_true',
                        help='Show image results after generating')
    parser.add_argument('--noinline', action='store_true',
                        help='Skip inline image thumbnails (Kitty graphics protocol)')
    parser.add_argument('--prompt', default='',
                        help='Initial prompt; use @path/to/image.png to inline images, e.g. "paint @ref.png in {watercolor|oil} style"')
    parser.add_argument('--confirm', default=None,
                        help='Threshold for simulatenous generates to require confirmation; default disabled')
    args = parser.parse_args()

    # Register the completer function and set the 'Tab' key for completion
    readline.set_completer(completer)
    readline.parse_and_bind('tab: complete')

    # Resolve model flag to full model ID
    model_id = {
        'flash': 'gemini-3.1-flash-image-preview',
        'pro':   'gemini-3-pro-image-preview',
    }[args.model]

    # Initialize the commands module with dependencies
    commands.init(HELP_TEXT, prompt_yn)

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
    if not prompt:
        print('Provide a prompt for initial context')
        print('  example: Baby t-rex with {faint scales|feathers} on a small pacific atoll. {Daytime|Moonless night with a brilliant aurora borealis}.,following its mother\'s huge indented footprints,Studio Ghibli inspired,cinematic lighting\n')
        print('  tip: use @path/to/image.png anywhere in the prompt to inline a reference image\n')
        try:
            prompt = input(f"Prompt >> ").strip()
            if not prompt:
                print('Error: Empty initial prompt, aborting...')
                return
        except EOFError:
            # Handles Ctrl+D
            print('\nExiting...')
            return
        except KeyboardInterrupt:
            # Handles Ctrl+C
            print('\nInterrupted by user.')
            return

    # Validate the prompt now so we fail fast on syntax errors or missing image files
    try:
        for variant in prompt_ast.expand(prompt_ast.parse(prompt)):
            for node in variant:
                if isinstance(node, prompt_ast.ImageNode) and not os.path.isfile(node.path):
                    print(f"Error: image file not found: {node.path}")
                    return
    except (ValueError, OSError) as e:
        print(f"Error in initial prompt: {e}")
        return

    # Prep output path and ensure directory exists
    path = args.path
    if path is None:
        path = 'out-' + str(uuid.uuid4())[0:4]
    if path:
        os.makedirs(path, exist_ok=True if args.path else False)
    img_prefix = os.path.join(path, args.prefix)

    # Gemini
    with genai.Client() as client:
        root = GenNode.new_root([prompt])
        node = root
        prev_node = None
        # Start the interactive command loop
        print(f"\n--- Interactive Nano Banana Shell (Crl+D to quit) ---")
        print(f"    Model: {model_id}")
        print(f"    Output path:  {path}" + os.path.sep + "\n")
        while node is not None:
            if node != prev_node:
                node.print_summary(full=False)
                prev_node = node
                print('')
            node, out = await interactive_session(client, image_config, model_id, node, confirm_at=args.confirm)
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
