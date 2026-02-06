import asyncio
import argparse
import readline
import textwrap
import os
import time
import uuid
from google import genai
from google.genai import types
from PIL import Image

import kittygraphics
import commands
from imagewrappers import PILImageWrapper, GoogleGenAITypesImageWrapper

# TODO:
#   - Record error nodes (and allow retries?)
#   - Support supplying reference images as arguments to generate-family commands

MODEL_ID = 'gemini-3-pro-image-preview'

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


class GenNode:
    """Node in the image generation conversation tree."""

    def __init__(self, parent, prompt, response=None, ref_imgs=[], virtualroot=False, seed=None):
        self.level = parent.level + 1 if parent else 0
        self.turn = parent.turn + 1 if parent and not virtualroot else 0
        self.virtualroot = virtualroot
        self.local_id = len(parent.children) if parent else 0
        self.parent = parent
        self.children = []
        assert prompt is not None, 'Prompt must not be None'
        self.prompt = prompt
        self.ref_imgs = ref_imgs
        self.seed = seed
        self.response = response
        self.elapsed_sec = None

    @classmethod
    def new_root(cls, prompt, ref_imgs=[]):
        """Root nodes are special in that they are never executed.

        They're essentially just a placeholder for the prompt and ref_imgs which
        the top-level children of the root node will use without any additions

        """
        return GenNode(None, prompt, ref_imgs=ref_imgs)

    def create_virtualroot(self, ref_imgs):
        if self.is_root():
            raise RuntimeError('Can\'t create a virtual root from a root node')
        if not ref_imgs:
            raise ValueError(
                'Must provide at least 1 reference image to initialize a virtual root')
        child = GenNode(self, '', ref_imgs=ref_imgs, virtualroot=True)
        self.children.append(child)
        return child

    def create_child(self, prompt, ref_imgs=[], seed=None):
        if self.response is None and not self.is_root():
            raise RuntimeError(
                'Attempted to create child of non-root node that has no response {self.id()}')
        child = GenNode(self, prompt, ref_imgs=ref_imgs, seed=None)
        self.children.append(child)
        return child

    async def generate(self, client, image_config):
        if self.parent is None:
            print(
                'Error: root node is a meta node for which content is never directly generated')
            return
        start_time = time.perf_counter()
        response = await client.aio.models.generate_content(
            model=MODEL_ID,
            contents=self.history(),
            config=self._gen_content_config(image_config),
        )
        elapsed_sec = time.perf_counter() - start_time
        self._set_response(response, elapsed_sec)
        return elapsed_sec

    def _gen_content_config(self, image_config):
        return types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE'],
            image_config=image_config,
            seed=self.seed,
        )

    def _set_response(self, response, elapsed_sec):
        # TODO: Check for a failed generate response (and record the failed state)
        if self.response is not None:
            raise RuntimeError(
                'Attempted to overwrite preexisting response recorded on node {self.id()}')
        self.response = response
        self.elapsed_sec = elapsed_sec

    def is_root(self):
        return self.parent is None or self.virtualroot

    def id(self, sep='/'):
        if self.parent is None:
            return ''
        return f"{self.parent.id(sep=sep)}{sep}{self.local_id}"

    def get_root(self, virtualroots=True):
        return self if (self.is_root() and virtualroots) or self.parent is None else self.parent.get_root(virtualroots=virtualroots)

    def history(self):
        if self.is_root():
            return []
        h = self.parent.history()
        # Prompt content
        h.append(types.Content(
            role='user',
            parts=[types.Part(text=self.prompt)] +
            [img.as_google_genai_types_part() for img in self.ref_imgs]
        ))
        # Response content
        if self.response is not None:
            # TODO: needs update if we ever actually support multiple candidates
            h.append(self.response.candidates[0].content)
        return h

    def title(self, prompt_length=72):
        if self.parent and self.parent.is_root() and self.prompt == self.parent.prompt:
            return f"{self.local_id}  [root -> generate]"
        id = 'BASE' if self.parent is None else self.local_id
        root = '[root]' if self.is_root() else ''
        imgs = f"({len(self.ref_imgs)} ref img{'s' if len(self.ref_imgs) > 1 else ''})" if self.ref_imgs else ''
        prompt = self.prompt[0:min(len(self.prompt), max(
            12, prompt_length - 4 * self.level))]
        return f"{id}  {root + ' ' if root else ''}{imgs + ' ' if imgs else ''}{prompt + ('...' if len(self.prompt) > len(prompt) else '')}"

    def print_summary(self, full=True):
        if full and not self.is_root() and not self.parent.is_root():
            self.parent.print_summary(full=True)
            print('|')
        id = ''
        if self.parent is None:
            id = 'Root /'
        elif self.is_root():
            id = f"VirtualRoot {self.id()}"
        else:
            id = f"Node {self.id()}"
        if not full:
            children = f"  (children = {len(self.children)})" if self.children else ''
            print(f"@{id}    turn = {self.turn}{children}")
        if self.ref_imgs:
            print(
                f"+ {len(self.ref_imgs)} reference image{'s' if len(self.ref_imgs) > 1 else ''}")
        prompt = self.prompt if self.prompt else '[empty prompt]'
        print(textwrap.fill(prompt, width=80,
              initial_indent='| ', subsequent_indent='| '))

    def print_prompt_history(self):
        if self.parent:
            self.parent.print_prompt_history()
        print(textwrap.fill(self.prompt, width=80,
              initial_indent='  ', subsequent_indent='  '))
        print('')

    def print_tree_node(self, prefix=''):
        indent = ' ' * (4 * self.level)
        print(prefix + indent + self.title())

    def print_ancestor_tree(self, include_self=True, prefix=''):
        if self.parent:
            self.parent.print_ancestor_tree(prefix=prefix)
        if include_self:
            self.print_tree_node(prefix=prefix)
            if len(self.children) > 1:
                indent = ' ' * (4 * (self.level + 1))
                print(prefix + indent +
                      f"+ {len(self.children) - 1} other children")

    def print_descendant_tree(self, include_self=True, prefix=''):
        if include_self:
            self.print_tree_node(prefix=prefix)
        for c in self.children:
            c.print_descendant_tree(prefix=prefix)

    def print_tree(self, up=True, down=True):
        if up:
            self.print_ancestor_tree(include_self=False, prefix='   ')
        self.print_tree_node(prefix='** ')
        if down:
            self.print_descendant_tree(include_self=False, prefix='   ')

    def all_imgs(self):
        imgs = []
        if self.ref_imgs:
            imgs += self.ref_imgs
        imgs += self.generated_imgs()
        return imgs

    def generated_imgs(self):
        imgs = []
        if self.response and self.response.candidates:
            for candidate in self.response.candidates:
                imgs += [GoogleGenAITypesImageWrapper(part)
                         for part in candidate.content.parts if part.as_image()]
        return imgs

    def output(self, img_prefix='', hide=False, write=True, inline=False):
        if not self.response:
            print('Error: No generate response exists')
            return False

        if not self.response.candidates:
            print('Error: No candidates returned')
            return False

        for i, candidate in enumerate(self.response.candidates):
            # Check why the generation stopped
            # "STOP" indicates successful completion.
            if candidate.finish_reason != 'STOP':
                # Common failure reasons: "SAFETY", "RECITATION", "BLOCKLIST"
                print(f"\n[!] Generation failed for {self.id()}")
                print(f"    Finish Reason: {candidate.finish_reason}")
                # Print safety ratings if available to see which category tripped
                if candidate.safety_ratings:
                    print('    Safety Ratings:')
                    for rating in candidate.safety_ratings:
                        print(
                            f"      - {rating.category}: {rating.probability}")
            else:
                # generation completed normally
                n = 0
                for part in candidate.content.parts:
                    if part.text is not None:
                        print(part.text)
                    elif image := part.as_image():
                        name = [img_prefix, self.id(sep='-')[1:]]
                        if i > 0:
                            name.append(f"c{i}")
                        if n > 0:
                            name.append(f"p{n}")
                        if write:
                            filename = f"{'_'.join(name)}.png"
                            image.save(filename)
                            print(f"Image saved as {filename}")
                        n += 1
                        if not hide:
                            image.show()
                        if inline:
                            pil_img = GoogleGenAITypesImageWrapper(part).as_pil_image()
                            w = min(600, pil_img.width)
                            pil_resized = pil_img.resize((w, int(pil_img.height / pil_img.width * w)))
                            kittygraphics.display_pil_image(pil_resized)
                if n == 0:
                    print(
                        'Error: Generation completed normally, but no image found in content for {self.id()}.')


def prompt_yn(prompt):
    try:
        return input(f"{prompt} (y/n) [default: y]  ").strip().lower() in ['', 'y', 'yes']
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
    parser.add_argument('--noshow', action='store_true',
                        help='Skip showing generate image results')
    parser.add_argument('--inline', action='store_true',
                        help='Show image thumbnails inline in terminal using the Kitty graphics protocol')
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
    kitty_graphics_enabled = args.inline and kittygraphics.supports_kitty_graphics()
    if args.inline and not kitty_graphics_enabled:
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
                    n.output(img_prefix, hide=args.noshow, inline=kitty_graphics_enabled)
            print('')
            # Confirm exit
            if not node and not prompt_yn('Confirm exit?'):
                # Cancel exit
                node = prev_node

if __name__ == '__main__':
    asyncio.run(main())
