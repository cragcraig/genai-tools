import argparse
import readline
import textwrap
import os
import io
import time
import uuid
from google import genai
from google.genai import types
from PIL import Image

MODEL_ID = "gemini-3-pro-image-preview"

def pil_image_to_png_bytes(img):
    """Helper to convert PIL Image to bytes for the API."""
    with io.BytesIO() as output:
        img.save(output, 'PNG')
        return output.getvalue()


class PILImageWrapper:
    def __init__(self, img):
        assert(img is not None)
        self.img = img

    def as_google_genai_types_image(self):
        return types.Part(
            inline_data=types.Blob(
                mime_type='image/png',
                data=pil_image_to_png_bytes(self.img)
            )
        )

    def as_pil_image(self):
        return self.img

    def save(self, path_or_bytes, format=None):
        return self.img.save(path_or_bytes, format)

    def show(self):
        return self.img.show()


class GoogleGenAITypesImageWrapper:
    """Wrapper for google.genai.types.Part representing an image part.

    Can be easily converted to a google.genai.types.Image by calling google.genai.types.Part.as_image()
    """
    def __init__(self, part):
        self.part = part
        self.img = part.as_image()
        assert(self.img is not None)

    def as_google_genai_types_image(self):
        return self.part

    def as_pil_image(self):
        return Image.open(io.Bytes(self.img.image_bytes))

    def save(self, path):
        return self.img.save(path)

    def show(self):
        return self.img.show()


class GenNode:
    """Image generation tree node"""
    def __init__(self, parent, prompt, response=None, ref_imgs=[], virtualroot=False, variations=None):
        self.level = parent.level + 1 if parent else 0
        self.turn = parent.turn + 1 if parent and not virtualroot else 0
        self.virtualroot = virtualroot
        self.local_id = len(parent.children) if parent else 0
        self.parent = parent
        self.children = []
        assert prompt is not None, 'Prompt must not be None'
        self.prompt = prompt
        self.ref_imgs = ref_imgs
        self.response = response
        self.variations = variations
        self.elapsed_sec = None

    @classmethod
    def new_root(cls, prompt, ref_imgs=[]):
        """Root nodes are special in that they are never executed.

        They're essentially just a placeholder for the prompt and ref_imgs which
        the top-level children of the root node will use without any additions"""
        return GenNode(None, prompt, ref_imgs=ref_imgs)

    def create_virtualroot(self, ref_imgs):
        if self.is_root():
            raise RuntimeError('Attempted to create a virtual root as a direct child of a root node')
        if not ref_imgs:
            raise ValueError('Must provide at least 1 reference image to initialize a virtual root')
        child = GenNode(self, '', ref_imgs=ref_imgs, virtualroot=True)
        self.children.append(child)
        return child

    def create_child(self, prompt, ref_imgs=[], variations=None):
        if self.response is None and not self.is_root():
            raise RuntimeError('Attempted to create child of non-root node that has no response {self.id()}')
        child = GenNode(self, prompt, ref_imgs=ref_imgs, variations=variations)
        self.children.append(child)
        return child

    def generate(self, client, image_config):
        if self.parent is None:
            print("Error: root node is a meta node for which content is never directly generated")
            return
        start_time = time.perf_counter()
        content = client.models.generate_content(
            model=MODEL_ID,
            contents=self.history(),
            config=self._gen_content_config(image_config)
        )
        elapsed_sec = time.perf_counter() - start_time
        self._set_response(content, elapsed_sec)
        return elapsed_sec

    def _gen_content_config(self, image_config):
        return types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE'],
            image_config=image_config,
            candidate_count=self.variations
        )

    def _set_response(self, response, elapsed_sec):
        if self.response is not None:
            raise RuntimeError('Attempted to overwrite response for node {self.id()}')
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
            role="user",
            parts=[types.Part(text=self.prompt)] + [img.as_google_genai_types_image() for img in self.ref_imgs]
        ))
        # Response content
        if self.response is not None:
            h.append(self.response.candidates[0].content)  # TODO: needs update if we ever actually support multiple candidates
        return h

    def title(self, prompt_length=72):
        if self.parent and self.parent.is_root() and self.prompt == self.parent.prompt:
            return f"{self.local_id}  [root -> generate]"
        id = 'BASE' if self.parent is None else self.local_id
        root = '[root]' if self.is_root() else ''
        imgs = f"({len(self.ref_imgs)} ref img{'s' if len(self.ref_imgs) > 1 else ''})" if self.ref_imgs else ''
        prompt = self.prompt[0:min(len(self.prompt), max(12, prompt_length - 4 * self.level))]
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
            print(f"+ {len(self.ref_imgs)} reference image{'s' if len(self.ref_imgs) > 1 else ''}")
        prompt = self.prompt if self.prompt else '[empty prompt]'
        print(textwrap.fill(prompt, width=80, initial_indent='| ', subsequent_indent='| '))

    def print_prompt_history(self):
        if self.parent:
            self.parent.print_prompt_history()
        print(textwrap.fill(self.prompt, width=80, initial_indent='  ', subsequent_indent='  '))
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
                print(prefix + indent + f"+ {len(self.children) - 1} other children")

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
        if self.response and self.response.candidates:
            for candidate in self.response.candidates:
                imgs += [GoogleGenAITypesImageWrapper(part) for part in candidate.content.parts if part.as_image()]
        return imgs

    def output(self, img_prefix='', hide=False, write=True):
        if not self.response:
            print("Error: No generate response exists")
            return False

        if not self.response.candidates:
            print("Error: No candidates returned")
            return False

        for i, candidate in enumerate(self.response.candidates):
            # Check why the generation stopped
            # "STOP" indicates successful completion.
            if candidate.finish_reason != "STOP":
                # Common failure reasons: "SAFETY", "RECITATION", "BLOCKLIST"
                print(f"\n[!] Generation failed for {self.id()}")
                print(f"    Finish Reason: {candidate.finish_reason}")
                # Print safety ratings if available to see which category tripped
                if candidate.safety_ratings:
                    print("    Safety Ratings:")
                    for rating in candidate.safety_ratings:
                        print(f"      - {rating.category}: {rating.probability}")
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
                if n == 0:
                    print("Error: Generation completed normally, but no image found in content for {self.id()}.")


def interactive_session(client, image_config, node):
    try:
        # Read an individual line from the user
        line = input(f"> ").strip()
        
        # Handle exit conditions
        if line.lower() in ['exit', 'quit']:
            print("Goodbye!")
            return None, False

        prompt = None
        ref_imgs = []
        variations = None
        if line:
            splits = line.split()
            cmd = splits[0].lower()

            # Show the content at the node
            if cmd == "show":
                node.output(hide=False, write=False)
                return node, False

            # Print prompt history for this node
            elif cmd == "context":
                print('')
                node.print_summary(full=True)
                return node, False

            # Switch to the parent node
            elif cmd == "up":
                if len(splits) == 2 and splits[1] in ['root', 'top']:
                    # up to next root or top root
                    return node.get_root(virtualroots=True if splits[1] == 'root' else False), False
                lvls = int(splits[1]) if len(splits) == 2 else 1
                if not lvls or lvls < 1 or len(splits) > 2:
                    print(f"Error: Unexpected parameter(s), expecting:  up [root | top | NUM_LEVELS]")
                    return node, False
                for i in range(0, lvls):
                    if node.parent is None:
                        print("Warning: halted early, reached the root of the tree")
                        return node, False
                    # up to parent
                    node = node.parent
                return node, False

            # Switch to a child node
            elif cmd == "down":
                if len(node.children) == 0:
                    print(f"Error: Can't go down from a leaf node")
                    return node, False
                idx = 0
                if len(node.children) == 1:
                    idx = 0
                elif len(splits) == 2:
                    idx = int(splits[1])
                else:
                    print(f"Missing argument, expected CHILD_INDEX to specify which of the {len(node.children)} children")
                    return node, False
                if idx >= len(node.children):
                    print(f"Error: Out of bounds index {idx}, node has {len(node.children)} children")
                    return node, False
                return node.children[idx], False

            # Create a new virtual root based on node
            elif cmd == "newroot":
                if node.is_root():
                    print('Error: already a root')
                    return node, False
                return node.create_virtualroot(node.all_imgs()), False

            # Print slice of the tree containing node
            elif cmd == "tree":
                filter = None if len(splits) < 2 else splits[1].lower()
                if filter and filter not in ['up', 'down']:
                    print("Warning: filter argument ignored, optional but if set must be one of: {up, down}")
                print('')
                node.print_tree(up=False if filter == 'down' else True, down=False if filter == 'up' else True)
                return node, False

            # Generate a new node
            elif cmd in ["generate", "sibling", "retry"]:
                if len(splits) == 2:
                    variations = int(splits[1])
                # Generate a sibling (new child of the parent)
                if cmd in ["sibling", "retry"]:
                    if node.is_root():
                        print("Error: can't create siblings for root nodes")
                        return node, False
                    else:
                        if cmd == "retry":
                            prompt = node.prompt
                            ref_imgs = node.ref_imgs
                        node = node.parent
                # Generate a child
                if node.is_root():
                    ref_imgs = node.ref_imgs
                    if node.prompt:
                        print("INFO: Starting a new top-level generate using root prompt")
                        prompt = node.prompt
                    elif ref_imgs:
                        print(f"INFO: Starting a new top-level generate using the provided image{'s' if len(ref_imgs) > 1 else ''}")
                        prompt = input(f"[start] Prompt >> ").strip()
                    else:
                        # Should never happen(tm)
                        raise ValueError('Invalid state: root node with no prompt and no reference images')
                elif not prompt:
                    prompt = input(f"[Turn {node.turn}] Prompt >> ").strip()
            else:
                # Input is not a recognized command
                if node.is_root() and node.prompt:
                    # Invalid command, but it's also not valid to use a generate prompt for this node
                    print("Warning: Invalid command, ignored. Try again.")
                else:
                    # Invalid command, but maybe it was intended as a generate prompt?
                    if node.is_root():
                        ref_imgs = node.ref_imgs
                    gencheck = input("Unrecognized as a command. Generate as a prompt? (y/n) [default: y]  ").strip().lower()
                    if gencheck in ['', 'y', 'yes']:
                        prompt = line
                    else:
                        print("ACK: Not a prompt, ignoring invalid command. Try again.")

            # Do the generate
            if prompt:
                assert(variations is None) # Multiple candidates is not supported by nano banana models
                child = node.create_child(prompt, ref_imgs=ref_imgs, variations=variations)
                print("Generating...  ", end='', flush=True)
                elapsed_sec = child.generate(client, image_config)
                print(f"{elapsed_sec:.1f} seconds")
                return child, True
            elif prompt == '':
                print("Generate canceled (empty prompt)")

            # TODO:
            #           - generate  support for seed, temperature, topn, topp, etc as parameters
            #           - help  list of commands
        
    except EOFError:
        # Handles Ctrl+D
        print("\nExiting...")
        return None, False
    except KeyboardInterrupt:
        # Handles Ctrl+C
        print("\nInterrupted by user.")
        return None, False
    return node, False

# Simple autocomplete function
def completer(text, state):
    options = [i for i in ['exit', 'quit', 'sibling', 'retry', 'generate', 'context', 'down', 'up', 'up root', 'up top', 'newroot', 'tree', 'tree up', 'tree down', 'show'] if i.startswith(text)]
    if state < len(options):
        return options[state]
    else:
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='nano-banana')
    parser.add_argument('--path', default=None, help='Directory to write output files, if not set a unique subdirectory will be made')
    parser.add_argument('--prefix', default='img')
    parser.add_argument('--resolution', choices=["1K", "2K", "4K"], default="1K")
    parser.add_argument('--aspect_ratio', choices=[None, "1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9"], default=None)
    parser.add_argument('--ref', nargs='*', default=[], help='Reference images (optional)')
    parser.add_argument('--noshow', action='store_true', help='Skip showing images when they\'re generated')
    parser.add_argument('--prompt', default='', help='Base prompt, optional if reference image(s) are provided')
    args = parser.parse_args()

    # Register the completer function and set the 'Tab' key for completion
    readline.set_completer(completer)
    readline.parse_and_bind("tab: complete")

    # Load any reference images now so that we can fail early if they're missing
    ref_imgs=[PILImageWrapper(Image.open(path)) for path in args.ref ]

    # Define the Gemini image config
    image_config = types.ImageConfig(
        aspect_ratio=args.aspect_ratio,
        image_size=args.resolution
    )

    # Prompt for base prompt, if necessary
    prompt = args.prompt
    if not args.prompt and not args.ref:
        print('Provide an initial prompt to generate')
        print('  example: Baby t-rex on a small pacific atoll. T-rex don\'t have feathers.,following in its mother\'s huge footprints,Studio Ghibli inspired,cinematic lighting\n')
        prompt = input(f"Prompt >> ").strip()
        if not prompt:
            print('Error: Empty prompt, exiting')
            exit(0);

    # Prep output path and ensure directory exists
    path = args.path
    if path is None:
        path = 'out-' + str(uuid.uuid4())[0:4]
        print(f"Output path set to:  {path}")
    if path:
        os.makedirs(path, exist_ok=True if args.path else False)
    img_prefix = os.path.join(path, args.prefix)

    # Gemini loop
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
            node, out = interactive_session(client, image_config, node)
            if out and node is not None:
                node.output(img_prefix, hide=args.noshow)
            print('')
            # Confirm exit
            if not node:
                confirm = input("Are you sure you want to exit? (y/n) [default: y]  ").strip().lower()
                if confirm not in ['', 'y', 'yes']:
                    # Cancel exit
                    node = prev_node
