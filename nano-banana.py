import argparse
import readline
import textwrap
import os
from google import genai
from google.genai import types
from PIL import Image

MODEL_ID = "gemini-3-pro-image-preview"

class GenNode:
    """Image generation history node"""
    def __init__(self, parent, prompt, response=None, ref_imgs=[], variations=None):
        self.level = parent.level + 1 if parent else 0
        self.local_id = len(parent.children) if parent else 0
        self.parent = parent
        self.children = []
        self.prompt = prompt
        self.ref_imgs = ref_imgs
        self.response = response
        self.variations = variations

    @classmethod
    def new_root(cls, prompt, ref_imgs=[]):
        """Root nodes are special in that they are never executed.

        They're essentially just a placeholder for the prompt and ref_imgs which
        the top-level children of the root node will use without any additions"""
        return GenNode(None, prompt, ref_imgs=[])

    def create_child(self, prompt, ref_imgs=[], variations=None):
        if self.response is None and self.parent:
            raise RuntimeError('Attempted to create child of non-root node that has no response {self.id()}')
        child = GenNode(self, prompt, ref_imgs=ref_imgs, variations=variations)
        self.children.append(child)
        return child

    def generate(self, client, image_config):
        if self.is_root():
            print("Error: root node is a special case for which content is never directly generated")
            return
        self._set_response(client.models.generate_content(
                model=MODEL_ID,
                contents=self.history(),
                config=self._gen_content_config(image_config)
            )
        )

    def _gen_content_config(self, image_config):
        return types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE'],
            image_config=image_config,
            candidate_count=self.variations
        )

    def _set_response(self, response):
        if self.response is not None:
            raise RuntimeError('Attempted to overwrite response for node {self.id()}')
        self.response = response

    def is_root(self):
        return self.parent is None

    def id(self):
        if self.is_root():
            return ''
        return f"{self.parent.id()}/{self.local_id}"

    def get_root(self):
        return self if self.is_root() else self.parent.get_root()

    def history(self):
        if self.is_root():
            return []
        h = self.parent.history()
        # Prompt
        h.append(types.Content(
            role="user",
            parts=[types.Part(text=self.prompt)] + [types.Part.from_image(img) for img in self.ref_imgs]
        ))
        # Response
        if self.response is not None:
            h.append(self.response.candidates[0].content)
        return h

    def title(self, prompt_length=65):
        if self.parent and self.parent.is_root():
            return f"[{self.local_id}] Top-level node #{self.local_id}"
        return f"[{'root' if self.is_root() else self.local_id}] {self.prompt[0:min(len(self.prompt), max(12, prompt_length - 4 * self.level))]}"

    def print_summary(self, short=True):
        print(f"@Node id = {'/' if self.is_root() else self.id()}, turn = {self.level}, variations = {self.variations}, children = {len(self.children)}")
        if short:
            print(textwrap.fill(self.prompt, width=80, initial_indent='  ', subsequent_indent='  '))
        else:
            self.print_prompt_history()

    def print_prompt_history(self):
        if self.parent:
            self.parent.print_prompt_history()
        print(textwrap.fill(self.prompt, width=80, initial_indent='  ', subsequent_indent='  '))
        print('')

    def print_tree_node(self, prefix=' '):
        indent = ' ' * (4 * self.level)
        print(prefix + indent + self.title())

    def print_ancestor_tree(self, include_self=True):
        if not self.is_root():
            self.parent.print_ancestor_tree()
        if include_self:
            self.print_tree_node()

    def print_descendant_tree(self, include_self=True):
        if include_self:
            self.print_tree_node()
        for c in self.children:
            c.print_descendant_tree()

    def print_tree(self):
        self.print_ancestor_tree(include_self=False)
        self.print_tree_node(prefix='*')
        self.print_descendant_tree(include_self=False)


    def output(self, img_prefix='', show=True, write=True):
        if not self.response:
            print("Error: Prompt has not been executed")
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
                        name = [img_prefix]
                        if i > 0:
                            name.append(f"c{i}")
                        if n > 0:
                            name.append(f"i{n}")
                        if write:
                            image.save(f"{'-'.join(name)}.png")
                        n += 1
                        if show:
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

        if line:
            splits = line.split()
            cmd = splits[0].lower()

            # Show the content at the node
            if cmd == "show":
                node.output(show=True, write=False)
                return node, False

            # Print prompt history for this node
            if cmd == "history":
                print('')
                node.print_summary(short=True)
                print('')
                return node, False

            # Switch to the parent node
            if cmd == "up":
                if node.is_root():
                    print("Error: can't go higher than root")
                    return node, False
                return node.parent, False

            # Switch to a child node
            if cmd == "down":
                if len(splits) < 2:
                    print("Missing argument, expected CHILD_INDEX")
                    return node, False
                idx = int(splits[1])
                if idx >= len(node.children):
                    print(f"Error: Out of bounds index {idx}, node has {len(node.children)} children")
                    return node, False
                return node.children[idx], False

            # Switch to root node
            if cmd == "root":
                return node.get_root(), False

            # Print slice of the tree containing node
            if cmd == "tree":
                node.print_tree()
                return node, False

            # Generate a child
            if cmd in ["generate", "sibling"]:
                variations = None if len(splits) < 2 else int(splits[1])
                # Generate a sibling (new child of the parent)
                if cmd == "sibling":
                    if node.is_root():
                        print("Error: can't create siblings for root")
                        return node, False
                    else:
                        node = node.parent
                prompt = None
                ref_imgs = []
                # Generate a child
                if node.is_root():
                    print("NOTE: This is a new top-level generate using only the original prompt")
                    prompt = node.prompt
                    ref_imgs = node.ref_imgs
                else:
                    prompt = input(f"[Turn {node.level}] Prompt >> ").strip()

                # Do the generate
                if prompt:
                    assert(variations is None) # Multiple candidates is not supported by nano banana models
                    child = node.create_child(prompt, ref_imgs=ref_imgs, variations=variations)
                    print("Generating...")
                    child.generate(client, image_config)
                    return child, True
                else:
                    print("Generate canceled (empty prompt)")
        
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
    options = [i for i in ['exit', 'quit', 'sibling', 'generate', 'history', 'down', 'up', 'root', 'tree', 'show'] if i.startswith(text)]
    if state < len(options):
        return options[state]
    else:
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='nano-banana')
    parser.add_argument('--path', default='out')
    parser.add_argument('--prefix', default='img')
    parser.add_argument('--resolution', choices=["1K", "2K", "4K"], default="2K")
    parser.add_argument('--aspect_ratio', choices=["1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9"], default="16:9")
    parser.add_argument('--ref', nargs='*', default=[])
    parser.add_argument('prompt')
    args = parser.parse_args()

    # Register the completer function and set the 'Tab' key for completion
    readline.set_completer(completer)
    readline.parse_and_bind("tab: complete")

    # Load any reference images so that we can fail early if they're missing
    ref_imgs=[ Image.open(path) for path in args.ref ]

    # Define the image config
    image_config = types.ImageConfig(
        aspect_ratio=args.aspect_ratio,
        image_size=args.resolution
    )

    # Prep output path
    if args.path:
        os.makedirs(args.path, exist_ok=True)
    img_prefix = os.path.join(args.path, args.prefix)

    # Gemini
    with genai.Client() as client:
        root = GenNode.new_root(args.prompt, ref_imgs=ref_imgs)
        # Always start by generating one child of the root node
        print("Generating initial image...")
        node = root.create_child(root.prompt, ref_imgs=root.ref_imgs)
        node.generate(client, image_config)
        node.output(img_prefix)
        # Then switch over to the interactive prompt
        print("\n--- Interactive Nano Banana Shell (type 'exit' or Ctrl+D to quit) ---\n")
        while node is not None:
            node.print_summary(short=True)
            print('')
            node, out = interactive_session(client, image_config, node)
            if out and node is not None:
                node.output(img_prefix)
            print('')
