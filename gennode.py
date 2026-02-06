"""GenNode class for the Nano Banana conversation tree."""

import textwrap
import time
from google.genai import types

import kittygraphics
from imagewrappers import GoogleGenAITypesImageWrapper

MODEL_ID = 'gemini-3-pro-image-preview'


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
