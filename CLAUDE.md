# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Nano Banana** - An interactive Python REPL for iterative AI image generation using Google's Gemini API. Users build a conversation tree of prompts and generated images with branching/versioning capabilities, enabling exploration of image generation variations.

## Running the Application

```bash
python3 nano-banana.py [options]
```

Key options:
- `--path DIR` - Output directory (auto-generates `out-{uuid}` if not specified)
- `--resolution 1K|2K|4K` - Image resolution (default: 1K)
- `--aspect_ratio` - 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9
- `--img FILE` - Reference images to include (can repeat)
- `--inline` - Show thumbnails inline using Kitty protocol
- `--prompt TEXT` - Initial conversation prompt

## Dependencies

- `google-genai` - Google Generative AI SDK
- `pillow` - Image processing
- `google-auth` - Authentication

No formal package manager setup; install dependencies with pip.

## Architecture

### Core Components

**GenNode** (`nano-banana.py`): Tree node representing a conversation state. Tracks parent/children relationships, prompts, responses, and reference images. Key methods:
- `create_child()` / `create_virtualroot()` - Build conversation tree
- `generate()` - Call Gemini API with conversation history
- `history()` - Reconstruct prompt chain for API calls

**Prompt Template System**: Supports variant expansion syntax `{varA|varB|varC}` that expands into multiple concrete prompts for batch generation.

**kittygraphics.py**: Terminal graphics support via Kitty protocol for inline image display. Detects terminal capabilities via handshake.

### Interactive Commands

- Tree traversal: `tree`, `up`, `down`
- Node operations: `show`, `context`
- Generation: `generate`, `sibling`, `retry`, `root`

### API Integration

Uses `gemini-3-pro-image-preview` model with async/await pattern. Conversation history maintained through GenNode tree structure.

## Output

Generated images saved as `{prefix}_{node-id}.png` in the output directory. Batch generations use `_c{candidate}_p{part}` suffixes.
