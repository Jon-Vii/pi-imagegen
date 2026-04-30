# pi-imagegen

[![npm version](https://img.shields.io/npm/v/pi-imagegen.svg)](https://www.npmjs.com/package/pi-imagegen)
[![npm downloads](https://img.shields.io/npm/dm/pi-imagegen.svg)](https://www.npmjs.com/package/pi-imagegen)

A [Pi](https://pi.dev) package for generating images with your existing OpenAI/Codex subscription session.

`pi-imagegen` adds an agent-callable `imagegen` tool, a `/img` command namespace, and a local browser studio for visual image workflows.

## What it does

- Generates images through Pi's existing `openai-codex` OAuth login.
- Uses the Codex Responses backend with native `gpt-image-2` image generation.
- Saves images and sidecar metadata locally.
- Supports batches, style presets, reference images, and sketch references.
- Provides a browser-based studio for browsing, comparing, rerunning, varying, and referencing images.

## Install

Install the published npm package:

```bash
pi install npm:pi-imagegen
```

npm package: <https://www.npmjs.com/package/pi-imagegen>

For local development:

```bash
git clone https://github.com/Jon-Vii/pi-imagegen.git
cd pi-imagegen
pi install .
```

Then reload Pi:

```txt
/reload
```

You also need to be logged into Pi's OpenAI/Codex provider:

```txt
/login
```

Select the ChatGPT/Codex option that provides the `openai-codex` provider.

## Commands

```txt
/img studio
/img gen [--thinking off|minimal|low|medium|high] [--style name] <prompt>
/img batch <count> [--thinking off|minimal|low|medium|high] [--style name] <prompt>
/img styles
/img list [count]
/img open [latest|number|path]
/img reveal [latest|number|path]
/img path [latest|number|path]
/img info [latest|number|path]
```

Examples:

```txt
/img gen tiny blue ceramic fish on white background
/img gen --thinking off --style poster a cinematic expedition poster for a lava cavern
/img batch 4 --style wallpaper a quiet mountain observatory at sunrise
/img studio
```

## Studio

Run:

```txt
/img studio
```

The studio opens a local browser UI served from `127.0.0.1`.

It supports:

- image history wall
- grouped batch/contact-sheet view
- modal preview with rerun/vary/reference actions
- prompt composer with style, aspect, quality, thinking, and count controls
- real image references sent as `input_image` content
- sketch references via a simple drawing canvas

A typical loop:

```txt
Draw or pick reference → Generate 4 → inspect → Vary or Rerun → keep exploring
```

## Agent tool

The package also registers a model-facing tool:

```txt
imagegen
```

It can generate an image and return both a saved file and an inline image attachment. It supports options such as:

- `prompt`
- `size`
- `quality`
- `background`
- `outputFormat`
- `thinking`
- `referencePaths`
- `outputPath`

## How it works

This package does **not** use the public OpenAI image API or a separate API key.

It uses Pi's existing `openai-codex` OAuth token and calls:

```txt
https://chatgpt.com/backend-api/codex/responses
```

with the native Responses image generation tool:

```json
{
  "type": "image_generation",
  "model": "gpt-image-2"
}
```

Generated image results are received from streamed SSE events and saved locally.

## Files and metadata

By default, generated images are saved under:

```txt
~/.pi/agent/generated-images/
```

Each image gets a JSON sidecar with prompt, model, path, format, reference, batch, and generation metadata.

Batches are saved under:

```txt
~/.pi/agent/generated-images/batches/
```

Sketch references are saved under:

```txt
~/.pi/agent/generated-images/sketches/
```

## Notes

This package relies on Pi internals and the Codex Responses backend. It is intended for personal/local Pi workflows and may need updates if the upstream backend changes.

## License

MIT
