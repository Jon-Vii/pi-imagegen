/**
 * OpenAI Codex Image Generation for pi
 *
 * Registers `imagegen`, a custom tool that uses pi's existing openai-codex
 * OAuth credentials to call the Codex Responses backend with the native
 * `image_generation` tool (`gpt-image-2`).
 */

import { Buffer } from "node:buffer";
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, readdir, readFile, stat, writeFile } from "node:fs/promises";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { basename, dirname, extname, join, resolve } from "node:path";
import { StringEnum } from "@mariozechner/pi-ai";
import { type ExtensionAPI, type ExtensionContext, getAgentDir, withFileMutationQueue } from "@mariozechner/pi-coding-agent";
import { Text } from "@mariozechner/pi-tui";
import { type Static, Type } from "typebox";

const PROVIDER = "openai-codex";
const CODEX_BASE_URL = "https://chatgpt.com/backend-api";
const DEFAULT_RESPONSE_MODEL = "gpt-5.5";
const IMAGE_MODEL = "gpt-image-2";

const SIZES = ["auto", "1024x1024", "1536x1024", "1024x1536"] as const;
const QUALITIES = ["auto", "low", "medium", "high"] as const;
const BACKGROUNDS = ["auto", "opaque", "transparent"] as const;
const OUTPUT_FORMATS = ["png", "webp", "jpeg"] as const;
const THINKING_MODES = ["off", "minimal", "low", "medium", "high"] as const;

const STYLE_PRESETS: Record<string, Partial<ToolParams> & { suffix: string }> = {
	"minecraft-screenshot": {
		size: "1536x1024",
		quality: "medium",
		background: "opaque",
		suffix:
			"Minecraft Java Edition in-game screenshot, blocky voxel style, modded gameplay scene, coherent block lighting, no photorealism, no UI overlays unless explicitly requested.",
	},
	minecraft: {
		size: "1536x1024",
		quality: "medium",
		background: "opaque",
		suffix: "Minecraft in-game screenshot, blocky voxel style, Java Edition modded scene, no photorealism.",
	},
	poster: {
		size: "1024x1536",
		quality: "high",
		suffix: "Editorial poster composition, striking layout, cinematic lighting, polished art direction.",
	},
	wallpaper: {
		size: "1536x1024",
		quality: "high",
		suffix: "Desktop wallpaper composition, wide cinematic framing, visually rich but uncluttered.",
	},
};

const TOOL_PARAMS = Type.Object({
	prompt: Type.String({ description: "Image description/prompt." }),
	size: Type.Optional(StringEnum(SIZES)),
	quality: Type.Optional(StringEnum(QUALITIES)),
	background: Type.Optional(StringEnum(BACKGROUNDS)),
	outputFormat: Type.Optional(StringEnum(OUTPUT_FORMATS)),
	thinking: Type.Optional(
		StringEnum(THINKING_MODES, {
			description:
				"Dispatcher model reasoning effort before calling image_generation. Use 'off' to omit explicit reasoning. Defaults to 'low'.",
		}),
	),
	referencePaths: Type.Optional(
		Type.Array(Type.String(), {
			description: "Optional local image paths to send as visual references using input_image content.",
		}),
	),
	outputPath: Type.Optional(
		Type.String({
			description:
				"Optional exact path where the generated image should be saved. Defaults to ~/.pi/agent/generated-images/<id>.<format>.",
		}),
	),
});

type ToolParams = Static<typeof TOOL_PARAMS>;

interface ImagegenMetadata {
	createdAt: string;
	prompt: string;
	provider: string;
	responseModel: string;
	imageModel: string;
	imageId: string;
	savedPath: string;
	metadataPath: string;
	mimeType: string;
	revisedPrompt?: string;
	size: string;
	quality: string;
	background: string;
	outputFormat: string;
	thinking: string;
	referenceImageIds?: string[];
	referencePaths?: string[];
	batchId?: string;
	batchPrompt?: string;
	batchIndex?: number;
	batchCount?: number;
	kind?: "generated" | "sketch";
}

interface ImagegenDetails {
	provider: string;
	responseModel: string;
	imageModel: string;
	imageId: string;
	savedPath: string;
	metadataPath: string;
	mimeType: string;
	revisedPrompt?: string;
	size: string;
	quality: string;
	background: string;
	outputFormat: string;
	thinking: string;
}

interface CodexAccountClaims {
	"https://api.openai.com/auth"?: {
		chatgpt_account_id?: string;
	};
}

function decodeJwtPayload(token: string): CodexAccountClaims {
	const parts = token.split(".");
	if (parts.length < 2) {
		throw new Error("OpenAI Codex OAuth access token is not a JWT.");
	}
	return JSON.parse(Buffer.from(parts[1]!, "base64url").toString("utf8")) as CodexAccountClaims;
}

function getAccountId(token: string): string {
	const accountId = decodeJwtPayload(token)["https://api.openai.com/auth"]?.chatgpt_account_id;
	if (!accountId) {
		throw new Error("Could not find chatgpt_account_id in OpenAI Codex OAuth token.");
	}
	return accountId;
}

function mimeFromFormat(format: string): string {
	if (format === "jpeg") return "image/jpeg";
	if (format === "webp") return "image/webp";
	return "image/png";
}

function extensionFromFormat(format: string): string {
	return format === "jpeg" ? "jpg" : format;
}

function defaultOutputPath(imageId: string, format: string): string {
	const ext = extensionFromFormat(format);
	const stamp = new Date().toISOString().replace(/[:.]/g, "-");
	return join(getAgentDir(), "generated-images", `${stamp}-${imageId}.${ext}`);
}

function resolveOutputPath(path: string | undefined, cwd: string, imageId: string, format: string): string {
	if (!path || !path.trim()) return defaultOutputPath(imageId, format);
	const raw = path.trim().startsWith("@") ? path.trim().slice(1) : path.trim();
	const absolute = resolve(cwd, raw);
	if (!extname(absolute)) {
		return join(absolute, `${imageId}.${extensionFromFormat(format)}`);
	}
	return absolute;
}

async function saveImage(path: string, base64: string): Promise<void> {
	await withFileMutationQueue(path, async () => {
		await mkdir(dirname(path), { recursive: true });
		await writeFile(path, Buffer.from(base64, "base64"));
	});
}

function metadataPathForImage(path: string): string {
	const ext = extname(path);
	return ext ? `${path.slice(0, -ext.length)}.json` : `${path}.json`;
}

async function saveMetadata(metadata: ImagegenMetadata): Promise<void> {
	await withFileMutationQueue(metadata.metadataPath, async () => {
		await mkdir(dirname(metadata.metadataPath), { recursive: true });
		await writeFile(metadata.metadataPath, JSON.stringify(metadata, null, 2), "utf8");
	});

	// Global index: keeps /img list working even when outputPath points outside
	// ~/.pi/agent/generated-images, e.g. /tmp/foo.png or a project asset dir.
	const indexPath = join(getAgentDir(), "generated-images", "index", `${metadata.imageId}.json`);
	await withFileMutationQueue(indexPath, async () => {
		await mkdir(dirname(indexPath), { recursive: true });
		await writeFile(indexPath, JSON.stringify(metadata, null, 2), "utf8");
	});
}

async function findJsonFilesRecursive(dir: string): Promise<string[]> {
	if (!existsSync(dir)) return [];
	const entries = await readdir(dir, { withFileTypes: true });
	const files: string[] = [];
	for (const entry of entries) {
		const path = join(dir, entry.name);
		if (entry.isDirectory()) {
			files.push(...(await findJsonFilesRecursive(path)));
		} else if (entry.isFile() && entry.name.endsWith(".json")) {
			files.push(path);
		}
	}
	return files;
}

async function readRecentMetadata(limit = 10): Promise<ImagegenMetadata[]> {
	const dir = join(getAgentDir(), "generated-images");
	const files = await findJsonFilesRecursive(dir);
	const byImageId = new Map<string, ImagegenMetadata>();
	for (const file of files) {
		try {
			const parsed = JSON.parse(await readFile(file, "utf8")) as Partial<ImagegenMetadata>;
			if (!parsed.imageId || !parsed.savedPath || !parsed.createdAt || !parsed.prompt) continue;
			byImageId.set(parsed.imageId, parsed as ImagegenMetadata);
		} catch {
			// Ignore stale/bad sidecars and batch.json files.
		}
	}
	return [...byImageId.values()].sort((a, b) => b.createdAt.localeCompare(a.createdAt)).slice(0, limit);
}

async function resolveImageTarget(target: string, cwd: string): Promise<string | undefined> {
	const trimmed = target.trim() || "latest";
	if (trimmed === "latest" || /^\d+$/.test(trimmed)) {
		const index = trimmed === "latest" ? 0 : Number.parseInt(trimmed, 10) - 1;
		const recent = await readRecentMetadata(Math.max(index + 1, 1));
		return recent[index]?.savedPath;
	}
	const raw = trimmed.startsWith("@") ? trimmed.slice(1) : trimmed;
	return resolve(cwd, raw);
}

function macOpen(path: string, reveal = false): void {
	const args = reveal ? ["-R", path] : [path];
	const child = spawn("open", args, { detached: true, stdio: "ignore" });
	child.unref();
}

function parseImgArgs(input: string): { options: Partial<ToolParams> & { style?: string }; positional: string[] } {
	const tokens = input.match(/(?:[^\s"]+|"[^"]*")+/g)?.map((token) => token.replace(/^"|"$/g, "")) ?? [];
	const options: Partial<ToolParams> & { style?: string } = {};
	const positional: string[] = [];
	for (let index = 0; index < tokens.length; index++) {
		const token = tokens[index]!;
		const next = tokens[index + 1];
		if (token === "--style" && next) {
			options.style = next;
			index++;
		} else if (token === "--size" && next) {
			options.size = next as ToolParams["size"];
			index++;
		} else if (token === "--quality" && next) {
			options.quality = next as ToolParams["quality"];
			index++;
		} else if (token === "--background" && next) {
			options.background = next as ToolParams["background"];
			index++;
		} else if ((token === "--format" || token === "--output-format") && next) {
			options.outputFormat = next as ToolParams["outputFormat"];
			index++;
		} else if ((token === "--thinking" || token === "--reasoning") && next) {
			options.thinking = next as ToolParams["thinking"];
			index++;
		} else if ((token === "--out" || token === "--output") && next) {
			options.outputPath = next;
			index++;
		} else {
			positional.push(token);
		}
	}
	return { options, positional };
}

function applyStyle(prompt: string, options: Partial<ToolParams> & { style?: string }): ToolParams {
	const preset = options.style ? STYLE_PRESETS[options.style] : undefined;
	const styledPrompt = preset?.suffix ? `${prompt}. ${preset.suffix}` : prompt;
	return {
		prompt: styledPrompt,
		size: options.size ?? preset?.size,
		quality: options.quality ?? preset?.quality,
		background: options.background ?? preset?.background,
		outputFormat: options.outputFormat ?? preset?.outputFormat,
		thinking: options.thinking,
		referencePaths: options.referencePaths,
		outputPath: options.outputPath,
	};
}

function batchDirName(prompt: string): string {
	const stamp = new Date().toISOString().replace(/[:.]/g, "-");
	const slug = prompt
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, "-")
		.replace(/^-+|-+$/g, "")
		.slice(0, 48) || "batch";
	return `${stamp}-${slug}`;
}

async function findMetadataByImageId(imageId: string): Promise<ImagegenMetadata | undefined> {
	const recent = await readRecentMetadata(1000);
	return recent.find((item) => item.imageId === imageId);
}

function insertImageIntoPrompt(path: string, ctx: ExtensionContext | undefined): boolean {
	if (!ctx?.hasUI) return false;
	const ref = `@${path}`;
	const current = ctx.ui.getEditorText();
	const separator = current.length === 0 || /\s$/.test(current) ? "" : " ";
	ctx.ui.setEditorText(`${current}${separator}${ref}`);
	ctx.ui.notify(`Added image to prompt: ${path}`, "info");
	return true;
}

async function buildRequest(params: ToolParams, responseModel: string, sessionId: string) {
	const size = params.size ?? "auto";
	const quality = params.quality ?? "auto";
	const background = params.background ?? "auto";
	const outputFormat = params.outputFormat ?? "png";
	const thinking = params.thinking ?? "low";
	const content: any[] = [{ type: "input_text", text: `Generate this image: ${params.prompt}` }];
	for (const path of params.referencePaths ?? []) {
		const format = extname(path).toLowerCase() === ".jpg" ? "jpeg" : extname(path).toLowerCase().replace(/^\./, "") || "png";
		const mimeType = mimeFromFormat(format);
		const data = await readFile(path);
		content.push({ type: "input_image", image_url: `data:${mimeType};base64,${data.toString("base64")}` });
	}
	const request: any = {
		model: responseModel,
		store: false,
		stream: true,
		instructions:
			"You are an image generation dispatcher. Use the image_generation tool to create exactly the image requested by the user. Do not write code.",
		input: [
			{
				role: "user",
				content,
			},
		],
		text: { verbosity: "low" },
		prompt_cache_key: sessionId,
		tool_choice: "auto",
		parallel_tool_calls: true,
		tools: [
			{
				type: "image_generation",
				background,
				model: IMAGE_MODEL,
				moderation: "auto",
				output_compression: 100,
				output_format: outputFormat,
				quality,
				size,
			},
		],
	};
	if (thinking !== "off") {
		request.include = ["reasoning.encrypted_content"];
		request.reasoning = { effort: thinking, summary: "auto" };
	}
	return request;
}

async function parseSseForImage(response: Response, signal?: AbortSignal) {
	if (!response.body) throw new Error("No response body from Codex image generation request.");

	const reader = response.body.getReader();
	const decoder = new TextDecoder();
	let buffer = "";

	try {
		while (true) {
			if (signal?.aborted) throw new Error("Request was aborted");
			const { done, value } = await reader.read();
			if (done) break;
			buffer += decoder.decode(value, { stream: true });

			let index: number;
			while ((index = buffer.indexOf("\n\n")) !== -1) {
				const chunk = buffer.slice(0, index);
				buffer = buffer.slice(index + 2);
				const data = chunk
					.split("\n")
					.filter((line) => line.startsWith("data:"))
					.map((line) => line.slice(5).trim())
					.join("\n")
					.trim();
				if (!data || data === "[DONE]") continue;

				let event: any;
				try {
					event = JSON.parse(data);
				} catch {
					continue;
				}

				if (event.type === "error") {
					throw new Error(event.message || event.code || JSON.stringify(event));
				}
				if (event.type === "response.failed") {
					throw new Error(event.response?.error?.message || "Codex image generation failed.");
				}

				const item = event.item;
				if (event.type === "response.output_item.done" && item?.type === "image_generation_call") {
					if (!item.result) throw new Error("Image generation completed without image data.");
					return {
						id: item.id as string,
						base64: item.result as string,
						revisedPrompt: (item.revised_prompt ?? item.revisedPrompt) as string | undefined,
					};
				}
			}
		}
	} finally {
		try {
			reader.releaseLock();
		} catch {
			// ignore
		}
	}

	throw new Error("No image_generation_call result returned by Codex.");
}

type ImagegenContext = {
	cwd: string;
	model?: { provider: string; id: string };
	modelRegistry: { getApiKeyForProvider: (provider: string) => Promise<string | undefined> };
};

type ToolUpdate = (result: { content: Array<{ type: "text"; text: string }>; details?: unknown }) => void;

async function generateImage(
	params: ToolParams,
	signal: AbortSignal | undefined,
	onUpdate: ToolUpdate | undefined,
	ctx: ImagegenContext,
	extraMetadata: Partial<Pick<ImagegenMetadata, "batchId" | "batchPrompt" | "batchIndex" | "batchCount" | "referenceImageIds" | "referencePaths">> = {},
) {
	const token = await ctx.modelRegistry.getApiKeyForProvider(PROVIDER);
	if (!token) {
		throw new Error("Missing OpenAI Codex OAuth credentials. Run /login and select OpenAI ChatGPT Plus/Pro (Codex).");
	}

	const accountId = getAccountId(token);
	const responseModel = ctx.model?.provider === PROVIDER ? ctx.model.id : DEFAULT_RESPONSE_MODEL;
	const sessionId = randomUUID();
	const body = await buildRequest(params, responseModel, sessionId);
	const outputFormat = params.outputFormat ?? "png";
	const mimeType = mimeFromFormat(outputFormat);

	onUpdate?.({
		content: [{ type: "text", text: `Requesting image from ${PROVIDER}/${IMAGE_MODEL}...` }],
		details: { provider: PROVIDER, imageModel: IMAGE_MODEL, responseModel },
	});

	const response = await fetch(`${CODEX_BASE_URL}/codex/responses`, {
		method: "POST",
		headers: {
			Authorization: `Bearer ${token}`,
			"chatgpt-account-id": accountId,
			originator: "pi-imagegen-extension",
			"OpenAI-Beta": "responses=experimental",
			accept: "text/event-stream",
			"content-type": "application/json",
			session_id: sessionId,
			"x-client-request-id": sessionId,
			"User-Agent": `pi-imagegen-extension (${process.platform}; ${process.arch})`,
		},
		body: JSON.stringify(body),
		signal,
	});

	if (!response.ok) {
		const errorText = await response.text();
		throw new Error(`Codex image request failed (${response.status}): ${errorText}`);
	}

	const image = await parseSseForImage(response, signal);
	const savedPath = resolveOutputPath(params.outputPath, ctx.cwd, image.id, outputFormat);
	await saveImage(savedPath, image.base64);

	const metadataPath = metadataPathForImage(savedPath);
	const createdAt = new Date().toISOString();
	const metadata: ImagegenMetadata = {
		createdAt,
		prompt: params.prompt,
		provider: PROVIDER,
		responseModel,
		imageModel: IMAGE_MODEL,
		imageId: image.id,
		savedPath,
		metadataPath,
		mimeType,
		revisedPrompt: image.revisedPrompt,
		size: params.size ?? "auto",
		quality: params.quality ?? "auto",
		background: params.background ?? "auto",
		outputFormat,
		thinking: params.thinking ?? "low",
		...extraMetadata,
	};
	await saveMetadata(metadata);

	const details: ImagegenDetails = metadata;

	const text = [
		`Generated image with ${PROVIDER}/${IMAGE_MODEL}.`,
		`Saved to: ${savedPath}`,
		image.revisedPrompt ? `Revised prompt: ${image.revisedPrompt}` : undefined,
	]
		.filter(Boolean)
		.join("\n");

	return { image, text, details };
}

function writeHtml(res: ServerResponse, html: string) {
	res.writeHead(200, { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-cache, no-store, must-revalidate" });
	res.end(html);
}

function writeJson(res: ServerResponse, statusCode: number, value: unknown) {
	res.writeHead(statusCode, { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-cache, no-store, must-revalidate" });
	res.end(JSON.stringify(value));
}

function writeText(res: ServerResponse, statusCode: number, text: string) {
	res.writeHead(statusCode, { "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-cache, no-store, must-revalidate" });
	res.end(text);
}

async function readRequestBody(req: IncomingMessage, maxBytes = 20 * 1024 * 1024): Promise<Buffer> {
	const chunks: Buffer[] = [];
	let total = 0;
	for await (const chunk of req) {
		const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
		total += buffer.length;
		if (total > maxBytes) throw new Error(`Request body too large; max ${Math.round(maxBytes / 1024 / 1024)}MB.`);
		chunks.push(buffer);
	}
	return Buffer.concat(chunks);
}

async function readJsonBody(req: IncomingMessage): Promise<any> {
	const body = await readRequestBody(req);
	if (body.length === 0) return {};
	return JSON.parse(body.toString("utf8"));
}

function isPng(buffer: Buffer): boolean {
	return buffer.length >= 8 && buffer[0] === 0x89 && buffer[1] === 0x50 && buffer[2] === 0x4e && buffer[3] === 0x47 && buffer[4] === 0x0d && buffer[5] === 0x0a && buffer[6] === 0x1a && buffer[7] === 0x0a;
}

function openBrowser(url: string): Promise<void> {
	const command = process.platform === "darwin" ? "open" : process.platform === "win32" ? "cmd" : "xdg-open";
	const args = process.platform === "win32" ? ["/c", "start", "", url] : [url];
	return new Promise((resolve, reject) => {
		const child = spawn(command, args, { detached: true, stdio: "ignore" });
		child.once("error", reject);
		child.once("spawn", () => {
			child.unref();
			resolve();
		});
	});
}

function renderStudioPage(token: string): string {
	return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Pi Image Studio</title>
<style>
/*
  Pi Image Studio — Design Language: "Quiet Atelier"
  --------------------------------------------------
  A local-first creative instrument, not a SaaS dashboard.
  · Surface  — warm off-white paper with a layered radial mesh + faint grain
               (paper-shader atmosphere) so the canvas breathes without noise.
  · Wall     — full uncropped images, no chrome, no cards. The work is the hero.
  · Dock     — a Cursor-style command surface: serif-italic prompt voice,
               mono numerics for controls, integrated primary action.
  · Signal   — Lovable-style optimistic status pill (live · generating · updated).
  · Accent   — a single muted ultramarine reserved for focus + live moments.
*/
:root{
  color-scheme:light;
  /* paper */
  --paper:#fafaf6;
  --paper-2:#f1efe9;
  --paper-3:#e7e4dc;
  /* ink */
  --ink:#0e0d0b;
  --ink-2:#272622;
  --ink-3:#5a5750;
  --muted:#8b877d;
  --hair:rgba(14,13,11,0.07);
  --hair-2:rgba(14,13,11,0.14);
  --hair-3:rgba(14,13,11,0.30);
  /* signals */
  --live:#2f6b4a;
  --accent:#2745d6;
  --accent-soft:rgba(39,69,214,0.10);
  /* shadows */
  --shadow-card:0 1px 0 rgba(14,13,11,0.05);
  --shadow-float:0 28px 80px -28px rgba(14,13,11,0.22),0 2px 8px rgba(14,13,11,0.05);
  --shadow-modal:0 60px 120px -30px rgba(14,13,11,0.35);
  /* type */
  --serif:"New York","Iowan Old Style","Hoefler Text",Charter,"Source Serif Pro",Cambria,Georgia,serif;
  --sans:"SF Pro Text","Helvetica Neue","Inter",-apple-system,system-ui,"Segoe UI",sans-serif;
  --mono:"JetBrains Mono","SF Mono",ui-monospace,Menlo,Consolas,"Roboto Mono",monospace;
}
*{box-sizing:border-box}
html,body{height:100%;margin:0}
body{
  background:var(--paper);
  color:var(--ink);
  font:13.5px/1.5 var(--sans);
  font-feature-settings:"kern","tnum","ss01";
  -webkit-font-smoothing:antialiased;
  overflow:hidden;
}
/* paper-shader atmosphere: two warm radial washes + faint grain dot pattern */
body::before{
  content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
  background:
    radial-gradient(900px 620px at 8% 4%,  rgba(255,217,168,0.22), transparent 62%),
    radial-gradient(1100px 720px at 96% 96%, rgba(180,200,255,0.18), transparent 65%),
    radial-gradient(700px 500px at 60% 30%, rgba(255,200,210,0.10), transparent 70%);
}
body::after{
  content:"";position:fixed;inset:0;pointer-events:none;z-index:0;opacity:0.55;
  background-image:radial-gradient(rgba(14,13,11,0.020) 1px,transparent 1px);
  background-size:3px 3px;
}
::selection{background:var(--ink);color:var(--paper)}

.studio{height:100vh;display:flex;flex-direction:column;position:relative;z-index:1}

/* ─────────── top bar ─────────── */
.top{
  height:56px;
  display:grid;grid-template-columns:1fr auto 1fr;align-items:center;
  padding:0 28px;
  border-bottom:1px solid var(--hair);
  background:rgba(250,250,246,0.72);
  backdrop-filter:blur(10px) saturate(1.05);
  -webkit-backdrop-filter:blur(10px) saturate(1.05);
  position:relative;z-index:4;
}
.brand{display:flex;align-items:center;gap:10px;color:var(--ink);font-family:var(--sans);font-size:13px;letter-spacing:-0.005em}
.brand .glyph{
  width:18px;height:18px;display:grid;place-items:center;
  background:var(--ink);color:var(--paper);
  font-family:var(--serif);font-style:italic;font-size:13px;line-height:1;
  border-radius:2px;
}
.brand strong{font-weight:600;letter-spacing:-0.01em}
.brand .sub{
  font-family:var(--mono);font-size:10.5px;color:var(--muted);
  text-transform:uppercase;letter-spacing:0.16em;
  padding-left:9px;border-left:1px solid var(--hair-2);
}
.filters{
  justify-self:center;display:flex;gap:2px;align-items:center;
  background:rgba(255,255,255,0.55);border:1px solid var(--hair);
  border-radius:999px;padding:3px;
}
.filters button{
  all:unset;cursor:pointer;
  font-family:var(--mono);font-size:10.5px;letter-spacing:0.12em;text-transform:uppercase;
  color:var(--muted);padding:5px 12px;border-radius:999px;
  transition:color .15s ease,background .15s ease;
}
.filters button:hover{color:var(--ink-2)}
.filters button.active{color:var(--ink);background:var(--paper)}
.right{
  justify-self:end;display:flex;gap:8px;align-items:center;
  font-family:var(--mono);font-size:10.5px;letter-spacing:0.14em;text-transform:uppercase;color:var(--muted);
  padding:5px 11px;border:1px solid var(--hair);border-radius:999px;background:rgba(255,255,255,0.55);
}
.pulse{
  display:inline-block;width:6px;height:6px;border-radius:50%;background:#c4bfb1;
  box-shadow:0 0 0 0 rgba(196,191,177,0.6);
}
.right.live .pulse{background:var(--live);box-shadow:0 0 0 0 rgba(47,107,74,0.45);animation:beat 1.8s ease-out infinite}
.right.busy .pulse{background:var(--accent);animation:beat 1.0s ease-out infinite}
@keyframes beat{
  0%{box-shadow:0 0 0 0 currentColor;opacity:1}
  70%{box-shadow:0 0 0 7px rgba(0,0,0,0);opacity:0.85}
  100%{box-shadow:0 0 0 0 rgba(0,0,0,0);opacity:1}
}

/* ─────────── canvas / wall ─────────── */
.canvas{
  flex:1;overflow:auto;
  padding:44px 44px 220px;
  scroll-behavior:smooth;
}
.headline{
  max-width:1500px;margin:0 auto 32px;
  display:flex;align-items:flex-end;gap:20px;
}
.headline h1{
  font-family:var(--serif);font-weight:400;font-style:italic;
  font-size:clamp(34px,4.6vw,52px);
  letter-spacing:-0.025em;
  margin:0;color:var(--ink);line-height:0.95;
}
.headline h1 .amp{color:var(--ink-3);font-style:italic;margin:0 0.18em}
.headline .rule{flex:1;height:1px;background:var(--hair);margin-bottom:9px}
.headline .meta{
  font-family:var(--mono);font-size:10.5px;letter-spacing:0.18em;text-transform:uppercase;color:var(--muted);
  font-variant-numeric:tabular-nums;white-space:nowrap;margin-bottom:6px;
}

.wall{max-width:1500px;margin:0 auto;display:grid;grid-template-columns:repeat(var(--cols,4),minmax(0,1fr));gap:18px;align-items:start}
.mcol{display:flex;flex-direction:column;gap:18px;min-width:0}
.tile{
  display:block;margin:0;padding:0;
  border:0;background:transparent;cursor:pointer;position:relative;
  appearance:none;-webkit-appearance:none;line-height:0;text-align:left;
}
.tile img{
  display:block;width:100%;height:auto;object-fit:contain;border-radius:0;cursor:pointer;
  background:var(--paper-2);
  box-shadow:0 1px 0 var(--hair),0 2px 12px -8px rgba(14,13,11,0.10);
}
.tile::after{
  content:"";position:absolute;inset:0;pointer-events:none;
  border:1px solid transparent;
  transition:border-color .14s ease;
}
.tile:hover::after{border-color:var(--hair-3)}
.tile:hover img{
  box-shadow:0 1px 0 var(--hair),0 2px 12px -8px rgba(14,13,11,0.10);
}
.tile:focus-visible{outline:none}
.tile:focus-visible::after{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}

.batch-list{max-width:1500px;margin:0 auto;display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:18px;align-items:start}
.batch-card{background:rgba(255,255,255,0.46);border:1px solid var(--hair);border-radius:10px;padding:10px;box-shadow:var(--shadow-card);backdrop-filter:blur(8px) saturate(1.03)}
.batch-preview{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;min-height:120px;align-items:center;background:rgba(245,242,234,0.48);padding:7px;border:1px solid var(--hair)}
.batch-preview .tile{margin:0;min-width:0}
.batch-preview .tile img{max-height:150px;width:100%;object-fit:contain}
.batch-info{display:grid;gap:4px;padding:10px 2px 1px}
.batch-title{font-family:var(--serif);font-style:italic;font-size:17px;line-height:1.2;color:var(--ink);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.batch-meta{font-family:var(--mono);font-size:10px;letter-spacing:0.14em;text-transform:uppercase;color:var(--muted);white-space:nowrap}

/* ─────────── empty state ─────────── */
.empty{max-width:560px;margin:14vh auto;text-align:center;color:var(--muted)}
.empty .mark{
  font-family:var(--serif);font-style:italic;color:var(--hair-3);
  font-size:84px;line-height:0.9;margin-bottom:20px;letter-spacing:-0.04em;
}
.empty h2{
  font-family:var(--serif);font-weight:400;font-style:italic;color:var(--ink);
  font-size:38px;letter-spacing:-0.022em;margin:0 0 12px;
}
.empty p{font-size:13.5px;line-height:1.65;color:var(--muted);max-width:380px;margin:0 auto}
.empty .kbd{
  display:inline-flex;gap:5px;margin-top:18px;font-family:var(--mono);font-size:10.5px;
  letter-spacing:0.12em;text-transform:uppercase;color:var(--ink-3);
}
.empty .kbd kbd{
  font:inherit;padding:3px 7px;border:1px solid var(--hair-2);border-radius:4px;
  background:rgba(255,255,255,0.7);color:var(--ink-2);
}

/* ─────────── composer ─────────── */
.composer{
  position:fixed;left:50%;bottom:28px;transform:translateX(-50%);
  width:min(860px,calc(100vw - 56px));
  background:rgba(250,250,246,0.85);
  border:1px solid var(--hair-2);
  box-shadow:var(--shadow-float);
  backdrop-filter:blur(24px) saturate(1.10);
  -webkit-backdrop-filter:blur(24px) saturate(1.10);
  z-index:5;
  border-radius:14px;
  padding:18px 18px 12px;
}
.composer::before{
  content:"";position:absolute;inset:0;pointer-events:none;border-radius:inherit;
  background:linear-gradient(180deg,rgba(255,255,255,0.65),transparent 38%);
  mix-blend-mode:overlay;opacity:0.7;
}
.refs{display:none;gap:8px;flex-wrap:wrap;padding:0 6px 10px}
.refs.hasRefs{display:flex}
.ref-chip{position:relative;width:54px;height:42px;border:1px solid var(--hair-2);background:var(--paper-2);overflow:hidden}
.ref-chip img{width:100%;height:100%;object-fit:cover;display:block}
.ref-chip button{position:absolute;right:2px;top:2px;border:0;background:rgba(250,250,246,.9);color:var(--ink);width:18px;height:18px;border-radius:999px;cursor:pointer;font-size:12px;line-height:1}
.sketch-ref{border:1px dashed var(--hair-3);background:rgba(255,255,255,.45);color:var(--ink-2);height:42px;padding:0 12px;font-family:var(--mono);font-size:10px;letter-spacing:.12em;text-transform:uppercase;cursor:pointer}
.promptRow{display:flex;align-items:flex-start;gap:12px;padding:4px 6px 14px}
.slash{
  font-family:var(--mono);font-size:11px;color:var(--muted);
  letter-spacing:0.06em;padding:6px 8px;border:1px solid var(--hair);border-radius:6px;
  background:rgba(255,255,255,0.6);user-select:none;line-height:1;margin-top:6px;
}
.promptBox{
  flex:1;width:100%;min-height:32px;max-height:160px;resize:none;overflow:auto;
  border:0;outline:0;background:transparent;color:var(--ink);
  font-family:var(--serif);font-size:21px;font-style:italic;line-height:1.4;
  padding:0;letter-spacing:-0.005em;
}
.promptBox::placeholder{color:var(--muted);font-style:italic}
.controls{
  display:flex;gap:6px;align-items:center;flex-wrap:wrap;
  border-top:1px solid var(--hair);padding-top:10px;
  font-family:var(--mono);font-size:10.5px;letter-spacing:0.12em;text-transform:uppercase;color:var(--muted);
}
.field{
  display:flex;align-items:center;gap:8px;white-space:nowrap;
  padding:6px 10px;border-radius:6px;
  transition:background .12s ease;
}
.field:hover{background:rgba(255,255,255,0.55)}
.field > label{color:var(--muted);font-size:10px;letter-spacing:0.16em}
.select{
  appearance:none;-webkit-appearance:none;
  border:0;background:transparent;color:var(--ink);
  font-family:var(--mono);font-size:11px;letter-spacing:0.06em;text-transform:none;
  padding:0 14px 0 0;cursor:pointer;line-height:1.6;
  background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='8' height='5' viewBox='0 0 8 5'><path d='M0 0l4 5 4-5z' fill='%230e0d0b'/></svg>");
  background-repeat:no-repeat;background-position:right center;
}
.select:focus{outline:none;color:var(--accent)}
.steps{display:inline-flex;border:1px solid var(--hair-2);border-radius:6px;overflow:hidden;background:rgba(255,255,255,0.6)}
.steps button{
  all:unset;cursor:pointer;
  width:28px;height:24px;display:inline-grid;place-items:center;
  font-family:var(--mono);font-size:11px;color:var(--ink-2);
  border-right:1px solid var(--hair);
  font-variant-numeric:tabular-nums;
  transition:background .12s ease,color .12s ease;
}
.steps button:last-child{border-right:0}
.steps button:hover{background:var(--paper-2)}
.steps button.active{background:var(--ink);color:var(--paper)}
.spacer{flex:1;min-width:8px}
.generate{
  border:0;cursor:pointer;
  display:inline-flex;align-items:center;gap:10px;
  background:var(--ink);color:var(--paper);
  font-family:var(--sans);font-size:11.5px;font-weight:600;letter-spacing:0.14em;text-transform:uppercase;
  padding:0 14px;height:34px;border-radius:8px;
  transition:background .15s ease,transform .12s ease;
  box-shadow:0 1px 0 rgba(0,0,0,0.20),0 6px 18px -10px rgba(14,13,11,0.5);
}
.generate:hover{background:#000;transform:translateY(-0.5px)}
.generate:active{transform:translateY(0)}
.generate:disabled{opacity:0.55;cursor:default;transform:none}
.generate .ret{
  font-family:var(--mono);font-size:10.5px;font-weight:500;letter-spacing:0;
  display:inline-grid;place-items:center;width:18px;height:18px;
  background:rgba(255,255,255,0.14);border:1px solid rgba(255,255,255,0.18);border-radius:4px;
  color:rgba(250,250,246,0.85);
}

.sketch-modal{position:fixed;inset:0;z-index:11;display:none;align-items:center;justify-content:center;background:rgba(250,250,246,.72);backdrop-filter:blur(8px)}
.sketch-modal.open{display:flex}
.sketch-panel{width:min(980px,calc(100vw - 32px));background:var(--paper);border:1px solid var(--hair-2);box-shadow:var(--shadow-float);border-radius:14px;padding:14px}
.sketch-top{display:flex;align-items:center;gap:8px;margin-bottom:12px;font-family:var(--mono);font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
.sketch-top strong{font-family:var(--sans);font-size:13px;letter-spacing:0;text-transform:none;color:var(--ink);margin-right:auto}
.sketch-top button,.sketch-top input{font:inherit}
.sketch-top button{border:1px solid var(--hair);background:white;border-radius:7px;padding:7px 10px;cursor:pointer;color:var(--ink)}
.sketch-top button.active{background:var(--ink);color:var(--paper)}
#sketchCanvas{display:block;width:100%;height:min(62vh,620px);background:white;border:1px solid var(--hair-2);touch-action:none;cursor:crosshair}

/* ─────────── modal ─────────── */
.modal{
  position:fixed;inset:0;z-index:8;display:none;align-items:center;justify-content:center;
  background:radial-gradient(ellipse at center,rgba(250,250,246,0.55),rgba(250,250,246,0.78));
  backdrop-filter:blur(4px) saturate(1.02);
  -webkit-backdrop-filter:blur(4px) saturate(1.02);
}
.modal.open{display:flex;animation:fadeIn .18s ease}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
.modalImg{
  max-width:min(92vw,1400px);max-height:78vh;
  width:auto;height:auto;object-fit:contain;border-radius:0;
  box-shadow:var(--shadow-modal);
  background:var(--paper-2);
  animation:zoom .22s cubic-bezier(.2,.7,.2,1);
}
@keyframes zoom{from{opacity:0;transform:scale(.985)}to{opacity:1;transform:scale(1)}}
.modal-counter{
  position:fixed;top:22px;left:50%;transform:translateX(-50%);
  font-family:var(--mono);font-size:10.5px;letter-spacing:0.18em;text-transform:uppercase;color:var(--ink-3);
  font-variant-numeric:tabular-nums;z-index:9;
  padding:6px 12px;background:rgba(255,255,255,0.7);border:1px solid var(--hair);border-radius:999px;
}
.modal-bar{
  position:fixed;left:50%;bottom:28px;transform:translateX(-50%);
  display:flex;align-items:center;background:rgba(250,250,246,0.92);
  border:1px solid var(--hair-2);box-shadow:var(--shadow-float);z-index:9;
  border-radius:10px;overflow:hidden;
  backdrop-filter:blur(16px) saturate(1.05);
  -webkit-backdrop-filter:blur(16px) saturate(1.05);
}
.modal-bar button{
  all:unset;cursor:pointer;
  font-family:var(--mono);font-size:10.5px;letter-spacing:0.12em;text-transform:uppercase;font-weight:500;
  color:var(--ink-2);padding:11px 16px;
  border-right:1px solid var(--hair);
  transition:background .12s ease,color .12s ease;
  display:inline-flex;align-items:center;gap:7px;
}
.modal-bar button:last-child{border-right:0}
.modal-bar button:hover{background:var(--paper-2);color:var(--ink)}
.close{
  position:fixed;right:22px;top:18px;border:0;background:rgba(255,255,255,0.7);
  font-family:var(--sans);font-size:18px;line-height:1;cursor:pointer;color:var(--ink);z-index:9;
  width:34px;height:34px;display:grid;place-items:center;border-radius:999px;
  border:1px solid var(--hair);transition:background .15s ease,transform .15s ease;
}
.close:hover{background:var(--paper);transform:rotate(90deg)}
.nav{
  position:fixed;top:50%;transform:translateY(-50%);background:rgba(255,255,255,0.7);
  border:1px solid var(--hair);
  font-family:var(--serif);font-style:italic;font-size:24px;line-height:1;
  cursor:pointer;color:var(--ink-2);width:42px;height:42px;z-index:9;border-radius:999px;
  transition:background .15s ease,color .15s ease,transform .15s ease;display:grid;place-items:center;
}
.nav:hover{background:var(--paper);color:var(--ink)}
.nav.prev{left:22px}.nav.prev:hover{transform:translateY(-50%) translateX(-2px)}
.nav.next{right:22px}.nav.next:hover{transform:translateY(-50%) translateX(2px)}

/* ─────────── toast ─────────── */
.toast{
  position:fixed;left:50%;top:18px;transform:translateX(-50%);
  background:var(--ink);color:var(--paper);padding:9px 16px;
  font-family:var(--mono);font-size:10.5px;letter-spacing:0.14em;text-transform:uppercase;font-weight:500;
  display:none;z-index:10;border-radius:999px;
  box-shadow:0 10px 30px -10px rgba(14,13,11,0.35);
}
.toast.show{display:block;animation:fadeIn .18s ease}

/* ─────────── mobile ─────────── */
@media(max-width:760px){
  .top{padding:0 16px;grid-template-columns:1fr auto;height:52px}
  .filters{display:none}
  .canvas{padding:24px 16px 280px}
  .headline{margin-bottom:18px;gap:12px}
  .headline .rule{display:none}
  .wall{gap:10px}
  .mcol{gap:10px}
  .batch-list{grid-template-columns:1fr;gap:12px}
  .batch-title{font-size:16px}
  .composer{width:calc(100vw - 24px);bottom:14px;padding:14px 14px 10px;border-radius:12px}
  .promptBox{font-size:18px}
  .refs{padding:0 2px 8px}
  .promptRow{padding:2px 2px 12px;gap:8px}
  .slash{display:none}
  .controls{flex-wrap:wrap;gap:4px}
  .field{padding:6px 8px}
  .spacer{display:none}
  .generate{margin-left:auto;height:32px}
  .nav{display:none}
  .modal-bar{flex-wrap:wrap;width:calc(100vw - 24px);justify-content:center}
}
</style>
</head>
<body>
<div class="studio">
  <header class="top">
    <div class="brand">
      <span class="glyph">π</span>
      <strong>Image Studio</strong>
      <span class="sub">Local · gpt-image-2</span>
    </div>
    <nav class="filters">
      <button class="active" data-filter="all">All</button>
      <button data-filter="batch">Batches</button>
      <button data-filter="tmp">Scratch</button>
    </nav>
    <div class="right" id="status-wrap">
      <span class="pulse"></span>
      <span id="status">Connecting</span>
    </div>
  </header>
  <main class="canvas">
    <div class="headline">
      <h1>Atelier<span class="amp">,</span> <em>quietly.</em></h1>
      <span class="rule"></span>
      <span class="meta" id="count">00 plates</span>
    </div>
    <div id="wall" class="wall"></div>
  </main>
  <div id="modal" class="modal" aria-hidden="true"></div>
  <div id="sketchModal" class="sketch-modal" aria-hidden="true">
    <div class="sketch-panel">
      <div class="sketch-top">
        <strong>Sketch reference</strong>
        <button type="button" id="sketchBrush" class="active">Brush</button>
        <button type="button" id="sketchEraser">Eraser</button>
        <label>Size <input id="sketchSize" type="range" min="2" max="48" value="8"></label>
        <button type="button" id="sketchClear">Clear</button>
        <button type="button" id="sketchUse">Use ref</button>
        <button type="button" id="sketchClose">×</button>
      </div>
      <canvas id="sketchCanvas" width="1024" height="1024"></canvas>
    </div>
  </div>
  <form id="composer" class="composer" autocomplete="off">
    <div id="refs" class="refs"></div>
    <div class="promptRow">
      <span class="slash" aria-hidden="true">/</span>
      <textarea id="prompt" class="promptBox" placeholder="Describe what you'd like to see…" rows="1"></textarea>
    </div>
    <div class="controls">
      <div class="field"><label>Style</label>
        <select id="style" class="select">
          <option value="">auto</option>
          <option value="minecraft-screenshot">minecraft</option>
          <option value="poster">poster</option>
          <option value="wallpaper">wallpaper</option>
        </select>
      </div>
      <div class="field"><label>Aspect</label>
        <select id="size" class="select">
          <option value="auto">auto</option>
          <option value="1024x1024">1:1 square</option>
          <option value="1536x1024">3:2 land</option>
          <option value="1024x1536">2:3 port</option>
        </select>
      </div>
      <div class="field"><label>Quality</label>
        <select id="quality" class="select">
          <option value="auto">auto</option>
          <option value="medium">medium</option>
          <option value="high">high</option>
          <option value="low">low</option>
        </select>
      </div>
      <div class="field"><label>Thinking</label>
        <select id="thinking" class="select">
          <option value="off">off</option>
          <option value="minimal">minimal</option>
          <option value="low" selected>low</option>
          <option value="medium">medium</option>
          <option value="high">high</option>
        </select>
      </div>
      <div class="field"><label>Count</label>
        <div class="steps" id="counts">
          <button type="button" data-n="1">1</button>
          <button type="button" data-n="2">2</button>
          <button type="button" data-n="4" class="active">4</button>
          <button type="button" data-n="6">6</button>
          <button type="button" data-n="9">9</button>
        </div>
      </div>
      <span class="spacer"></span>
      <button id="generate" class="generate" type="submit">
        <span>Generate</span>
        <span class="ret" aria-hidden="true">↩</span>
      </button>
    </div>
  </form>
  <div id="toast" class="toast"></div>
</div>
<script>
const TOKEN=${JSON.stringify(token)};
let images=[],selected=null,filter='all',count=4,refs=[];
const $=s=>document.querySelector(s),$$=s=>document.querySelectorAll(s);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pad=n=>String(n).padStart(2,'0');
const imgUrl=x=>'/api/image/'+encodeURIComponent(x.imageId)+'?token='+encodeURIComponent(TOKEN);
async function api(path,opts={}){const sep=path.includes('?')?'&':'?';const r=await fetch(path+sep+'token='+encodeURIComponent(TOKEN),opts);if(!r.ok)throw new Error(await r.text());return r.headers.get('content-type')?.includes('json')?r.json():r.text()}
function toast(t){const el=$('#toast');el.textContent=t;el.classList.add('show');clearTimeout(toast._t);toast._t=setTimeout(()=>el.classList.remove('show'),1800)}
function renderRefs(){const el=$('#refs');el.classList.toggle('hasRefs',refs.length>0);el.innerHTML=refs.map(r=>'<div class="ref-chip" title="Reference image"><img src="'+imgUrl(r)+'" alt=""><button type="button" data-ref="'+esc(r.imageId)+'" aria-label="Remove reference">×</button></div>').join('')+'<button type="button" class="sketch-ref" id="openSketch">+ Sketch</button>';$$('[data-ref]').forEach(b=>b.onclick=()=>{refs=refs.filter(r=>r.imageId!==b.dataset.ref);renderRefs()});$('#openSketch').onclick=openSketch}
function addRef(x){if(!x||refs.some(r=>r.imageId===x.imageId))return;refs.push(x);renderRefs();toast('Reference added')}
function batchKey(x){return x.batchId||(x.savedPath.includes('/batches/')?x.savedPath.split('/batches/')[1]?.split('/')[0]:'')||''}
function passes(x){if(filter==='batch'&&!batchKey(x))return false;if(filter==='tmp'&&!x.savedPath.startsWith('/tmp/'))return false;return true}
function visibleImages(){return images.filter(passes)}
function columnCount(){const w=$('#wall')?.clientWidth||window.innerWidth;return Math.max(1,Math.min(6,Math.floor(w/292)||1))}
function tileHtml(x){return '<div class="tile" role="button" tabindex="0" aria-label="Open image" data-id="'+esc(x.imageId)+'"><img src="'+imgUrl(x)+'" loading="lazy" alt=""></div>'}
function bindTiles(){$$('.tile').forEach(t=>{t.onclick=()=>select(t.dataset.id);t.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select(t.dataset.id)}}})}
function emptyHtml(blank){return '<div class="empty">'+
  '<div class="mark">'+(blank?'¶':'∅')+'</div>'+
  '<h2>'+(blank?'A blank canvas.':'Nothing in this view.')+'</h2>'+
  '<p>'+(blank?'Type a prompt below. Generations land here — full and uncropped.':'Switch the filter to see other generations.')+'</p>'+
  (blank?'<div class="kbd"><kbd>/</kbd> focus prompt &nbsp;·&nbsp; <kbd>↩</kbd> generate</div>':'')+
'</div>'}
function renderWall(visible){
  const cols=columnCount();
  $('#wall').className='wall';
  $('#wall').style.setProperty('--cols',String(cols));
  const buckets=Array.from({length:cols},()=>[]);
  visible.forEach((x,i)=>buckets[i%cols].push(x));
  $('#wall').innerHTML=buckets.map(col=>'<div class="mcol">'+col.map(tileHtml).join('')+'</div>').join('');
  bindTiles();
}
function renderBatches(visible){
  $('#wall').className='batch-list';
  $('#wall').style.removeProperty('--cols');
  const groups=new Map();
  visible.filter(x=>batchKey(x)).forEach(x=>{const key=batchKey(x);if(!groups.has(key))groups.set(key,[]);groups.get(key).push(x)});
  const sorted=[...groups.entries()].map(([key,items])=>({key,items,title:items[0]?.batchPrompt||key,date:items.map(x=>x.createdAt).sort().at(-1)||''})).sort((a,b)=>b.date.localeCompare(a.date));
  if(!sorted.length){$('#wall').innerHTML=emptyHtml(false);return}
  $('#wall').innerHTML=sorted.map(g=>{
    const items=g.items.slice().sort((a,b)=>(a.batchIndex||0)-(b.batchIndex||0)||a.createdAt.localeCompare(b.createdAt));
    const date=g.date?new Date(g.date).toLocaleString([],{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}):'';
    const preview=items.slice(0,4);
    return '<section class="batch-card"><div class="batch-preview">'+preview.map(tileHtml).join('')+'</div><div class="batch-info"><div class="batch-title">'+esc(g.title)+'</div><div class="batch-meta">'+pad(items.length)+' plates'+(date?' · '+esc(date):'')+'</div></div></section>'
  }).join('');
  bindTiles();
}

function render(){
  const visible=visibleImages();
  $('#count').textContent=pad(visible.length)+' '+(visible.length===1?'plate':'plates');
  if(!visible.length){
    $('#wall').className='wall';
    $('#wall').style.removeProperty('--cols');
    $('#wall').innerHTML=emptyHtml(!images.length);
    return;
  }
  if(filter==='batch') renderBatches(visible); else renderWall(visible);
}

async function load(){
  try{const data=await api('/api/images');images=data.images||[];render()}
  catch(e){$('#wall').innerHTML='<div class="empty"><div class="mark">!</div><h2>Couldn\\'t load images</h2><p>'+esc(e.message)+'</p></div>'}
}

function select(id){selected=images.find(x=>x.imageId===id);openModal()}

function openModal(){
  const x=selected;if(!x)return;
  const m=$('#modal');
  if(!m.classList.contains('open')){
    m.classList.add('open');m.setAttribute('aria-hidden','false');
    m.innerHTML='<div class="modal-counter" id="mc"></div>'+
      '<button class="close" data-close aria-label="Close">×</button>'+
      '<button class="nav prev" data-nav="prev" aria-label="Previous">‹</button>'+
      '<button class="nav next" data-nav="next" aria-label="Next">›</button>'+
      '<img class="modalImg" id="mi" alt="">'+
      '<div class="modal-bar">'+
        '<button data-act="vary">Vary</button>'+
        '<button data-act="rerun">Rerun</button>'+
        '<button data-act="ref">Use ref</button>'+
        '<button data-act="copyprompt">Copy prompt</button>'+
        '<button data-act="open">Open</button>'+
        '<button data-act="reveal">Reveal</button>'+
      '</div>';
    $('[data-close]').onclick=closeModal;
    $$('[data-act]').forEach(b=>b.onclick=e=>{e.stopPropagation();act(b.dataset.act)});
    $$('[data-nav]').forEach(b=>b.onclick=e=>{e.stopPropagation();move(b.dataset.nav==='next'?1:-1)});
    m.onclick=e=>{if(e.target.id==='modal')closeModal()};
  }
  const v=visibleImages(),idx=v.findIndex(y=>y.imageId===x.imageId);
  $('#mc').textContent=pad(idx+1)+' — '+pad(v.length);
  $('#mi').src=imgUrl(x);$('#mi').alt=x.prompt||'';
}
function closeModal(){selected=null;const m=$('#modal');m.classList.remove('open');m.setAttribute('aria-hidden','true');m.innerHTML=''}
function move(delta){const v=visibleImages();if(!selected||!v.length)return;const i=v.findIndex(x=>x.imageId===selected.imageId);selected=v[(i+delta+v.length)%v.length];openModal()}

function selectedPrompt(){return selected?.batchPrompt || selected?.prompt || ''}
function setComposerPrompt(text){ta.value=text;autosize();ta.focus();closeModal()}
async function generateFromSelected(prompt,countOverride=count){
  await api('/api/generate',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({prompt,style:$('#style').value,size:$('#size').value,quality:$('#quality').value,thinking:$('#thinking').value,count:countOverride,references:refs.map(r=>r.imageId)})});
}
async function act(a){
  if(!selected)return;
  if(a==='copyprompt'){await navigator.clipboard.writeText(selectedPrompt());return toast('Prompt copied')}
  if(a==='vary')return setComposerPrompt(selectedPrompt()+'\\n\\nVariation: ')
  if(a==='rerun'){await generateFromSelected(selectedPrompt());return toast('Rerunning')}
  if(a==='ref'){addRef(selected);closeModal();ta.focus();return}
  if(a==='copypath'){await navigator.clipboard.writeText(selected.savedPath);return toast('Path copied')}
  await api('/api/'+(a==='attach'?'insert':a),{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({imageId:selected.imageId})});
  toast(a==='attach'?'Attached':a);
}

const ta=$('#prompt');
function autosize(){ta.style.height='auto';ta.style.height=Math.min(ta.scrollHeight,160)+'px'}
ta.addEventListener('input',autosize);
ta.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();$('#composer').requestSubmit()}});

document.addEventListener('keydown',e=>{
  if(e.key==='/' && document.activeElement!==ta && !$('#modal').classList.contains('open')){
    e.preventDefault();ta.focus();
  }
});

$('#counts').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;$$('#counts button').forEach(x=>x.classList.toggle('active',x===b));count=Number(b.dataset.n)||1});

$('#composer').onsubmit=async e=>{
  e.preventDefault();
  const prompt=ta.value.trim();if(!prompt && refs.length===0)return toast('Add a prompt or reference');
  const btn=$('#generate'),lbl=btn.querySelector('span:first-child');
  btn.disabled=true;if(lbl)lbl.textContent='Generating';
  const wrap=$('#status-wrap'),statusEl=$('#status');
  wrap.classList.remove('live');wrap.classList.add('busy');statusEl.textContent='Generating';
  try{
    await api('/api/generate',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({prompt,style:$('#style').value,size:$('#size').value,quality:$('#quality').value,thinking:$('#thinking').value,count,references:refs.map(r=>r.imageId)})});
    toast('Generation started');
  }catch(err){toast(err.message)}
  finally{btn.disabled=false;if(lbl)lbl.textContent='Generate';wrap.classList.remove('busy')}
};

$$('.filters [data-filter]').forEach(b=>b.onclick=()=>{$$('.filters [data-filter]').forEach(x=>x.classList.remove('active'));b.classList.add('active');filter=b.dataset.filter;render()});

document.addEventListener('keydown',e=>{
  if($('#modal').classList.contains('open')){
    if(e.key==='Escape')closeModal();
    if(e.key==='ArrowRight')move(1);
    if(e.key==='ArrowLeft')move(-1);
  }
});

const events=new EventSource('/events?token='+encodeURIComponent(TOKEN));
const statusEl=$('#status'),wrap=$('#status-wrap');
events.addEventListener('ready',()=>{statusEl.textContent='Live';wrap.classList.add('live');wrap.classList.remove('busy')});
events.addEventListener('imagegen:generated',()=>{statusEl.textContent='Updated '+new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});wrap.classList.add('live');wrap.classList.remove('busy');load()});
events.addEventListener('generation:start',e=>{statusEl.textContent=JSON.parse(e.data).message;wrap.classList.add('busy');wrap.classList.remove('live')});
events.onerror=()=>{statusEl.textContent='Disconnected';wrap.classList.remove('live');wrap.classList.remove('busy')};

renderRefs();
const sketchModal=$('#sketchModal'),sketchCanvas=$('#sketchCanvas'),sketchCtx=sketchCanvas.getContext('2d');let sketchTool='brush',drawing=false,last=null;
function initSketch(){sketchCtx.fillStyle='white';sketchCtx.fillRect(0,0,sketchCanvas.width,sketchCanvas.height);sketchCtx.lineCap='round';sketchCtx.lineJoin='round'}
function openSketch(){sketchModal.classList.add('open');sketchModal.setAttribute('aria-hidden','false')}
function closeSketch(){sketchModal.classList.remove('open');sketchModal.setAttribute('aria-hidden','true')}
function sketchPoint(e){const r=sketchCanvas.getBoundingClientRect();return {x:(e.clientX-r.left)*sketchCanvas.width/r.width,y:(e.clientY-r.top)*sketchCanvas.height/r.height}}
function sketchLine(a,b){sketchCtx.globalCompositeOperation=sketchTool==='eraser'?'destination-out':'source-over';sketchCtx.strokeStyle='black';sketchCtx.lineWidth=Number($('#sketchSize').value)||8;sketchCtx.beginPath();sketchCtx.moveTo(a.x,a.y);sketchCtx.lineTo(b.x,b.y);sketchCtx.stroke()}
sketchCanvas.addEventListener('pointerdown',e=>{drawing=true;last=sketchPoint(e);sketchCanvas.setPointerCapture(e.pointerId)})
sketchCanvas.addEventListener('pointermove',e=>{if(!drawing)return;const p=sketchPoint(e);sketchLine(last,p);last=p})
sketchCanvas.addEventListener('pointerup',()=>{drawing=false;last=null})
$('#sketchBrush').onclick=()=>{sketchTool='brush';$('#sketchBrush').classList.add('active');$('#sketchEraser').classList.remove('active')}
$('#sketchEraser').onclick=()=>{sketchTool='eraser';$('#sketchEraser').classList.add('active');$('#sketchBrush').classList.remove('active')}
$('#sketchClear').onclick=initSketch;$('#sketchClose').onclick=closeSketch;
$('#sketchUse').onclick=async()=>{const blob=await new Promise(r=>sketchCanvas.toBlob(r,'image/png'));const res=await api('/api/sketch',{method:'POST',headers:{'content-type':'image/png'},body:blob});addRef(res.metadata);closeSketch();initSketch()};
initSketch();
load();
</script>
</body>
</html>`;
}

export default function imagegen(pi: ExtensionAPI) {
	let studioServer: Server | undefined;
	let studioBaseUrl: string | undefined;
	let studioToken = randomUUID();
	let lastCtx: ExtensionContext | undefined;
	const studioEventClients = new Set<ServerResponse>();

	function broadcastStudioEvent(event: string, data: unknown) {
		const payload = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
		for (const client of studioEventClients) {
			if (!client.destroyed) client.write(payload);
		}
	}

	function handleStudioEvents(req: IncomingMessage, res: ServerResponse) {
		res.writeHead(200, {
			"Content-Type": "text/event-stream; charset=utf-8",
			"Cache-Control": "no-cache, no-store, must-revalidate",
			Connection: "keep-alive",
		});
		res.write("event: ready\ndata: {}\n\n");
		studioEventClients.add(res);
		if (lastCtx?.hasUI) lastCtx.ui.setStatus("img", "img: studio open");
		const ping = setInterval(() => {
			if (!res.destroyed) res.write(": ping\n\n");
		}, 15_000);
		req.on("close", () => {
			clearInterval(ping);
			studioEventClients.delete(res);
			if (studioEventClients.size === 0 && lastCtx?.hasUI) lastCtx.ui.setStatus("img", undefined);
		});
	}

	function getRequestOrigin(req: IncomingMessage): string {
		const host = req.headers.host ?? "127.0.0.1";
		return `http://${host}`;
	}

	function isLocalStudioRequest(req: IncomingMessage): boolean {
		const host = req.headers.host ?? "";
		return host.startsWith("127.0.0.1:") || host.startsWith("localhost:") || host === "127.0.0.1" || host === "localhost";
	}

	async function handleStudioRequest(req: IncomingMessage, res: ServerResponse) {
		const url = new URL(req.url ?? "/", getRequestOrigin(req));
		if (url.pathname !== "/favicon.ico" && !isLocalStudioRequest(req)) {
			writeText(res, 403, "Forbidden");
			return;
		}
		if (url.pathname === "/favicon.ico") return writeText(res, 204, "");
		if (req.method === "GET" && (url.pathname === "/" || url.pathname === "/studio")) return writeHtml(res, renderStudioPage(studioToken));
		if (req.method === "GET" && url.pathname === "/events") return handleStudioEvents(req, res);
		if (req.method === "GET" && url.pathname === "/api/images") return writeJson(res, 200, { images: await readRecentMetadata(1000) });
		if (req.method === "POST" && url.pathname === "/api/sketch") {
			const body = await readRequestBody(req, 20 * 1024 * 1024);
			if (!isPng(body)) return writeJson(res, 415, { ok: false, error: "Expected PNG sketch upload." });
			const imageId = `sketch_${randomUUID()}`;
			const stamp = new Date().toISOString().replace(/[:.]/g, "-");
			const savedPath = join(getAgentDir(), "generated-images", "sketches", `${stamp}-${imageId}.png`);
			await withFileMutationQueue(savedPath, async () => {
				await mkdir(dirname(savedPath), { recursive: true });
				await writeFile(savedPath, body);
			});
			const metadataPath = metadataPathForImage(savedPath);
			const metadata: ImagegenMetadata = {
				createdAt: new Date().toISOString(),
				prompt: "Sketch reference",
				provider: "local-sketch",
				responseModel: "none",
				imageModel: "canvas",
				imageId,
				savedPath,
				metadataPath,
				mimeType: "image/png",
				size: "1024x1024",
				quality: "n/a",
				background: "opaque",
				outputFormat: "png",
				thinking: "off",
				kind: "sketch",
			};
			await saveMetadata(metadata);
			broadcastStudioEvent("imagegen:generated", metadata);
			return writeJson(res, 200, { ok: true, metadata });
		}
		const imageMatch = url.pathname.match(/^\/api\/image\/([^/]+)$/);
		if (req.method === "GET" && imageMatch) {
			const metadata = await findMetadataByImageId(decodeURIComponent(imageMatch[1]!));
			if (!metadata) return writeText(res, 404, "Image not found");
			try {
				await stat(metadata.savedPath);
				res.writeHead(200, { "Content-Type": metadata.mimeType || mimeFromFormat(metadata.outputFormat), "Cache-Control": "no-cache" });
				res.end(await readFile(metadata.savedPath));
			} catch {
				writeText(res, 404, "Image file not found");
			}
			return;
		}
		if (req.method === "POST" && url.pathname === "/api/generate") {
			if (!lastCtx) return writeJson(res, 400, { ok: false, error: "No active Pi context." });
			const body = await readJsonBody(req);
			const prompt = String(body.prompt ?? "").trim();
			const count = Math.min(Math.max(Number.parseInt(String(body.count ?? "1"), 10) || 1, 1), 12);
			const style = String(body.style ?? "").trim();
			if (style && !STYLE_PRESETS[style]) return writeJson(res, 400, { ok: false, error: `Unknown style: ${style}` });
			const thinking = String(body.thinking ?? "low");
			if (!THINKING_MODES.includes(thinking as (typeof THINKING_MODES)[number])) {
				return writeJson(res, 400, { ok: false, error: `Unknown thinking mode: ${thinking}` });
			}
			const referenceIds = Array.isArray(body.references) ? body.references.map(String).slice(0, 8) : [];
			const references = (await Promise.all(referenceIds.map((id) => findMetadataByImageId(id)))).filter(Boolean) as ImagegenMetadata[];
			if (!prompt && references.length === 0) return writeJson(res, 400, { ok: false, error: "Prompt or reference image is required." });
			const basePrompt = prompt || "Create a new image using the provided reference image(s) for visual style, subject, and composition.";
			const referencePrompt = basePrompt;
			const referenceMetadata = references.length
				? { referenceImageIds: references.map((item) => item.imageId), referencePaths: references.map((item) => item.savedPath) }
				: {};
			const options = {
				style: style || undefined,
				size: String(body.size ?? "auto") as ToolParams["size"],
				quality: String(body.quality ?? "auto") as ToolParams["quality"],
				thinking: thinking as ToolParams["thinking"],
				referencePaths: references.map((item) => item.savedPath),
			};
			const results: ImagegenDetails[] = [];
			if (count === 1) {
				broadcastStudioEvent("generation:start", { message: "Generating image…" });
				const { details } = await generateImage(applyStyle(referencePrompt, options), lastCtx.signal, undefined, lastCtx, referenceMetadata);
				pi.events.emit("imagegen:generated", details);
				broadcastStudioEvent("imagegen:generated", details);
				results.push(details);
			} else {
				const batchId = batchDirName(prompt || `reference-${references.map((item) => item.imageId.slice(-6)).join("-")}`);
				const batchDir = join(getAgentDir(), "generated-images", "batches", batchId);
				for (let index = 0; index < count; index++) {
					broadcastStudioEvent("generation:start", { message: `Generating ${index + 1}/${count}…` });
					const params = applyStyle(`${referencePrompt}. Variation ${index + 1} of ${count}; make this composition distinct from the others.`, {
						...options,
						outputPath: join(batchDir, `${String(index + 1).padStart(2, "0")}.png`),
					});
					const { details } = await generateImage(params, lastCtx.signal, undefined, lastCtx, {
						batchId,
						batchPrompt: prompt || "Reference-only generation",
						batchIndex: index + 1,
						batchCount: count,
						...referenceMetadata,
					});
					pi.events.emit("imagegen:generated", details);
					broadcastStudioEvent("imagegen:generated", details);
					results.push(details);
				}
				await mkdir(batchDir, { recursive: true });
				await writeFile(join(batchDir, "batch.json"), JSON.stringify({ createdAt: new Date().toISOString(), prompt: prompt || "Reference-only generation", references: referenceMetadata, count, images: results }, null, 2), "utf8");
			}
			return writeJson(res, 200, { ok: true, images: results });
		}

		if (req.method === "POST" && ["/api/open", "/api/reveal", "/api/insert"].includes(url.pathname)) {
			const body = await readJsonBody(req);
			const metadata = await findMetadataByImageId(String(body.imageId ?? ""));
			if (!metadata) return writeJson(res, 404, { ok: false, error: "Image not found" });
			if (url.pathname === "/api/open") macOpen(metadata.savedPath);
			if (url.pathname === "/api/reveal") macOpen(metadata.savedPath, true);
			if (url.pathname === "/api/insert") insertImageIntoPrompt(metadata.savedPath, lastCtx);
			return writeJson(res, 200, { ok: true });
		}
		writeText(res, 404, "Not found");
	}

	async function ensureStudioServer(): Promise<string> {
		if (studioServer && studioBaseUrl) return studioBaseUrl;
		studioToken = randomUUID();
		studioServer = createServer((req, res) => void handleStudioRequest(req, res).catch((error) => writeJson(res, 500, { ok: false, error: error instanceof Error ? error.message : String(error) })));
		await new Promise<void>((resolve, reject) => {
			studioServer!.once("error", reject);
			studioServer!.listen(0, "127.0.0.1", () => resolve());
		});
		const address = studioServer.address();
		if (!address || typeof address === "string") throw new Error("Could not determine studio server port.");
		studioBaseUrl = `http://127.0.0.1:${address.port}`;
		return studioBaseUrl;
	}

	async function openStudio(ctx: ExtensionContext) {
		lastCtx = ctx;
		const base = await ensureStudioServer();
		const url = `${base}/studio?token=${encodeURIComponent(studioToken)}`;
		await openBrowser(url);
		ctx.ui.notify("Image studio opened.", "info");
	}

	pi.on("session_start", (_event, ctx) => {
		lastCtx = ctx;
	});

	pi.on("session_shutdown", async () => {
		for (const client of studioEventClients) client.end();
		studioEventClients.clear();
		if (studioServer) await new Promise<void>((resolve) => studioServer!.close(() => resolve()));
		studioServer = undefined;
		studioBaseUrl = undefined;
		lastCtx = undefined;
	});

	pi.registerMessageRenderer("imagegen-result", (message, _options, theme) => {
		const details = message.details as ImagegenMetadata | undefined;
		const lines = [
			theme.fg("success", "✓ Image generated"),
			details?.savedPath ? theme.fg("muted", details.savedPath) : message.content,
			details?.metadataPath ? theme.fg("dim", `metadata: ${details.metadataPath}`) : undefined,
		].filter(Boolean) as string[];
		return new Text(lines.join("\n"), 0, 0);
	});

	pi.registerTool({
		name: "imagegen",
		label: "Imagegen",
		description:
			"Generate an image using OpenAI Codex/ChatGPT subscription image generation (gpt-image-2). Returns an image attachment and saves it to disk.",
		promptSnippet: "Generate images via OpenAI Codex/ChatGPT subscription image generation",
		promptGuidelines: [
			"Use imagegen when the user asks to create, generate, draw, render, or make an image.",
			"Use imagegen instead of writing image-generation API code when the user wants an actual generated image.",
		],
		parameters: TOOL_PARAMS,

		async execute(_toolCallId, params: ToolParams, signal, onUpdate, ctx) {
			const { image, text, details } = await generateImage(params, signal, onUpdate as ToolUpdate | undefined, ctx);
			pi.events.emit("imagegen:generated", details);
			broadcastStudioEvent("imagegen:generated", details);

			return {
				content: [
					{ type: "text", text },
					{ type: "image", data: image.base64, mimeType: details.mimeType },
				],
				details,
			};
		},

		renderResult(result, _options, theme) {
			const details = result.details as ImagegenDetails | undefined;
			if (!details) {
				const first = result.content[0];
				return new Text(first?.type === "text" ? first.text : "Generated image", 0, 0);
			}
			const lines = [
				theme.fg("success", `✓ Generated image via ${details.imageModel}`),
				theme.fg("muted", details.savedPath),
				details.revisedPrompt ? theme.fg("dim", `Prompt: ${details.revisedPrompt}`) : undefined,
			].filter(Boolean) as string[];
			return new Text(lines.join("\n"), 0, 0);
		},
	});


	pi.registerCommand("img", {
		description: "Image workflows: /img gen|list|open|reveal|path|info ...",
		handler: async (args, ctx) => {
			const input = args.trim();
			const [subcommandRaw = "list", ...restParts] = input.split(/\s+/);
			const subcommand = subcommandRaw.toLowerCase();
			const rest = restParts.join(" ").trim();

			if (["help", "-h", "--help"].includes(subcommand)) {
				pi.sendMessage({
					customType: "imagegen-result",
					content: [
						"Image commands:",
						"/img gen [--thinking off|minimal|low|medium|high] [--style name] <prompt>",
						"/img batch <count> [--thinking off|minimal|low|medium|high] [--style name] <prompt>",
						"/img styles",
						"/img studio",
						"/img list [count]",
						"/img open [latest|number|path]",
						"/img reveal [latest|number|path]",
						"/img path [latest|number|path]",
						"/img info [latest|number|path]",
					].join("\n"),
					display: true,
				});
				return;
			}

			if (["studio", "gallery", "browse"].includes(subcommand)) {
				await openStudio(ctx);
				return;
			}

			if (["styles", "style"].includes(subcommand)) {
				const lines = Object.entries(STYLE_PRESETS).map(([name, preset]) => {
					return `${name}\n   size=${preset.size ?? "auto"}, quality=${preset.quality ?? "auto"}\n   ${preset.suffix}`;
				});
				pi.sendMessage({ customType: "imagegen-result", content: `Image styles:\n\n${lines.join("\n\n")}`, display: true });
				return;
			}

			if (["gen", "generate", "create"].includes(subcommand)) {
				const parsed = parseImgArgs(rest);
				const prompt = parsed.positional.join(" ").trim();
				if (!prompt) {
					ctx.ui.notify("Usage: /img gen [--style name] <prompt>", "warning");
					return;
				}
				if (parsed.options.style && !STYLE_PRESETS[parsed.options.style]) {
					ctx.ui.notify(`Unknown style '${parsed.options.style}'. Try /img styles.`, "warning");
					return;
				}
				if (parsed.options.thinking && !THINKING_MODES.includes(parsed.options.thinking)) {
					ctx.ui.notify(`Unknown thinking mode '${parsed.options.thinking}'.`, "warning");
					return;
				}
				ctx.ui.notify("Generating image...", "info");
				const { details } = await generateImage(applyStyle(prompt, parsed.options), ctx.signal, undefined, ctx);
				pi.events.emit("imagegen:generated", details);
				broadcastStudioEvent("imagegen:generated", details);
				ctx.ui.notify(`Generated image: ${details.savedPath}`, "success");
				pi.sendMessage({
					customType: "imagegen-result",
					content: `Generated image: ${details.savedPath}`,
					display: true,
					details,
				});
				return;
			}

			if (["batch", "variants"].includes(subcommand)) {
				const [countRaw = "", ...batchRestParts] = rest.split(/\s+/);
				const count = Math.min(Math.max(Number.parseInt(countRaw, 10) || 0, 1), 12);
				const parsed = parseImgArgs(batchRestParts.join(" "));
				const prompt = parsed.positional.join(" ").trim();
				if (!count || !prompt) {
					ctx.ui.notify("Usage: /img batch <count> [--style name] <prompt>", "warning");
					return;
				}
				if (parsed.options.style && !STYLE_PRESETS[parsed.options.style]) {
					ctx.ui.notify(`Unknown style '${parsed.options.style}'. Try /img styles.`, "warning");
					return;
				}
				if (parsed.options.thinking && !THINKING_MODES.includes(parsed.options.thinking)) {
					ctx.ui.notify(`Unknown thinking mode '${parsed.options.thinking}'.`, "warning");
					return;
				}
				const batchId = batchDirName(prompt);
				const batchDir = join(getAgentDir(), "generated-images", "batches", batchId);
				const results: ImagegenDetails[] = [];
				for (let index = 0; index < count; index++) {
					ctx.ui.notify(`Generating image ${index + 1}/${count}...`, "info");
					const params = applyStyle(`${prompt}. Variation ${index + 1} of ${count}; make this composition distinct from the others.`, {
						...parsed.options,
						outputPath: join(batchDir, `${String(index + 1).padStart(2, "0")}.png`),
					});
					const { details } = await generateImage(params, ctx.signal, undefined, ctx, {
						batchId,
						batchPrompt: prompt,
						batchIndex: index + 1,
						batchCount: count,
					});
					pi.events.emit("imagegen:generated", details);
					broadcastStudioEvent("imagegen:generated", details);
					results.push(details);
				}
				const batchPath = join(batchDir, "batch.json");
				await mkdir(batchDir, { recursive: true });
				await writeFile(batchPath, JSON.stringify({ createdAt: new Date().toISOString(), prompt, count, images: results }, null, 2), "utf8");
				pi.sendMessage({
					customType: "imagegen-result",
					content: [`Generated ${results.length} images:`, ...results.map((item, i) => `${i + 1}. ${item.savedPath}`), `Batch: ${batchPath}`].join("\n"),
					display: true,
					details: { batchPath, images: results },
				});
				ctx.ui.notify(`Generated batch: ${batchDir}`, "success");
				return;
			}

			if (["list", "ls", "recent"].includes(subcommand)) {
				const limit = Math.min(Math.max(Number.parseInt(rest || "10", 10) || 10, 1), 50);
				const recent = await readRecentMetadata(limit);
				if (recent.length === 0) {
					ctx.ui.notify("No imagegen outputs found.", "info");
					return;
				}
				const lines = recent.map((item, index) => {
					const prompt = item.prompt.length > 90 ? `${item.prompt.slice(0, 87)}...` : item.prompt;
					return `${index + 1}. ${basename(item.savedPath)}\n   ${item.savedPath}\n   ${prompt}`;
				});
				pi.sendMessage({
					customType: "imagegen-result",
					content: `Recent generated images:\n\n${lines.join("\n\n")}`,
					display: true,
					details: { recent },
				});
				return;
			}

			if (["open", "reveal", "path", "info"].includes(subcommand)) {
				const path = await resolveImageTarget(rest, ctx.cwd);
				if (!path) {
					ctx.ui.notify("No generated image found. Try /img list first.", "warning");
					return;
				}

				if (subcommand === "open") {
					macOpen(path);
					ctx.ui.notify(`Opened ${path}`, "success");
					return;
				}

				if (subcommand === "reveal") {
					macOpen(path, true);
					ctx.ui.notify(`Revealed ${path}`, "success");
					return;
				}

				if (subcommand === "path") {
					pi.sendMessage({
						customType: "imagegen-result",
						content: path,
						display: true,
						details: { savedPath: path },
					});
					return;
				}

				const metaPath = metadataPathForImage(path);
				let content = `Image: ${path}`;
				let details: unknown = { savedPath: path };
				try {
					const metadata = JSON.parse(await readFile(metaPath, "utf8")) as ImagegenMetadata;
					content = JSON.stringify(metadata, null, 2);
					details = metadata;
				} catch {
					content = `Image: ${path}\nNo metadata sidecar found at ${metaPath}`;
				}
				pi.sendMessage({ customType: "imagegen-result", content, display: true, details });
				return;
			}

			ctx.ui.notify(`Unknown /img subcommand: ${subcommand}. Try /img help.`, "warning");
		},
	});
}
