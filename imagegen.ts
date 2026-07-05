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

function spawnDetached(command: string, args: string[]): Promise<void> {
	return new Promise((resolve, reject) => {
		const child = spawn(command, args, { detached: true, stdio: "ignore", windowsHide: true });
		child.once("error", reject);
		child.once("spawn", () => {
			child.unref();
			resolve();
		});
	});
}

async function openPath(targetPath: string): Promise<void> {
	if (process.platform === "darwin") return spawnDetached("open", [targetPath]);
	if (process.platform === "win32") return spawnDetached("cmd", ["/c", "start", "", targetPath]);
	return spawnDetached("xdg-open", [targetPath]);
}

async function revealPath(targetPath: string): Promise<void> {
	if (process.platform === "darwin") return spawnDetached("open", ["-R", targetPath]);
	if (process.platform === "win32") return spawnDetached("cmd", ["/c", "start", "", "explorer.exe", `/select,\"${targetPath}\"`]);
	return spawnDetached("xdg-open", [dirname(targetPath)]);
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
	if (process.platform === "darwin") return spawnDetached("open", [url]);
	if (process.platform === "win32") return spawnDetached("cmd", ["/c", "start", "", url]);
	return spawnDetached("xdg-open", [url]);
}

function renderStudioPage(token: string): string {
	return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Pi Image Studio</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&family=Newsreader:ital,opsz,wght@1,6..72,400;1,6..72,500&display=swap" rel="stylesheet">
<style>
/*
  Pi Image Studio — "Daylight Gallery"
  ------------------------------------
  A bright, quiet room for images. The chrome stays out of the way:
  warm paper surface, hairline borders, one cobalt signal color.
  Geist for UI, Geist Mono for numerics, Newsreader italic as the
  prompt voice. The work hangs uncropped on the wall; the composer
  floats beneath it like a caption card.
*/
:root{
  color-scheme:light;
  --bg:#f5f5f1;
  --bg-2:#ecece6;
  --card:#ffffff;
  --ink:#161511;
  --ink-2:#3d3b34;
  --ink-3:#6f6c62;
  --muted:#98948a;
  --hair:rgba(22,21,17,0.08);
  --hair-2:rgba(22,21,17,0.14);
  --accent:#2b4bdf;
  --accent-soft:rgba(43,75,223,0.10);
  --live:#1f7a4d;
  --shadow-soft:0 1px 2px rgba(22,21,17,0.04),0 8px 24px -12px rgba(22,21,17,0.10);
  --shadow-float:0 2px 6px rgba(22,21,17,0.05),0 24px 64px -24px rgba(22,21,17,0.22);
  --shadow-modal:0 40px 120px -24px rgba(22,21,17,0.35);
  --sans:"Geist",-apple-system,"SF Pro Text","Helvetica Neue",system-ui,sans-serif;
  --serif:"Newsreader","Iowan Old Style","Hoefler Text",Georgia,serif;
  --mono:"Geist Mono","SF Mono",ui-monospace,Menlo,monospace;
}
*{box-sizing:border-box}
html,body{height:100%;margin:0}
body{
  background:var(--bg);
  color:var(--ink);
  font:13px/1.5 var(--sans);
  font-feature-settings:"tnum";
  -webkit-font-smoothing:antialiased;
  overflow:hidden;
}
body::before{
  content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
  background:
    radial-gradient(1000px 600px at 12% -6%, rgba(255,224,180,0.20), transparent 60%),
    radial-gradient(1100px 700px at 100% 104%, rgba(184,201,255,0.16), transparent 62%);
}
::selection{background:var(--ink);color:var(--bg)}
button{font:inherit}

.studio{height:100vh;display:flex;flex-direction:column;position:relative;z-index:1}

/* ── top bar ── */
.top{
  height:54px;flex:none;
  display:grid;grid-template-columns:1fr auto 1fr;align-items:center;
  padding:0 24px;
  border-bottom:1px solid var(--hair);
  background:rgba(245,245,241,0.78);
  backdrop-filter:blur(12px) saturate(1.05);
  -webkit-backdrop-filter:blur(12px) saturate(1.05);
  position:relative;z-index:4;
}
.brand{display:flex;align-items:baseline;gap:9px}
.brand .glyph{
  width:20px;height:20px;display:grid;place-items:center;align-self:center;
  background:var(--ink);color:var(--bg);
  font-family:var(--serif);font-style:italic;font-size:13px;
  border-radius:6px;
}
.brand .name{font-weight:600;font-size:13px;letter-spacing:-0.01em}
.brand .sub{font-family:var(--mono);font-size:10px;color:var(--muted);letter-spacing:0.08em}
.seg{
  justify-self:center;display:flex;gap:2px;
  background:var(--bg-2);border:1px solid var(--hair);border-radius:999px;padding:3px;
}
.seg button{
  all:unset;cursor:pointer;
  font-family:var(--sans);font-size:12px;font-weight:500;
  color:var(--ink-3);padding:4px 14px;border-radius:999px;
  transition:color .15s,background .15s,box-shadow .15s;
}
.seg button:hover{color:var(--ink)}
.seg button.active{color:var(--ink);background:var(--card);box-shadow:0 1px 2px rgba(22,21,17,0.08)}
.right{justify-self:end;display:flex;align-items:center;gap:14px}
.count{font-family:var(--mono);font-size:11px;color:var(--muted);font-variant-numeric:tabular-nums}
.status{
  display:inline-flex;align-items:center;gap:7px;
  font-family:var(--mono);font-size:10.5px;letter-spacing:0.06em;color:var(--ink-3);
  padding:5px 11px;border:1px solid var(--hair);border-radius:999px;background:var(--card);
}
.dot{width:6px;height:6px;border-radius:50%;background:#c9c5ba;flex:none}
.status.live .dot{background:var(--live);animation:beat 2s ease-out infinite}
.status.busy .dot{background:var(--accent);animation:beat 1s ease-out infinite}
@keyframes beat{
  0%{box-shadow:0 0 0 0 currentColor;opacity:1}
  70%{box-shadow:0 0 0 6px rgba(0,0,0,0);opacity:.85}
  100%{box-shadow:0 0 0 0 rgba(0,0,0,0);opacity:1}
}

/* ── wall ── */
.canvas{flex:1;overflow:auto;padding:28px 28px 200px;scroll-behavior:smooth}
.wall{max-width:1560px;margin:0 auto;display:grid;grid-template-columns:repeat(var(--cols,4),minmax(0,1fr));gap:14px;align-items:start}
.mcol{display:flex;flex-direction:column;gap:14px;min-width:0}
.tile{position:relative;margin:0;line-height:0;cursor:pointer;border-radius:12px;overflow:hidden;
  box-shadow:0 0 0 1px var(--hair),var(--shadow-soft);
  transition:transform .18s cubic-bezier(.2,.7,.2,1),box-shadow .18s;
  animation:rise .32s cubic-bezier(.2,.7,.2,1) backwards;
}
@keyframes rise{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.tile:hover{transform:translateY(-2px);box-shadow:0 0 0 1px var(--hair-2),var(--shadow-float)}
.tile:focus-visible{outline:none;box-shadow:0 0 0 2px var(--accent),var(--shadow-soft)}
.tile img{display:block;width:100%;height:auto;background:var(--bg-2)}
.tile .add{
  position:absolute;top:8px;right:8px;
  width:26px;height:26px;border:0;border-radius:8px;cursor:pointer;
  display:grid;place-items:center;
  background:rgba(255,255,255,0.92);color:var(--ink);
  box-shadow:0 1px 4px rgba(22,21,17,0.18);
  opacity:0;transform:translateY(-3px);
  transition:opacity .15s,transform .15s,background .15s;
}
.tile:hover .add{opacity:1;transform:none}
.tile .add:hover{background:var(--ink);color:var(--bg)}
.tile.ph{aspect-ratio:1;cursor:default;box-shadow:0 0 0 1px var(--hair);
  background:linear-gradient(100deg,var(--bg-2) 40%,#f7f7f3 50%,var(--bg-2) 60%);
  background-size:220% 100%;animation:shimmer 1.4s linear infinite;
}
.tile.ph::after{
  content:"";position:absolute;inset:0;margin:auto;width:8px;height:8px;border-radius:50%;
  background:var(--accent);opacity:.55;animation:beat 1s ease-out infinite;
}
@keyframes shimmer{from{background-position:120% 0}to{background-position:-100% 0}}

/* ── batches ── */
.batch-list{max-width:1560px;margin:0 auto;display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px;align-items:start}
.batch-card{background:var(--card);border:1px solid var(--hair);border-radius:14px;padding:8px;box-shadow:var(--shadow-soft)}
.batch-preview{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;border-radius:8px;overflow:hidden}
.batch-preview .tile{border-radius:6px;animation:none}
.batch-preview .tile img{aspect-ratio:1;object-fit:cover}
.batch-info{display:flex;align-items:baseline;gap:10px;padding:10px 6px 4px;min-width:0}
.batch-title{font-family:var(--serif);font-style:italic;font-size:15.5px;color:var(--ink);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1}
.batch-meta{font-family:var(--mono);font-size:10px;color:var(--muted);white-space:nowrap}

/* ── empty ── */
.empty{max-width:460px;margin:16vh auto 0;text-align:center}
.empty h2{font-family:var(--serif);font-style:italic;font-weight:400;font-size:34px;letter-spacing:-0.01em;margin:0 0 10px;color:var(--ink)}
.empty p{color:var(--muted);margin:0 auto;max-width:320px;line-height:1.6}
.empty .kbd{display:inline-flex;gap:6px;align-items:center;margin-top:16px;font-family:var(--mono);font-size:10.5px;color:var(--ink-3)}
.empty kbd{font:inherit;padding:2px 7px;border:1px solid var(--hair-2);border-bottom-width:2px;border-radius:5px;background:var(--card)}

/* ── composer ── */
.composer{
  position:fixed;left:50%;bottom:24px;transform:translateX(-50%);
  width:min(760px,calc(100vw - 40px));
  background:rgba(255,255,255,0.94);
  border:1px solid var(--hair-2);
  border-radius:18px;
  box-shadow:var(--shadow-float);
  backdrop-filter:blur(20px) saturate(1.1);
  -webkit-backdrop-filter:blur(20px) saturate(1.1);
  z-index:5;padding:14px 14px 10px;
}
.refs{display:none;gap:8px;flex-wrap:wrap;padding:0 4px 10px}
.refs.hasRefs{display:flex}
.ref-chip{position:relative;width:52px;height:40px;border-radius:8px;overflow:hidden;box-shadow:0 0 0 1px var(--hair-2)}
.ref-chip img{width:100%;height:100%;object-fit:cover;display:block}
.ref-chip button{
  position:absolute;right:2px;top:2px;border:0;cursor:pointer;
  width:16px;height:16px;border-radius:999px;font-size:11px;line-height:1;padding:0;
  background:rgba(22,21,17,0.75);color:#fff;display:grid;place-items:center;
}
.promptBox{
  display:block;width:100%;min-height:30px;max-height:150px;resize:none;overflow:auto;
  border:0;outline:0;background:transparent;color:var(--ink);
  font-family:var(--serif);font-style:italic;font-size:19px;line-height:1.45;
  padding:2px 6px 12px;
}
.promptBox::placeholder{color:var(--muted)}
.controls{display:flex;gap:6px;align-items:center;flex-wrap:wrap;border-top:1px solid var(--hair);padding-top:10px}
.pill{
  display:inline-flex;align-items:center;gap:7px;
  padding:0 10px;height:30px;border-radius:9px;
  font-family:var(--mono);font-size:10px;letter-spacing:0.08em;text-transform:uppercase;color:var(--muted);
  transition:background .12s;white-space:nowrap;
}
.pill:hover{background:var(--bg-2)}
.select{
  appearance:none;-webkit-appearance:none;border:0;background:transparent;cursor:pointer;
  color:var(--ink);font-family:var(--mono);font-size:11.5px;text-transform:none;letter-spacing:0;
  padding:0 13px 0 0;outline:none;
  background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='8' height='5' viewBox='0 0 8 5'><path d='M0 0l4 5 4-5z' fill='%236f6c62'/></svg>");
  background-repeat:no-repeat;background-position:right center;
}
.select:focus{color:var(--accent)}
.steps{display:inline-flex;border:1px solid var(--hair-2);border-radius:9px;overflow:hidden;background:var(--card)}
.steps button{
  all:unset;cursor:pointer;width:27px;height:28px;display:inline-grid;place-items:center;
  font-family:var(--mono);font-size:11px;color:var(--ink-2);border-right:1px solid var(--hair);
  font-variant-numeric:tabular-nums;transition:background .12s,color .12s;
}
.steps button:last-child{border-right:0}
.steps button:hover{background:var(--bg-2)}
.steps button.active{background:var(--ink);color:var(--bg)}
.spacer{flex:1;min-width:6px}
.ghost-btn{
  display:inline-flex;align-items:center;gap:7px;cursor:pointer;
  border:1px solid var(--hair-2);background:var(--card);color:var(--ink-2);
  font-size:12px;font-weight:500;height:32px;padding:0 12px;border-radius:10px;
  transition:background .12s,color .12s;
}
.ghost-btn:hover{background:var(--bg-2);color:var(--ink)}
.ghost-btn svg{width:13px;height:13px}
.generate{
  border:0;cursor:pointer;display:inline-flex;align-items:center;gap:9px;
  background:var(--ink);color:var(--bg);
  font-size:12.5px;font-weight:600;height:32px;padding:0 14px;border-radius:10px;
  transition:background .15s,transform .12s;
  box-shadow:0 1px 2px rgba(22,21,17,0.3),0 8px 20px -10px rgba(22,21,17,0.5);
}
.generate:hover{background:#000;transform:translateY(-1px)}
.generate:active{transform:none}
.generate:disabled{opacity:.55;cursor:default;transform:none}
.generate .ret{
  font-family:var(--mono);font-size:10px;display:inline-grid;place-items:center;
  width:17px;height:17px;border-radius:5px;
  background:rgba(255,255,255,0.16);border:1px solid rgba(255,255,255,0.2);
}

/* ── lightbox ── */
.modal{
  position:fixed;inset:0;z-index:8;display:none;flex-direction:column;align-items:center;justify-content:center;
  background:rgba(245,245,241,0.82);
  backdrop-filter:blur(10px) saturate(1.03);
  -webkit-backdrop-filter:blur(10px) saturate(1.03);
}
.modal.open{display:flex;animation:fadeIn .16s ease}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
.modalImg{
  max-width:min(90vw,1360px);max-height:calc(100vh - 220px);
  width:auto;height:auto;object-fit:contain;border-radius:10px;
  box-shadow:var(--shadow-modal);background:var(--bg-2);
  animation:zoom .2s cubic-bezier(.2,.7,.2,1);
}
@keyframes zoom{from{opacity:0;transform:scale(.985)}to{opacity:1;transform:scale(1)}}
.modal-cap{
  max-width:min(84vw,760px);margin-top:16px;text-align:center;
  font-family:var(--serif);font-style:italic;font-size:15px;line-height:1.5;color:var(--ink-2);
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;
}
.modal-sub{margin-top:6px;font-family:var(--mono);font-size:10px;letter-spacing:0.08em;color:var(--muted)}
.modal-counter{
  position:fixed;top:18px;left:50%;transform:translateX(-50%);
  font-family:var(--mono);font-size:10.5px;color:var(--ink-3);font-variant-numeric:tabular-nums;z-index:9;
  padding:5px 12px;background:var(--card);border:1px solid var(--hair);border-radius:999px;
}
.modal-bar{
  position:fixed;left:50%;bottom:24px;transform:translateX(-50%);
  display:flex;align-items:center;background:rgba(255,255,255,0.95);
  border:1px solid var(--hair-2);box-shadow:var(--shadow-float);z-index:9;
  border-radius:12px;overflow:hidden;
}
.modal-bar button{
  all:unset;cursor:pointer;font-size:12px;font-weight:500;color:var(--ink-2);
  padding:10px 15px;border-right:1px solid var(--hair);
  transition:background .12s,color .12s;
}
.modal-bar button:last-child{border-right:0}
.modal-bar button:hover{background:var(--bg-2);color:var(--ink)}
.close{
  position:fixed;right:20px;top:16px;z-index:9;cursor:pointer;
  width:32px;height:32px;display:grid;place-items:center;border-radius:999px;
  border:1px solid var(--hair);background:var(--card);color:var(--ink);font-size:16px;line-height:1;
  transition:transform .15s,background .15s;
}
.close:hover{transform:rotate(90deg);background:var(--bg-2)}
.nav{
  position:fixed;top:50%;transform:translateY(-50%);z-index:9;cursor:pointer;
  width:38px;height:38px;display:grid;place-items:center;border-radius:999px;
  border:1px solid var(--hair);background:var(--card);color:var(--ink-2);
  font-family:var(--serif);font-style:italic;font-size:20px;line-height:1;
  transition:background .15s,transform .15s;
}
.nav:hover{background:var(--bg-2);color:var(--ink)}
.nav.prev{left:20px}
.nav.next{right:20px}

/* ── canvas / sketch ── */
.sk-modal{
  position:fixed;inset:0;z-index:11;display:none;align-items:center;justify-content:center;
  background:rgba(236,236,230,0.85);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
}
.sk-modal.open{display:flex;animation:fadeIn .16s ease}
.sk-panel{
  width:min(1060px,calc(100vw - 28px));max-height:calc(100vh - 28px);
  display:flex;flex-direction:column;
  background:var(--card);border:1px solid var(--hair-2);border-radius:18px;
  box-shadow:var(--shadow-modal);overflow:hidden;
}
.sk-head{display:flex;align-items:center;gap:12px;padding:12px 16px;border-bottom:1px solid var(--hair)}
.sk-head strong{font-size:13px;font-weight:600}
.sk-hint{font-family:var(--mono);font-size:10px;color:var(--muted);letter-spacing:0.04em;margin-left:auto}
.sk-head .x{border:0;background:transparent;cursor:pointer;font-size:16px;color:var(--ink-3);width:28px;height:28px;border-radius:8px;display:grid;place-items:center}
.sk-head .x:hover{background:var(--bg-2);color:var(--ink)}
.sk-main{display:flex;min-height:0;flex:1}
.sk-rail{
  flex:none;display:flex;flex-direction:column;gap:4px;align-items:center;
  padding:12px 10px;border-right:1px solid var(--hair);background:var(--bg);
}
.sk-rail .tool,.sk-rail .op{
  all:unset;cursor:pointer;width:34px;height:34px;border-radius:9px;
  display:grid;place-items:center;color:var(--ink-3);
  transition:background .12s,color .12s;
}
.sk-rail .tool:hover,.sk-rail .op:hover{background:var(--bg-2);color:var(--ink)}
.sk-rail .tool.active{background:var(--ink);color:var(--bg)}
.sk-rail .op:disabled{opacity:.35;cursor:default}
.sk-rail .op:disabled:hover{background:transparent;color:var(--ink-3)}
.sk-rail hr{width:22px;border:0;border-top:1px solid var(--hair-2);margin:6px 0}
.sk-rail svg{width:16px;height:16px}
.sk-stage{
  flex:1;display:grid;place-items:center;padding:20px;min-width:0;
  background:
    linear-gradient(45deg,var(--bg-2) 25%,transparent 25%,transparent 75%,var(--bg-2) 75%),
    linear-gradient(45deg,var(--bg-2) 25%,transparent 25%,transparent 75%,var(--bg-2) 75%),
    var(--bg);
  background-size:18px 18px;background-position:0 0,9px 9px;
}
#skCanvas{
  display:block;max-width:100%;max-height:min(62vh,640px);aspect-ratio:1;
  background:#fff;border-radius:6px;
  box-shadow:0 0 0 1px var(--hair-2),0 12px 40px -16px rgba(22,21,17,0.25);
  touch-action:none;cursor:crosshair;
}
.sk-foot{display:flex;align-items:center;gap:14px;padding:12px 16px;border-top:1px solid var(--hair);flex-wrap:wrap}
.swatches{display:flex;gap:5px;align-items:center}
.sw{
  all:unset;cursor:pointer;width:20px;height:20px;border-radius:50%;
  box-shadow:inset 0 0 0 1px rgba(22,21,17,0.14);
  transition:transform .12s,box-shadow .12s;
}
.sw:hover{transform:scale(1.15)}
.sw.active{box-shadow:inset 0 0 0 1px rgba(22,21,17,0.14),0 0 0 2px var(--card),0 0 0 3.5px var(--ink)}
.sw-custom{position:relative;width:20px;height:20px}
.sw-custom input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.sw-custom .ring{
  pointer-events:none;position:absolute;inset:0;border-radius:50%;
  background:conic-gradient(#f43f5e,#f59e0b,#84cc16,#06b6d4,#8b5cf6,#f43f5e);
  box-shadow:inset 0 0 0 1px rgba(22,21,17,0.14);
}
.sizer{display:flex;align-items:center;gap:10px}
.sizer input{width:110px;accent-color:var(--ink)}
.size-dot{width:26px;height:26px;display:grid;place-items:center}
.size-dot i{display:block;border-radius:50%;background:var(--ink)}
.sk-foot .primary{
  border:0;cursor:pointer;margin-left:auto;
  background:var(--ink);color:var(--bg);font-size:12.5px;font-weight:600;
  height:32px;padding:0 14px;border-radius:10px;
  transition:background .15s,transform .12s;
}
.sk-foot .primary:hover{background:#000;transform:translateY(-1px)}

/* ── toast ── */
.toast{
  position:fixed;left:50%;top:16px;transform:translateX(-50%);
  background:var(--ink);color:var(--bg);padding:8px 16px;
  font-size:12px;font-weight:500;display:none;z-index:20;border-radius:999px;
  box-shadow:0 10px 30px -10px rgba(22,21,17,0.4);
}
.toast.show{display:block;animation:fadeIn .16s ease}

/* ── mobile ── */
@media(max-width:760px){
  .top{padding:0 14px;grid-template-columns:1fr auto;height:50px}
  .seg{display:none}
  .count{display:none}
  .canvas{padding:16px 14px 250px}
  .wall,.mcol{gap:9px}
  .batch-list{grid-template-columns:1fr}
  .composer{width:calc(100vw - 20px);bottom:12px;padding:12px 12px 8px;border-radius:15px}
  .promptBox{font-size:17px}
  .pill{padding:0 8px}
  .spacer{display:none}
  .generate{margin-left:auto}
  .nav{display:none}
  .modal-bar{flex-wrap:wrap;width:calc(100vw - 20px);justify-content:center}
  .sk-hint{display:none}
  .sk-main{flex-direction:column}
  .sk-rail{flex-direction:row;border-right:0;border-bottom:1px solid var(--hair);padding:8px}
  .sk-rail hr{width:0;height:22px;border-top:0;border-left:1px solid var(--hair-2);margin:0 6px}
}
</style>
</head>
<body>
<div class="studio">
  <header class="top">
    <div class="brand">
      <span class="glyph">&pi;</span>
      <span class="name">Studio</span>
      <span class="sub">gpt-image-2</span>
    </div>
    <nav class="seg" id="filters">
      <button class="active" data-filter="all">All</button>
      <button data-filter="batch">Batches</button>
    </nav>
    <div class="right">
      <span class="count" id="count"></span>
      <span class="status" id="status-wrap"><span class="dot"></span><span id="status">Connecting</span></span>
    </div>
  </header>
  <main class="canvas"><div id="wall" class="wall"></div></main>

  <div id="modal" class="modal" aria-hidden="true"></div>

  <div id="sketchModal" class="sk-modal" aria-hidden="true">
    <div class="sk-panel">
      <div class="sk-head">
        <strong>Canvas</strong>
        <span class="sk-hint">B brush &middot; E eraser &middot; L line &middot; R rect &middot; O ellipse &middot; shift constrains &middot; &#8984;Z undo</span>
        <button type="button" class="x" id="skClose" aria-label="Close">&times;</button>
      </div>
      <div class="sk-main">
        <div class="sk-rail">
          <button type="button" class="tool active" data-tool="brush" title="Brush (B)"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M11.5 1.5l3 3L6 13l-4 1 1-4z"/><path d="M9.5 3.5l3 3"/></svg></button>
          <button type="button" class="tool" data-tool="eraser" title="Eraser (E)"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5.5 13.5l-3.8-3.8a1 1 0 010-1.4l6-6a1 1 0 011.4 0l4.6 4.6a1 1 0 010 1.4l-5.2 5.2z"/><path d="M14 13.5H5.5"/><path d="M5 6.5l4.5 4.5"/></svg></button>
          <button type="button" class="tool" data-tool="line" title="Line (L)"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M2.5 13.5l11-11"/></svg></button>
          <button type="button" class="tool" data-tool="rect" title="Rectangle (R)"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="2.5" y="3.5" width="11" height="9" rx="1"/></svg></button>
          <button type="button" class="tool" data-tool="ellipse" title="Ellipse (O)"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><ellipse cx="8" cy="8" rx="5.5" ry="4.5"/></svg></button>
          <hr>
          <button type="button" class="op" id="skUndo" title="Undo" disabled><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M6 3.5L2.5 7 6 10.5"/><path d="M2.5 7h7a4 4 0 010 8H8"/></svg></button>
          <button type="button" class="op" id="skRedo" title="Redo" disabled><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M10 3.5L13.5 7 10 10.5"/><path d="M13.5 7h-7a4 4 0 000 8H8"/></svg></button>
          <button type="button" class="op" id="skClear" title="Clear"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 4.5h10"/><path d="M6.5 4.5V3a1 1 0 011-1h1a1 1 0 011 1v1.5"/><path d="M4.5 4.5l.7 8a1 1 0 001 .9h3.6a1 1 0 001-.9l.7-8"/></svg></button>
        </div>
        <div class="sk-stage"><canvas id="skCanvas" width="1024" height="1024"></canvas></div>
      </div>
      <div class="sk-foot">
        <div class="swatches" id="swatches"></div>
        <div class="sw-custom" title="Custom color"><span class="ring"></span><input type="color" id="skColor" value="#161511"></div>
        <div class="sizer">
          <input type="range" id="skSize" min="2" max="64" value="8" aria-label="Brush size">
          <span class="size-dot"><i id="skSizeDot"></i></span>
        </div>
        <button type="button" class="primary" id="skUse">Use as reference</button>
      </div>
    </div>
  </div>

  <form id="composer" class="composer" autocomplete="off">
    <div id="refs" class="refs"></div>
    <textarea id="prompt" class="promptBox" placeholder="Describe an image&hellip;" rows="1"></textarea>
    <div class="controls">
      <label class="pill">style
        <select id="style" class="select">
          <option value="">auto</option>
          <option value="minecraft-screenshot">minecraft</option>
          <option value="poster">poster</option>
          <option value="wallpaper">wallpaper</option>
        </select>
      </label>
      <label class="pill">aspect
        <select id="size" class="select">
          <option value="auto">auto</option>
          <option value="1024x1024">1:1</option>
          <option value="1536x1024">3:2</option>
          <option value="1024x1536">2:3</option>
        </select>
      </label>
      <label class="pill">quality
        <select id="quality" class="select">
          <option value="auto">auto</option>
          <option value="medium">medium</option>
          <option value="high">high</option>
          <option value="low">low</option>
        </select>
      </label>
      <label class="pill">thinking
        <select id="thinking" class="select">
          <option value="off">off</option>
          <option value="minimal">minimal</option>
          <option value="low" selected>low</option>
          <option value="medium">medium</option>
          <option value="high">high</option>
        </select>
      </label>
      <div class="steps" id="counts">
        <button type="button" data-n="1">1</button>
        <button type="button" data-n="2">2</button>
        <button type="button" data-n="4" class="active">4</button>
        <button type="button" data-n="6">6</button>
        <button type="button" data-n="9">9</button>
      </div>
      <span class="spacer"></span>
      <button id="openSketch" class="ghost-btn" type="button">
        <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M11.5 1.5l3 3L6 13l-4 1 1-4z"/></svg>
        Draw
      </button>
      <button id="generate" class="generate" type="submit">
        <span>Generate</span>
        <span class="ret" aria-hidden="true">&#8629;</span>
      </button>
    </div>
  </form>
  <div id="toast" class="toast"></div>
</div>
<script>
const TOKEN=${JSON.stringify(token)};
let images=[],selected=null,filter='all',count=4,refs=[],pending=0;
const $=s=>document.querySelector(s),$$=s=>document.querySelectorAll(s);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const imgUrl=x=>'/api/image/'+encodeURIComponent(x.imageId)+'?token='+encodeURIComponent(TOKEN);
async function api(path,opts={}){const sep=path.includes('?')?'&':'?';const r=await fetch(path+sep+'token='+encodeURIComponent(TOKEN),opts);if(!r.ok)throw new Error(await r.text());return r.headers.get('content-type')?.includes('json')?r.json():r.text()}
function toast(t){const el=$('#toast');el.textContent=t;el.classList.add('show');clearTimeout(toast._t);toast._t=setTimeout(()=>el.classList.remove('show'),1800)}
function setStatus(mode,text){const w=$('#status-wrap');w.classList.remove('live','busy');if(mode)w.classList.add(mode);$('#status').textContent=text}

function renderRefs(){
  const el=$('#refs');el.classList.toggle('hasRefs',refs.length>0);
  el.innerHTML=refs.map(r=>'<div class="ref-chip" title="Reference"><img src="'+imgUrl(r)+'" alt=""><button type="button" data-ref="'+esc(r.imageId)+'" aria-label="Remove reference">&times;</button></div>').join('');
  $$('[data-ref]').forEach(b=>b.onclick=()=>{refs=refs.filter(r=>r.imageId!==b.dataset.ref);renderRefs()});
}
function addRef(x){if(!x||refs.some(r=>r.imageId===x.imageId))return;refs.push(x);renderRefs();toast('Reference added')}

function batchKey(x){return x.batchId||(x.savedPath.includes('/batches/')?x.savedPath.split('/batches/')[1]?.split('/')[0]:'')||''}
function isOutput(x){return x.kind!=='sketch'&&x.provider!=='local-sketch'}
function passes(x){if(!isOutput(x))return false;if(filter==='batch'&&!batchKey(x))return false;return true}
function visibleImages(){return images.filter(passes)}
function columnCount(){const w=$('#wall')?.clientWidth||window.innerWidth;return Math.max(1,Math.min(6,Math.floor(w/300)||1))}
function tileHtml(x,i){
  return '<div class="tile" role="button" tabindex="0" aria-label="Open image" data-id="'+esc(x.imageId)+'" style="animation-delay:'+Math.min(i*22,260)+'ms">'+
    '<img src="'+imgUrl(x)+'" loading="lazy" alt="">'+
    '<button type="button" class="add" data-add="'+esc(x.imageId)+'" title="Use as reference">+</button>'+
  '</div>';
}
function phHtml(){return '<div class="tile ph" aria-hidden="true"></div>'}
function bindTiles(){
  $$('.tile[data-id]').forEach(t=>{
    t.onclick=()=>select(t.dataset.id);
    t.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select(t.dataset.id)}};
  });
  $$('[data-add]').forEach(b=>b.onclick=e=>{e.stopPropagation();addRef(images.find(x=>x.imageId===b.dataset.add))});
}
function emptyHtml(blank){
  return '<div class="empty">'+
    '<h2>'+(blank?'A blank wall.':'Nothing in this view.')+'</h2>'+
    '<p>'+(blank?'Describe an image below, or open the canvas and sketch one.':'Switch the filter to see other generations.')+'</p>'+
    (blank?'<div class="kbd"><kbd>/</kbd> prompt &nbsp;&middot;&nbsp; <kbd>&#8629;</kbd> generate</div>':'')+
  '</div>';
}
function renderWall(visible){
  const cols=columnCount();
  $('#wall').className='wall';
  $('#wall').style.setProperty('--cols',String(cols));
  const buckets=Array.from({length:cols},()=>[]);
  for(let i=0;i<pending;i++)buckets[i%cols].push(null);
  visible.forEach((x,i)=>buckets[(i+pending)%cols].push(x));
  let n=0;
  $('#wall').innerHTML=buckets.map(col=>'<div class="mcol">'+col.map(x=>x?tileHtml(x,n++):phHtml()).join('')+'</div>').join('');
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
    return '<section class="batch-card"><div class="batch-preview">'+items.slice(0,4).map(x=>tileHtml(x,0)).join('')+'</div><div class="batch-info"><span class="batch-title">'+esc(g.title)+'</span><span class="batch-meta">'+items.length+(date?' &middot; '+esc(date):'')+'</span></div></section>';
  }).join('');
  bindTiles();
}
function render(){
  const visible=visibleImages();
  $('#count').textContent=visible.length?visible.length+(visible.length===1?' image':' images'):'';
  if(!visible.length&&!pending){
    $('#wall').className='wall';
    $('#wall').style.removeProperty('--cols');
    $('#wall').innerHTML=emptyHtml(!images.length);
    return;
  }
  if(filter==='batch')renderBatches(visible);else renderWall(visible);
}
async function load(){
  try{const data=await api('/api/images');images=data.images||[];render()}
  catch(e){$('#wall').innerHTML='<div class="empty"><h2>Could not load images.</h2><p>'+esc(e.message)+'</p></div>'}
}

/* ── lightbox ── */
function select(id){selected=images.find(x=>x.imageId===id);openModal()}
function openModal(){
  const x=selected;if(!x)return;
  const m=$('#modal');
  if(!m.classList.contains('open')){
    m.classList.add('open');m.setAttribute('aria-hidden','false');
    m.innerHTML='<div class="modal-counter" id="mc"></div>'+
      '<button class="close" data-close aria-label="Close">&times;</button>'+
      '<button class="nav prev" data-nav="prev" aria-label="Previous">&lsaquo;</button>'+
      '<button class="nav next" data-nav="next" aria-label="Next">&rsaquo;</button>'+
      '<img class="modalImg" id="mi" alt="">'+
      '<div class="modal-cap" id="mcap"></div>'+
      '<div class="modal-sub" id="msub"></div>'+
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
  $('#mc').textContent=(idx+1)+' / '+v.length;
  $('#mi').src=imgUrl(x);$('#mi').alt=x.prompt||'';
  $('#mcap').textContent=x.batchPrompt||x.prompt||'';
  const bits=[x.size,x.quality,x.createdAt?new Date(x.createdAt).toLocaleString([],{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}):''].filter(b=>b&&b!=='auto');
  $('#msub').textContent=bits.join(' \\u00b7 ');
}
function closeModal(){selected=null;const m=$('#modal');m.classList.remove('open');m.setAttribute('aria-hidden','true');m.innerHTML=''}
function move(delta){const v=visibleImages();if(!selected||!v.length)return;const i=v.findIndex(x=>x.imageId===selected.imageId);selected=v[(i+delta+v.length)%v.length];openModal()}

function selectedPrompt(){return selected?.batchPrompt||selected?.prompt||''}
function setComposerPrompt(text){ta.value=text;autosize();ta.focus();closeModal()}
async function act(a){
  if(!selected)return;
  if(a==='copyprompt'){await navigator.clipboard.writeText(selectedPrompt());return toast('Prompt copied')}
  if(a==='vary')return setComposerPrompt(selectedPrompt()+'\\n\\nVariation: ');
  if(a==='rerun'){startGeneration(selectedPrompt(),count);toast('Rerunning');return}
  if(a==='ref'){addRef(selected);closeModal();ta.focus();return}
  await api('/api/'+a,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({imageId:selected.imageId})});
  toast(a==='open'?'Opened':'Revealed');
}

/* ── composer ── */
const ta=$('#prompt');
function autosize(){ta.style.height='auto';ta.style.height=Math.min(ta.scrollHeight,150)+'px'}
ta.addEventListener('input',autosize);
ta.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();$('#composer').requestSubmit()}});
document.addEventListener('keydown',e=>{
  if(e.key==='/'&&document.activeElement!==ta&&!$('#modal').classList.contains('open')&&!sketchOpen()){
    e.preventDefault();ta.focus();
  }
});
$('#counts').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;$$('#counts button').forEach(x=>x.classList.toggle('active',x===b));count=Number(b.dataset.n)||1});

async function startGeneration(prompt,n){
  pending+=n;render();
  setStatus('busy','Generating');
  try{
    await api('/api/generate',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({prompt,style:$('#style').value,size:$('#size').value,quality:$('#quality').value,thinking:$('#thinking').value,count:n,references:refs.map(r=>r.imageId)})});
  }catch(err){toast(err.message)}
  finally{pending=0;setStatus('live','Live');load()}
}
$('#composer').onsubmit=e=>{
  e.preventDefault();
  const prompt=ta.value.trim();
  if(!prompt&&refs.length===0)return toast('Add a prompt or reference');
  startGeneration(prompt,count);
};

$$('#filters [data-filter]').forEach(b=>b.onclick=()=>{$$('#filters [data-filter]').forEach(x=>x.classList.remove('active'));b.classList.add('active');filter=b.dataset.filter;render()});

document.addEventListener('keydown',e=>{
  if($('#modal').classList.contains('open')){
    if(e.key==='Escape')closeModal();
    if(e.key==='ArrowRight')move(1);
    if(e.key==='ArrowLeft')move(-1);
  }
});

let resizeT;window.addEventListener('resize',()=>{clearTimeout(resizeT);resizeT=setTimeout(render,140)});

/* ── live events ── */
const events=new EventSource('/events?token='+encodeURIComponent(TOKEN));
events.addEventListener('ready',()=>setStatus('live','Live'));
events.addEventListener('imagegen:generated',()=>{if(pending>0)pending--;setStatus(pending?'busy':'live',pending?'Generating':'Live');load()});
events.addEventListener('generation:start',e=>{if(!pending){pending=1;render()}setStatus('busy',JSON.parse(e.data).message||'Generating')});
events.onerror=()=>setStatus('','Disconnected');

/* ── canvas ── */
const SK_COLORS=['#161511','#ffffff','#8a857a','#e23c3c','#f08c1e','#f2c218','#3d9c53','#2b7fd4','#2b4bdf','#8b5cf6','#d648a5','#7a4a24'];
const skModal=$('#sketchModal'),skCanvas=$('#skCanvas'),skx=skCanvas.getContext('2d',{willReadFrequently:true});
const sk={tool:'brush',color:'#161511',size:8,drawing:false,last:null,start:null,snap:null,undo:[],redo:[]};
function sketchOpen(){return skModal.classList.contains('open')}
function skInit(){skx.globalCompositeOperation='source-over';skx.fillStyle='#ffffff';skx.fillRect(0,0,skCanvas.width,skCanvas.height);skx.lineCap='round';skx.lineJoin='round'}
function skOps(){$('#skUndo').disabled=!sk.undo.length;$('#skRedo').disabled=!sk.redo.length}
function skSnapshot(){sk.undo.push(skCanvas.toDataURL());if(sk.undo.length>40)sk.undo.shift();sk.redo.length=0;skOps()}
function skPaint(url){return new Promise(res=>{const im=new Image();im.onload=()=>{skx.globalCompositeOperation='source-over';skx.clearRect(0,0,skCanvas.width,skCanvas.height);skx.drawImage(im,0,0);res()};im.src=url})}
async function skUndoFn(){if(!sk.undo.length)return;sk.redo.push(skCanvas.toDataURL());await skPaint(sk.undo.pop());skOps()}
async function skRedoFn(){if(!sk.redo.length)return;sk.undo.push(skCanvas.toDataURL());await skPaint(sk.redo.pop());skOps()}
function skPt(e){const r=skCanvas.getBoundingClientRect();return {x:(e.clientX-r.left)*skCanvas.width/r.width,y:(e.clientY-r.top)*skCanvas.height/r.height}}
function skStroke(){skx.strokeStyle=sk.color;skx.lineWidth=sk.size;skx.globalCompositeOperation=sk.tool==='eraser'?'destination-out':'source-over'}
function skSeg(a,b){skStroke();skx.beginPath();skx.moveTo(a.x,a.y);skx.lineTo(b.x,b.y);skx.stroke()}
function skShape(a,b,constrain){
  skStroke();skx.globalCompositeOperation='source-over';
  let dx=b.x-a.x,dy=b.y-a.y;
  if(sk.tool==='line'){
    if(constrain){const ang=Math.round(Math.atan2(dy,dx)/(Math.PI/4))*(Math.PI/4),len=Math.hypot(dx,dy);dx=Math.cos(ang)*len;dy=Math.sin(ang)*len}
    skx.beginPath();skx.moveTo(a.x,a.y);skx.lineTo(a.x+dx,a.y+dy);skx.stroke();return;
  }
  if(constrain){const m=Math.max(Math.abs(dx),Math.abs(dy));dx=Math.sign(dx||1)*m;dy=Math.sign(dy||1)*m}
  if(sk.tool==='rect'){skx.strokeRect(Math.min(a.x,a.x+dx),Math.min(a.y,a.y+dy),Math.abs(dx),Math.abs(dy));return}
  skx.beginPath();skx.ellipse(a.x+dx/2,a.y+dy/2,Math.abs(dx/2),Math.abs(dy/2),0,0,Math.PI*2);skx.stroke();
}
skCanvas.addEventListener('pointerdown',e=>{
  e.preventDefault();skCanvas.setPointerCapture(e.pointerId);
  sk.drawing=true;skSnapshot();
  const p=skPt(e);sk.start=p;sk.last=p;
  if(sk.tool==='brush'||sk.tool==='eraser')skSeg(p,p);
  else sk.snap=skx.getImageData(0,0,skCanvas.width,skCanvas.height);
});
skCanvas.addEventListener('pointermove',e=>{
  if(!sk.drawing)return;
  const p=skPt(e);
  if(sk.tool==='brush'||sk.tool==='eraser'){skSeg(sk.last,p);sk.last=p}
  else{skx.globalCompositeOperation='source-over';skx.putImageData(sk.snap,0,0);skShape(sk.start,p,e.shiftKey)}
});
skCanvas.addEventListener('pointerup',()=>{sk.drawing=false;sk.snap=null;sk.last=null;sk.start=null});
function skSetTool(tool){sk.tool=tool;$$('.sk-rail .tool').forEach(b=>b.classList.toggle('active',b.dataset.tool===tool))}
$$('.sk-rail .tool').forEach(b=>b.onclick=()=>skSetTool(b.dataset.tool));
function skSetColor(c){sk.color=c;$('#skColor').value=c;$('#skSizeDot').style.background=c;$$('.sw').forEach(b=>b.classList.toggle('active',b.dataset.color===c))}
$('#swatches').innerHTML=SK_COLORS.map(c=>'<button type="button" class="sw" data-color="'+c+'" style="background:'+c+'" aria-label="'+c+'"></button>').join('');
$$('.sw').forEach(b=>b.onclick=()=>skSetColor(b.dataset.color));
$('#skColor').addEventListener('input',e=>skSetColor(e.target.value));
function skSizeDot(){const d=Math.max(3,Math.min(22,sk.size/2.6));const el=$('#skSizeDot');el.style.width=d+'px';el.style.height=d+'px'}
$('#skSize').addEventListener('input',e=>{sk.size=Number(e.target.value)||8;skSizeDot()});
$('#skUndo').onclick=skUndoFn;$('#skRedo').onclick=skRedoFn;
$('#skClear').onclick=()=>{skSnapshot();skInit()};
$('#openSketch').onclick=()=>{skModal.classList.add('open');skModal.setAttribute('aria-hidden','false')};
$('#skClose').onclick=()=>{skModal.classList.remove('open');skModal.setAttribute('aria-hidden','true')};
$('#skUse').onclick=async()=>{
  const blob=await new Promise(r=>skCanvas.toBlob(r,'image/png'));
  try{
    const res=await api('/api/sketch',{method:'POST',headers:{'content-type':'image/png'},body:blob});
    addRef(res.metadata);
    $('#skClose').click();
  }catch(err){toast(err.message)}
};
document.addEventListener('keydown',e=>{
  if(!sketchOpen())return;
  const mod=e.metaKey||e.ctrlKey;
  if(mod&&e.key.toLowerCase()==='z'){e.preventDefault();e.shiftKey?skRedoFn():skUndoFn();return}
  if(mod)return;
  const k=e.key.toLowerCase();
  if(k==='escape')return $('#skClose').click();
  if(k==='b')skSetTool('brush');
  if(k==='e')skSetTool('eraser');
  if(k==='l')skSetTool('line');
  if(k==='r')skSetTool('rect');
  if(k==='o')skSetTool('ellipse');
  if(k==='['){sk.size=Math.max(2,sk.size-4);$('#skSize').value=sk.size;skSizeDot()}
  if(k===']'){sk.size=Math.min(64,sk.size+4);$('#skSize').value=sk.size;skSizeDot()}
});

skInit();skSetColor('#161511');skSizeDot();
renderRefs();
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
			if (url.pathname === "/api/open") await openPath(metadata.savedPath);
			if (url.pathname === "/api/reveal") await revealPath(metadata.savedPath);
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
		try {
			await openBrowser(url);
			ctx.ui.notify("Image studio opened.", "info");
		} catch (error) {
			ctx.ui.notify("Could not open browser automatically. Copy the studio URL from the message.", "warning");
			pi.sendMessage({
				customType: "imagegen-result",
				content: `Image studio is running, but the browser could not be opened automatically.\n\nOpen this URL manually:\n${url}\n\nError: ${error instanceof Error ? error.message : String(error)}`,
				display: true,
				details: { url },
			});
		}
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
					await openPath(path);
					ctx.ui.notify(`Opened ${path}`, "success");
					return;
				}

				if (subcommand === "reveal") {
					await revealPath(path);
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
