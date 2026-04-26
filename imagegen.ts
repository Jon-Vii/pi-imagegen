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
	batchId?: string;
	batchPrompt?: string;
	batchIndex?: number;
	batchCount?: number;
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

function buildRequest(params: ToolParams, responseModel: string, sessionId: string) {
	const size = params.size ?? "auto";
	const quality = params.quality ?? "auto";
	const background = params.background ?? "auto";
	const outputFormat = params.outputFormat ?? "png";

	return {
		model: responseModel,
		store: false,
		stream: true,
		instructions:
			"You are an image generation dispatcher. Use the image_generation tool to create exactly the image requested by the user. Do not write code.",
		input: [
			{
				role: "user",
				content: [{ type: "input_text", text: `Generate this image: ${params.prompt}` }],
			},
		],
		text: { verbosity: "low" },
		include: ["reasoning.encrypted_content"],
		prompt_cache_key: sessionId,
		tool_choice: "auto",
		parallel_tool_calls: true,
		reasoning: { effort: "low", summary: "auto" },
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
	extraMetadata: Partial<Pick<ImagegenMetadata, "batchId" | "batchPrompt" | "batchIndex" | "batchCount">> = {},
) {
	const token = await ctx.modelRegistry.getApiKeyForProvider(PROVIDER);
	if (!token) {
		throw new Error("Missing OpenAI Codex OAuth credentials. Run /login and select OpenAI ChatGPT Plus/Pro (Codex).");
	}

	const accountId = getAccountId(token);
	const responseModel = ctx.model?.provider === PROVIDER ? ctx.model.id : DEFAULT_RESPONSE_MODEL;
	const sessionId = randomUUID();
	const body = buildRequest(params, responseModel, sessionId);
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

async function readJsonBody(req: IncomingMessage): Promise<any> {
	const chunks: Buffer[] = [];
	for await (const chunk of req) chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
	if (chunks.length === 0) return {};
	return JSON.parse(Buffer.concat(chunks).toString("utf8"));
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
:root{color-scheme:light;--bg:#fbfaf7;--paper:#fff;--ink:#24211d;--muted:#8f877c;--line:#ebe5dc;--soft:#f5f1ea;--accent:#ff6b57;--accent2:#ff9a3d;--shadow:0 18px 50px rgba(55,45,35,.10)}*{box-sizing:border-box}html,body{height:100%;margin:0}body{background:var(--bg);color:var(--ink);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;overflow:hidden}.studio{height:100vh;display:flex;flex-direction:column}.top{height:64px;display:flex;align-items:center;justify-content:space-between;padding:0 28px;border-bottom:1px solid var(--line);background:rgba(251,250,247,.82);backdrop-filter:blur(18px)}.brand{font-weight:760;font-size:18px;letter-spacing:-.02em}.search{width:min(520px,42vw);border:1px solid var(--line);background:var(--paper);border-radius:999px;padding:11px 16px;color:var(--ink);outline:none}.status{color:var(--muted);font-size:12px}.canvas{position:relative;flex:1;overflow:auto;padding:28px 28px 168px}.sectionHead{display:flex;align-items:end;justify-content:space-between;margin:8px auto 16px;max-width:1280px}.sectionHead h1{font-size:28px;line-height:1;margin:0;letter-spacing:-.04em}.filters{display:flex;gap:8px}.chip{border:1px solid var(--line);background:var(--paper);border-radius:999px;padding:8px 12px;color:var(--muted);cursor:pointer}.chip.active{background:#fff0ec;border-color:#ffc8bc;color:#d94934}.grid{max-width:1280px;margin:0 auto;display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:18px}.card{border:1px solid var(--line);background:var(--paper);border-radius:22px;overflow:hidden;box-shadow:0 1px 0 rgba(0,0,0,.02);cursor:pointer;transition:transform .16s ease,box-shadow .16s ease,border-color .16s ease}.card:hover{transform:translateY(-3px);box-shadow:var(--shadow);border-color:#e0d7cc}.card.selected{outline:3px solid #ffb4a8}.thumb{width:100%;aspect-ratio:1.18;object-fit:cover;background:#eee8df;display:block}.meta{padding:11px 12px}.prompt{font-size:13px;color:#4f4840;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.date{margin-top:3px;font-size:11px;color:var(--muted)}.empty{max-width:680px;margin:10vh auto;text-align:center;color:var(--muted)}.composer{position:fixed;left:50%;bottom:22px;transform:translateX(-50%);width:min(920px,calc(100vw - 44px));background:rgba(255,255,255,.94);border:1px solid var(--line);border-radius:28px;box-shadow:0 24px 80px rgba(60,45,30,.18);backdrop-filter:blur(20px);padding:14px}.promptRow{display:flex;gap:12px;align-items:end}.promptBox{flex:1;min-height:58px;max-height:140px;resize:vertical;border:0;outline:0;background:transparent;color:var(--ink);font:16px/1.4 inherit;padding:8px 10px}.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;border-top:1px solid var(--line);padding:12px 8px 0}.select,.count{border:1px solid var(--line);background:var(--soft);border-radius:13px;padding:9px 10px;color:var(--ink)}.count{width:72px}.generate{margin-left:auto;border:0;border-radius:15px;padding:12px 18px;background:linear-gradient(135deg,var(--accent),var(--accent2));color:white;font-weight:800;cursor:pointer;box-shadow:0 10px 28px rgba(255,107,87,.26)}.generate:disabled{opacity:.55;cursor:default}.panel{position:fixed;right:26px;top:84px;width:360px;max-height:calc(100vh - 260px);overflow:auto;background:rgba(255,255,255,.96);border:1px solid var(--line);border-radius:24px;box-shadow:var(--shadow);padding:14px;display:none}.panel.open{display:block}.preview{width:100%;max-height:260px;object-fit:contain;border-radius:16px;background:var(--soft)}.btns{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:12px 0}.btn{border:1px solid var(--line);background:var(--paper);border-radius:12px;padding:9px 10px;cursor:pointer;color:var(--ink)}.h{font-weight:750;margin:12px 0 5px}.kv{color:var(--muted);word-break:break-word;font-size:13px}code{font-size:12px;color:#b94f28}.toast{position:fixed;left:50%;bottom:142px;transform:translateX(-50%);background:#24211d;color:white;border-radius:999px;padding:10px 14px;display:none}@media(max-width:860px){.search{display:none}.panel{left:18px;right:18px;width:auto}.promptRow{display:block}.generate{width:100%;margin-left:0}.canvas{padding-bottom:230px}}
</style>
</head>
<body><div class="studio"><header class="top"><div class="brand">Pi Image Studio</div><input id="q" class="search" placeholder="Search images, prompts, paths…"><div class="status" id="status">Loading…</div></header><main class="canvas"><div class="sectionHead"><div><h1 id="title">Canvas</h1><div class="status" id="count">0 images</div></div><div class="filters"><button class="chip active" data-filter="all">All</button><button class="chip" data-filter="batch">Batches</button><button class="chip" data-filter="tmp">/tmp</button></div></div><div id="grid" class="grid"></div></main><aside id="panel" class="panel"></aside><form id="composer" class="composer"><div class="promptRow"><textarea id="prompt" class="promptBox" placeholder="Describe an image…"></textarea></div><div class="controls"><select id="style" class="select"><option value="">Auto style</option><option value="minecraft-screenshot">Minecraft screenshot</option><option value="minecraft">Minecraft</option><option value="poster">Poster</option><option value="wallpaper">Wallpaper</option></select><select id="size" class="select"><option value="auto">Auto size</option><option value="1024x1024">Square</option><option value="1536x1024">Landscape</option><option value="1024x1536">Portrait</option></select><select id="quality" class="select"><option value="auto">Auto quality</option><option value="medium">Medium</option><option value="high">High</option><option value="low">Low</option></select><label class="status">Count <input id="n" class="count" type="number" min="1" max="12" value="1"></label><button id="generate" class="generate" type="submit">Generate</button></div></form><div id="toast" class="toast"></div></div>
<script>
const TOKEN=${JSON.stringify(token)};let images=[],selected=null,filter='all';
const $=s=>document.querySelector(s);const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function imgUrl(x){return '/api/image/'+encodeURIComponent(x.imageId)+'?token='+encodeURIComponent(TOKEN)}
async function api(path,opts={}){const sep=path.includes('?')?'&':'?';const r=await fetch(path+sep+'token='+encodeURIComponent(TOKEN),opts);if(!r.ok)throw new Error(await r.text());return r.headers.get('content-type')?.includes('json')?r.json():r.text()}
function toast(t){$('#toast').textContent=t;$('#toast').style.display='block';setTimeout(()=>$('#toast').style.display='none',1800)}
async function load(){const data=await api('/api/images');images=data.images||[];render()}
function batchKey(x){return x.batchId || (x.savedPath.includes('/batches/') ? x.savedPath.split('/batches/')[1]?.split('/')[0] : '') || ''}
function passes(x){const q=$('#q').value.toLowerCase();if(filter==='batch'&&!batchKey(x))return false;if(filter==='tmp'&&!x.savedPath.startsWith('/tmp/'))return false;if(!q)return true;return [x.prompt,x.revisedPrompt,x.savedPath,x.imageId,x.batchPrompt,batchKey(x)].join(' ').toLowerCase().includes(q)}
function render(){const visible=images.filter(passes);$('#count').textContent=visible.length+' image'+(visible.length===1?'':'s');$('#grid').innerHTML=visible.map(x=>'<article class="card '+(selected?.imageId===x.imageId?'selected':'')+'" data-id="'+esc(x.imageId)+'"><img class="thumb" src="'+imgUrl(x)+'"><div class="meta"><div class="prompt">'+esc(x.prompt)+'</div><div class="date">'+esc(new Date(x.createdAt).toLocaleString())+(batchKey(x)?' · '+esc(x.batchIndex||'')+'/'+esc(x.batchCount||''):'')+'</div></div></article>').join('')||'<div class="empty"><h2>No images yet</h2><p>Write a prompt below and generate from the foundation up.</p></div>';document.querySelectorAll('.card').forEach(c=>c.onclick=()=>select(c.dataset.id))}
function select(id){selected=images.find(x=>x.imageId===id);render();const x=selected;$('#panel').classList.add('open');$('#panel').innerHTML='<img class="preview" src="'+imgUrl(x)+'"><div class="btns"><button class="btn" data-act="open">Open</button><button class="btn" data-act="reveal">Reveal</button><button class="btn" data-act="attach">Attach</button><button class="btn" data-act="copypath">Copy path</button></div><div class="h">Prompt</div><div class="kv">'+esc(x.prompt)+'</div>'+(x.revisedPrompt?'<div class="h">Revised</div><div class="kv">'+esc(x.revisedPrompt)+'</div>':'')+'<div class="h">Path</div><div class="kv"><code>'+esc(x.savedPath)+'</code></div>';document.querySelectorAll('[data-act]').forEach(b=>b.onclick=()=>act(b.dataset.act))}
async function act(a){if(!selected)return;if(a==='copypath'){await navigator.clipboard.writeText(selected.savedPath);return toast('Path copied')}await api('/api/'+(a==='attach'?'insert':a),{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({imageId:selected.imageId})});toast(a==='attach'?'Attached to prompt':'Done')}
$('#composer').onsubmit=async e=>{e.preventDefault();const prompt=$('#prompt').value.trim();if(!prompt)return;$('#generate').disabled=true;$('#generate').textContent='Generating…';try{await api('/api/generate',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({prompt,style:$('#style').value,size:$('#size').value,quality:$('#quality').value,count:Number($('#n').value)||1})});toast('Generation started')}catch(err){toast(err.message)}finally{$('#generate').disabled=false;$('#generate').textContent='Generate'}};
$('#q').oninput=render;document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{document.querySelectorAll('[data-filter]').forEach(x=>x.classList.remove('active'));b.classList.add('active');filter=b.dataset.filter;render()});
const events=new EventSource('/events?token='+encodeURIComponent(TOKEN));events.addEventListener('ready',()=>{$('#status').textContent='Live'});events.addEventListener('imagegen:generated',()=>{ $('#status').textContent='Updated '+new Date().toLocaleTimeString(); load() });events.addEventListener('generation:start',e=>{$('#status').textContent=JSON.parse(e.data).message});events.onerror=()=>{$('#status').textContent='Disconnected'};load().catch(e=>{$('#grid').innerHTML='<div class="empty">'+esc(e.message)+'</div>'});
</script></body></html>`;
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

	async function handleStudioRequest(req: IncomingMessage, res: ServerResponse) {
		const url = new URL(req.url ?? "/", "http://127.0.0.1");
		if (url.pathname !== "/favicon.ico" && url.searchParams.get("token") !== studioToken) {
			writeText(res, 403, "Forbidden");
			return;
		}
		if (url.pathname === "/favicon.ico") return writeText(res, 204, "");
		if (req.method === "GET" && (url.pathname === "/" || url.pathname === "/studio")) return writeHtml(res, renderStudioPage(studioToken));
		if (req.method === "GET" && url.pathname === "/events") return handleStudioEvents(req, res);
		if (req.method === "GET" && url.pathname === "/api/images") return writeJson(res, 200, { images: await readRecentMetadata(1000) });
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
			if (!prompt) return writeJson(res, 400, { ok: false, error: "Prompt is required." });
			if (style && !STYLE_PRESETS[style]) return writeJson(res, 400, { ok: false, error: `Unknown style: ${style}` });
			const options = {
				style: style || undefined,
				size: String(body.size ?? "auto") as ToolParams["size"],
				quality: String(body.quality ?? "auto") as ToolParams["quality"],
			};
			const results: ImagegenDetails[] = [];
			if (count === 1) {
				broadcastStudioEvent("generation:start", { message: "Generating image…" });
				const { details } = await generateImage(applyStyle(prompt, options), lastCtx.signal, undefined, lastCtx);
				pi.events.emit("imagegen:generated", details);
				broadcastStudioEvent("imagegen:generated", details);
				results.push(details);
			} else {
				const batchId = batchDirName(prompt);
				const batchDir = join(getAgentDir(), "generated-images", "batches", batchId);
				for (let index = 0; index < count; index++) {
					broadcastStudioEvent("generation:start", { message: `Generating ${index + 1}/${count}…` });
					const params = applyStyle(`${prompt}. Variation ${index + 1} of ${count}; make this composition distinct from the others.`, {
						...options,
						outputPath: join(batchDir, `${String(index + 1).padStart(2, "0")}.png`),
					});
					const { details } = await generateImage(params, lastCtx.signal, undefined, lastCtx, {
						batchId,
						batchPrompt: prompt,
						batchIndex: index + 1,
						batchCount: count,
					});
					pi.events.emit("imagegen:generated", details);
					broadcastStudioEvent("imagegen:generated", details);
					results.push(details);
				}
				await mkdir(batchDir, { recursive: true });
				await writeFile(join(batchDir, "batch.json"), JSON.stringify({ createdAt: new Date().toISOString(), prompt, count, images: results }, null, 2), "utf8");
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
						"/img gen [--style name] [--size 1536x1024] [--quality medium] <prompt>",
						"/img batch <count> [--style name] <prompt>",
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
