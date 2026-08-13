//! `image_analyze` tool — analyze images using a dedicated vision model.

use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use futures_util::StreamExt;
use serde_json::{Value, json};

use crate::config::VisionModelConfig;
use crate::llm_client::sanitize_http_error_body;
use crate::tools::spec::{
    ToolCapability, ToolContext, ToolError, ToolResult, ToolSpec, required_str,
};

// 统一输出上限(本地/云端一致):4096 覆盖长文档转写。慢设备上生成不满
// 4096 时由总超时(DEFAULT_VISION_REQUEST_TIMEOUT_SECS)截断返回部分内容
// (truncated 标记),不会重试死循环。可由 VisionModelConfig.max_output_tokens 覆盖。
const DEFAULT_VISION_MAX_OUTPUT_TOKENS: u32 = 4096;

// Pinvou fork:单次请求总预算默认 90s,从 send() 之前起算,覆盖连接、等待
// 响应头和流式全程;deadline 到 → 返回已累积内容 + truncated。可由
// VisionModelConfig.request_timeout_secs 覆盖。
const DEFAULT_VISION_REQUEST_TIMEOUT_SECS: u64 = 90;

// Pinvou fork:有限重试参数——仅 is_retryable_status 的 HTTP 状态码可重试,
// 最多 MAX_VISION_RETRIES 次,退避 1s→2s;Retry-After 封顶
// MAX_VISION_RETRY_AFTER_SECS。连接失败/超时/流中断/其它状态码不重试。
const MAX_VISION_RETRIES: u32 = 2;
const MAX_VISION_RETRY_AFTER_SECS: u64 = 10;

// Pinvou fork:仅这些 HTTP 状态码可重试(限流/服务端临时故障)。
fn is_retryable_status(status: u16) -> bool {
    matches!(status, 429 | 499 | 500 | 502 | 503 | 504)
}

// Pinvou fork:解析 Retry-After(秒数形式),封顶 MAX_VISION_RETRY_AFTER_SECS。
fn retry_after_delay(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let secs = headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<u64>().ok())?;
    Some(Duration::from_secs(secs.min(MAX_VISION_RETRY_AFTER_SECS)))
}

// Pinvou fork:SSE 字节行缓冲——按 \n 切出完整行后再 UTF-8 解码。
// 多字节字符(如汉字)被 chunk 边界切断时,逐 chunk 解码会产生替换字符
// 甚至静默丢 delta;字节缓冲保证只在完整行上解码(\n 不可能出现在
// 多字节 UTF-8 序列内部)。
#[derive(Default)]
struct SseLineBuffer {
    buffer: Vec<u8>,
}

impl SseLineBuffer {
    fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buffer.extend_from_slice(chunk);
        let mut lines = Vec::new();
        while let Some(pos) = self.buffer.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = self.buffer.drain(..=pos).collect();
            lines.push(String::from_utf8_lossy(&line[..line.len() - 1]).into_owned());
        }
        lines
    }

    // 流正常结束时,末尾可能残留无换行的最后一行,取出不丢 delta。
    fn finish(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            return None;
        }
        let rest = std::mem::take(&mut self.buffer);
        Some(String::from_utf8_lossy(&rest).into_owned())
    }
}

// Pinvou fork:处理一行 SSE;返回 true 表示收到 [DONE] 应结束流。
fn process_sse_line(
    line: &str,
    content: &mut String,
    truncated: &mut bool,
    saw_data_event: &mut bool,
) -> bool {
    let line = line.trim();
    if !line.starts_with("data:") {
        return false;
    }
    let data = line[5..].trim();
    if data == "[DONE]" {
        return true;
    }
    *saw_data_event = true;
    let Ok(event) = serde_json::from_str::<Value>(data) else {
        return false;
    };
    if let Some(delta) = event
        .pointer("/choices/0/delta/content")
        .and_then(|c| c.as_str())
    {
        content.push_str(delta);
    }
    if event
        .pointer("/choices/0/finish_reason")
        .and_then(|f| f.as_str())
        == Some("length")
    {
        *truncated = true;
    }
    false
}

// Pinvou fork:非流式响应解析(stream: Some(false))——普通 JSON 的
// choices[0].message.content;finish_reason == "length" 标 truncated。
fn parse_non_streaming_body(body: &str) -> Result<(String, bool), ToolError> {
    let json: Value = serde_json::from_str(body).map_err(|e| {
        ToolError::execution_failed(format!("Vision API returned invalid JSON: {e}"))
    })?;
    let content = json
        .pointer("/choices/0/message/content")
        .and_then(|c| c.as_str())
        .unwrap_or_default()
        .to_string();
    let truncated = json
        .pointer("/choices/0/finish_reason")
        .and_then(|f| f.as_str())
        == Some("length");
    Ok((content, truncated))
}

pub struct ImageAnalyzeTool {
    config: VisionModelConfig,
    client: reqwest::Client,
}

impl ImageAnalyzeTool {
    #[must_use]
    pub fn new(config: VisionModelConfig) -> Self {
        // Pinvou fork:不设 reqwest 整体 timeout(整体超时会把流掐断,拿不到
        // 部分内容);总时长由 execute 的总 deadline(默认 90s,见
        // DEFAULT_VISION_REQUEST_TIMEOUT_SECS)控制,超时返回已累积内容
        // (truncated 标记)。连接阶段仍限 30s 防挂死。
        let client = crate::tls::reqwest_client_builder()
            .connect_timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to build HTTP client");
        Self { config, client }
    }

    async fn read_image_file(path: &Path) -> Result<(String, String), ToolError> {
        let bytes = tokio::fs::read(path)
            .await
            .map_err(|e| ToolError::execution_failed(format!("Failed to read image file: {e}")))?;

        let mime_type = Self::detect_mime_type(path)?;
        let base64_data = BASE64.encode(&bytes);
        Ok((base64_data, mime_type))
    }

    fn resolve_image_path(workspace: &Path, image_path: &str) -> Result<PathBuf, ToolError> {
        let image_path_buf = Path::new(image_path);
        if image_path_buf.components().any(|c| {
            matches!(
                c,
                Component::Prefix(_) | Component::RootDir | Component::ParentDir
            )
        }) {
            return Err(ToolError::execution_failed(
                "image_path must be a relative path within the workspace and cannot escape it.",
            ));
        }

        let workspace = workspace.canonicalize().map_err(|e| {
            ToolError::execution_failed(format!("Failed to resolve workspace path: {e}"))
        })?;
        let candidate = workspace.join(image_path_buf);
        let resolved = candidate.canonicalize().map_err(|e| {
            ToolError::execution_failed(format!("Failed to resolve image file: {e}"))
        })?;
        if !resolved.starts_with(&workspace) {
            return Err(ToolError::execution_failed(
                "image_path must resolve within the workspace and cannot escape it.",
            ));
        }
        Ok(resolved)
    }

    fn detect_mime_type(path: &Path) -> Result<String, ToolError> {
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match extension.as_str() {
            "png" => Ok("image/png".to_string()),
            "jpg" | "jpeg" => Ok("image/jpeg".to_string()),
            "gif" => Ok("image/gif".to_string()),
            "webp" => Ok("image/webp".to_string()),
            "bmp" => Ok("image/bmp".to_string()),
            _ => Err(ToolError::execution_failed(format!(
                "Unsupported image format: {extension}"
            ))),
        }
    }

    fn base_url(&self) -> String {
        self.config
            .base_url
            .clone()
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string())
    }

    fn api_key(&self) -> String {
        self.config.api_key.clone().unwrap_or_default()
    }

    fn is_xiaomi_mimo_model(model: &str) -> bool {
        let normalized = model.trim().to_ascii_lowercase();
        let normalized = normalized.strip_prefix("xiaomi/").unwrap_or(&normalized);
        normalized.starts_with("mimo-")
    }

    fn uses_max_completion_tokens(config: &VisionModelConfig) -> bool {
        if Self::is_xiaomi_mimo_model(&config.model) {
            return true;
        }

        let base_url = config.base_url.as_deref().unwrap_or_default();
        let Ok(url) = reqwest::Url::parse(base_url) else {
            return false;
        };
        let Some(domain) = url.domain() else {
            return false;
        };

        domain.eq_ignore_ascii_case("xiaomimimo.com")
            || domain.to_ascii_lowercase().ends_with(".xiaomimimo.com")
    }

    fn request_payload(&self, prompt: &str, image_data: &str, mime_type: &str) -> Value {
        // Pinvou fork:system prompt 约束转写纪律(小模型对引导敏感)+
        // temperature 0.7 → 0.2(转写类任务低温度减少幻觉);两者均可由
        // VisionModelConfig.system_prompt / .temperature 覆盖(应用层注入)。
        let system_prompt = self.config.system_prompt.as_deref().unwrap_or(
            "You are an image content analyst. Describe the image accurately: \
             1) the image type (screenshot / photo / document / chart); \
             2) all visible text, transcribed verbatim in its original language; \
             3) key visual elements and layout. Never fabricate details you cannot see.",
        );
        let mut payload = json!({
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": format!("data:{};base64,{}", mime_type, image_data)
                            }
                        }
                    ]
                }
            ],
            "temperature": self.config.temperature.unwrap_or(0.2)
        });

        let token_limit_field = if Self::uses_max_completion_tokens(&self.config) {
            "max_completion_tokens"
        } else {
            "max_tokens"
        };
        payload[token_limit_field] = json!(
            self.config
                .max_output_tokens
                .unwrap_or(DEFAULT_VISION_MAX_OUTPUT_TOKENS)
        );
        // Pinvou fork:默认流式接收——超时时返回已生成的部分内容而非整请求失败;
        // 端点不支持 SSE 时可由 VisionModelConfig.stream 关闭走普通 JSON 解析。
        payload["stream"] = json!(self.config.stream.unwrap_or(true));

        payload
    }

    // Pinvou fork:流式累积——字节行缓冲解析 SSE(data: {json}),deadline 到
    // 或流中读取错误都返回已累积内容 + truncated(不再 hard-fail 丢弃全部)。
    async fn execute_streaming(
        &self,
        response: reqwest::Response,
        deadline: std::time::Instant,
    ) -> Result<ToolResult, ToolError> {
        let mut content = String::new();
        let mut truncated = false;
        let mut saw_data_event = false;
        let mut line_buffer = SseLineBuffer::default();
        let mut stream = response.bytes_stream();
        'stream: loop {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                truncated = true;
                break;
            }
            let chunk = match tokio::time::timeout(remaining, stream.next()).await {
                Ok(Some(Ok(chunk))) => chunk,
                Ok(Some(Err(_))) => {
                    truncated = true;
                    break;
                }
                Ok(None) => break, // 流正常结束
                Err(_) => {
                    truncated = true;
                    break;
                }
            };
            for line in line_buffer.push(&chunk) {
                if process_sse_line(&line, &mut content, &mut truncated, &mut saw_data_event) {
                    break 'stream;
                }
            }
        }
        // 末尾无换行的残留行也按一行处理,不丢最后一段 delta。
        if let Some(line) = line_buffer.finish() {
            let _ = process_sse_line(&line, &mut content, &mut truncated, &mut saw_data_event);
        }
        // Pinvou fork:非流式兜底——流正常结束但未解析到任何 SSE data 事件且
        // 内容为空,说明端点忽略了 stream 参数(返回普通 JSON);报错以区分
        // 「端点不支持流式」与「图无内容」(前者应配置 stream = false)。
        if !truncated && !saw_data_event && content.is_empty() {
            return Err(ToolError::execution_failed(
                "Vision API returned no SSE data events; the endpoint may ignore the \
                 stream parameter (set stream = false for non-streaming endpoints).",
            ));
        }
        self.vision_result(content, truncated)
    }

    // Pinvou fork:非流式接收(stream: Some(false))——解析普通 JSON 响应;
    // 无流式 deadline 语义,但读取 body 仍受总超时约束。
    async fn execute_non_streaming(
        &self,
        response: reqwest::Response,
        deadline: std::time::Instant,
    ) -> Result<ToolResult, ToolError> {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        if remaining.is_zero() {
            return self.vision_result(String::new(), true);
        }
        let body = match tokio::time::timeout(remaining, response.text()).await {
            Ok(Ok(body)) => body,
            Ok(Err(e)) => {
                return Err(ToolError::execution_failed(format!(
                    "Vision API request failed: {e}"
                )));
            }
            Err(_) => return self.vision_result(String::new(), true),
        };
        let (content, truncated) = parse_non_streaming_body(&body)?;
        self.vision_result(content, truncated)
    }

    fn vision_result(&self, content: String, truncated: bool) -> Result<ToolResult, ToolError> {
        // 流式响应无 model 字段,回退配置值(与上游非流式行为一致)。
        let mut result = json!({
            "analysis": content,
            "model": self.config.model,
        });
        if truncated {
            result["truncated"] = json!(true);
        }

        ToolResult::json(&result)
            .map_err(|e| ToolError::execution_failed(format!("Failed to serialize result: {e}")))
    }
}

#[async_trait]
impl ToolSpec for ImageAnalyzeTool {
    fn name(&self) -> &str {
        "image_analyze"
    }

    fn description(&self) -> &str {
        "Analyze an image using the configured vision model. \
         Supports PNG, JPEG, GIF, WebP, and BMP formats."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the image file to analyze"
                },
                "prompt": {
                    "type": "string",
                    "description": "Optional prompt to guide the analysis."
                }
            },
            "required": ["image_path"]
        })
    }

    fn capabilities(&self) -> Vec<ToolCapability> {
        vec![ToolCapability::ReadOnly]
    }

    async fn execute(&self, input: Value, context: &ToolContext) -> Result<ToolResult, ToolError> {
        let image_path = required_str(&input, "image_path")?;
        // Pinvou fork:默认提示词增强(主模型不传时质量兜底):
        // 类型/文字转写/元素布局三要素,与 system prompt 同向;可由
        // VisionModelConfig.default_prompt 覆盖(应用层注入)。
        let prompt = input
            .get("prompt")
            .and_then(|v| v.as_str())
            .or(self.config.default_prompt.as_deref())
            .unwrap_or(
                "Describe this image accurately: the image type (screenshot / photo / \
                 document / chart), all visible text transcribed verbatim, and the key \
                 visual elements and layout.",
            );

        let resolved_path = Self::resolve_image_path(&context.workspace, image_path)?;
        let (image_data, mime_type) = Self::read_image_file(&resolved_path).await?;

        let payload = self.request_payload(prompt, &image_data, &mime_type);

        let url = format!("{}/chat/completions", self.base_url());
        let api_key = self.api_key();

        // Pinvou fork:总预算 deadline 从 send() 之前起算(默认 90s,覆盖连接、
        // 等响应头和流式全程;send() 本身也用剩余时间包 timeout,修掉等响应头
        // 无界的坑)。deadline 到 → 返回已累积内容 + truncated,慢设备/长文档
        // 不会整请求失败触发重试死循环。
        let timeout_secs = self
            .config
            .request_timeout_secs
            .unwrap_or(DEFAULT_VISION_REQUEST_TIMEOUT_SECS);
        let deadline = std::time::Instant::now() + Duration::from_secs(timeout_secs);

        // Pinvou fork:有限重试——仅 is_retryable_status 的 HTTP 状态码重试
        // (最多 MAX_VISION_RETRIES 次,退避 1s→2s,尊重 Retry-After 封顶
        // 10s);连接失败/超时/流中断/其它状态码不重试。
        let mut attempt: u32 = 0;
        let response = loop {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                return self.vision_result(String::new(), true);
            }
            let send_result = tokio::time::timeout(
                remaining,
                self.client
                    .post(&url)
                    .header("Content-Type", "application/json")
                    .header("Authorization", format!("Bearer {api_key}"))
                    .json(&payload)
                    .send(),
            )
            .await;
            let response = match send_result {
                Ok(Ok(response)) => response,
                // 连接失败不重试,直接报错(现有语义)。
                Ok(Err(e)) => {
                    return Err(ToolError::execution_failed(format!(
                        "Vision API request failed: {e}"
                    )));
                }
                // 等响应头超时:不重试,按 deadline 语义返回空内容 + truncated。
                Err(_) => return self.vision_result(String::new(), true),
            };
            let status = response.status();
            if status.is_success() {
                break response;
            }
            if attempt < MAX_VISION_RETRIES && is_retryable_status(status.as_u16()) {
                attempt += 1;
                let backoff = retry_after_delay(response.headers())
                    .unwrap_or_else(|| Duration::from_secs(u64::from(attempt)));
                // 退避不超出总预算;预算耗尽后下一轮循环入口走 truncated 返回。
                let remaining = deadline.saturating_duration_since(std::time::Instant::now());
                tokio::time::sleep(backoff.min(remaining)).await;
                continue;
            }
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            let error_text =
                sanitize_http_error_body(Some("Vision provider"), status.as_u16(), &error_text);
            return Err(ToolError::execution_failed(format!(
                "Vision API request failed: {error_text}"
            )));
        };

        if self.config.stream.unwrap_or(true) {
            self.execute_streaming(response, deadline).await
        } else {
            self.execute_non_streaming(response, deadline).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[cfg(unix)]
    fn create_file_symlink(
        target: &std::path::Path,
        link: &std::path::Path,
    ) -> std::io::Result<()> {
        std::os::unix::fs::symlink(target, link)
    }

    #[cfg(windows)]
    fn create_file_symlink(
        target: &std::path::Path,
        link: &std::path::Path,
    ) -> std::io::Result<()> {
        std::os::windows::fs::symlink_file(target, link)
    }

    fn fake_config() -> VisionModelConfig {
        VisionModelConfig {
            model: "test-vision-model".to_string(),
            api_key: Some("test-key".to_string()),
            base_url: Some("https://example.invalid/v1".to_string()),
            system_prompt: None,
            default_prompt: None,
            max_output_tokens: None,
            temperature: None,
            request_timeout_secs: None,
            stream: None,
        }
    }

    #[test]
    fn tool_metadata_is_read_only_and_named_image_analyze() {
        let tool = ImageAnalyzeTool::new(fake_config());
        assert_eq!(tool.name(), "image_analyze");
        assert!(tool.capabilities().contains(&ToolCapability::ReadOnly));
    }

    #[test]
    fn mime_type_detection_covers_common_formats() {
        for (ext, expected) in [
            ("png", "image/png"),
            ("PNG", "image/png"),
            ("jpg", "image/jpeg"),
            ("jpeg", "image/jpeg"),
            ("gif", "image/gif"),
            ("webp", "image/webp"),
            ("bmp", "image/bmp"),
        ] {
            let path = std::path::PathBuf::from(format!("test.{ext}"));
            let mime = ImageAnalyzeTool::detect_mime_type(&path)
                .unwrap_or_else(|_| panic!("must detect {ext}"));
            assert_eq!(mime, expected);
        }
    }

    #[test]
    fn mime_type_detection_rejects_unsupported_extension() {
        let path = std::path::PathBuf::from("test.svg");
        let err = ImageAnalyzeTool::detect_mime_type(&path)
            .expect_err("svg is intentionally out of scope for vision tool");
        assert!(err.to_string().contains("Unsupported image format"));
    }

    #[test]
    fn generic_vision_payload_uses_max_tokens() {
        let tool = ImageAnalyzeTool::new(fake_config());

        let payload = tool.request_payload("describe", "abc123", "image/png");

        assert_eq!(
            payload.get("max_tokens").and_then(Value::as_u64),
            Some(u64::from(DEFAULT_VISION_MAX_OUTPUT_TOKENS))
        );
        assert!(payload.get("max_completion_tokens").is_none());
    }

    #[test]
    fn xiaomi_mimo_vision_payload_uses_max_completion_tokens() {
        let mut config = fake_config();
        config.model = "mimo-v2.5".to_string();
        config.base_url = Some("https://api.xiaomimimo.com/v1".to_string());
        let tool = ImageAnalyzeTool::new(config);

        let payload = tool.request_payload("describe", "abc123", "image/png");

        assert_eq!(
            payload.get("max_completion_tokens").and_then(Value::as_u64),
            Some(u64::from(DEFAULT_VISION_MAX_OUTPUT_TOKENS))
        );
        assert!(payload.get("max_tokens").is_none());
    }

    #[test]
    fn xiaomi_mimo_vision_payload_uses_max_completion_tokens_with_custom_proxy() {
        let mut config = fake_config();
        config.model = "mimo-v2.5".to_string();
        config.base_url = Some("https://vision-proxy.example.invalid/v1".to_string());
        let tool = ImageAnalyzeTool::new(config);

        let payload = tool.request_payload("describe", "abc123", "image/png");

        assert_eq!(
            payload.get("max_completion_tokens").and_then(Value::as_u64),
            Some(u64::from(DEFAULT_VISION_MAX_OUTPUT_TOKENS))
        );
        assert!(payload.get("max_tokens").is_none());
    }

    #[tokio::test]
    async fn execute_rejects_absolute_path() {
        // Trust-boundary pin: image_path must stay inside the workspace
        // — an absolute path or a `..`-traversing path must reject
        // before any base64 / API call.
        let tmp = tempdir().expect("tempdir");
        let ctx = ToolContext::new(tmp.path().to_path_buf());
        let tool = ImageAnalyzeTool::new(fake_config());
        let outside_workspace = if cfg!(windows) {
            r"C:\Windows\System32\drivers\etc\hosts"
        } else {
            "/etc/hosts"
        };
        let err = tool
            .execute(json!({"image_path": outside_workspace}), &ctx)
            .await
            .expect_err("absolute path must reject");
        assert!(
            err.to_string()
                .contains("relative path within the workspace"),
            "error must call out the workspace boundary; got {err}"
        );
    }

    #[tokio::test]
    async fn execute_rejects_parent_dir_traversal() {
        let tmp = tempdir().expect("tempdir");
        let ctx = ToolContext::new(tmp.path().to_path_buf());
        let tool = ImageAnalyzeTool::new(fake_config());
        let err = tool
            .execute(json!({"image_path": "../escape.png"}), &ctx)
            .await
            .expect_err("`..`-traversal must reject");
        assert!(
            err.to_string()
                .contains("relative path within the workspace"),
            "error must call out the workspace boundary; got {err}"
        );
    }

    #[tokio::test]
    async fn execute_rejects_symlink_that_resolves_outside_workspace() {
        let workspace = tempdir().expect("workspace tempdir");
        let outside = tempdir().expect("outside tempdir");
        let outside_image = outside.path().join("outside.png");
        std::fs::write(&outside_image, b"not a real png").expect("write outside image");
        let link = workspace.path().join("linked.png");
        if let Err(err) = create_file_symlink(&outside_image, &link) {
            eprintln!("skipping symlink assertion: {err}");
            return;
        }

        let ctx = ToolContext::new(workspace.path().to_path_buf());
        let tool = ImageAnalyzeTool::new(fake_config());
        let err = tool
            .execute(json!({"image_path": "linked.png"}), &ctx)
            .await
            .expect_err("symlink target outside workspace must reject before reading");
        assert!(
            err.to_string().contains("resolve within the workspace"),
            "error must call out the canonical workspace boundary; got {err}"
        );
    }

    #[test]
    fn retryable_status_covers_only_transient_server_errors() {
        for status in [429, 499, 500, 502, 503, 504] {
            assert!(is_retryable_status(status), "{status} must be retryable");
        }
        for status in [400, 401, 403, 404, 408, 413, 418, 422, 501] {
            assert!(
                !is_retryable_status(status),
                "{status} must not be retryable"
            );
        }
    }

    #[test]
    fn payload_falls_back_to_builtin_defaults_when_unset() {
        let tool = ImageAnalyzeTool::new(fake_config());

        let payload = tool.request_payload("describe", "abc123", "image/png");

        let system = payload
            .pointer("/messages/0/content")
            .and_then(Value::as_str)
            .expect("system prompt");
        assert!(system.contains("image content analyst"));
        assert_eq!(
            payload
                .get("temperature")
                .and_then(Value::as_f64)
                .map(|v| v as f32),
            Some(0.2)
        );
        assert_eq!(payload.get("stream").and_then(Value::as_bool), Some(true));
        assert_eq!(
            payload.get("max_tokens").and_then(Value::as_u64),
            Some(u64::from(DEFAULT_VISION_MAX_OUTPUT_TOKENS))
        );
    }

    #[test]
    fn payload_honors_config_overrides() {
        let mut config = fake_config();
        config.system_prompt = Some("custom system prompt".to_string());
        config.temperature = Some(0.7);
        config.max_output_tokens = Some(1024);
        config.stream = Some(false);
        let tool = ImageAnalyzeTool::new(config);

        let payload = tool.request_payload("describe", "abc123", "image/png");

        assert_eq!(
            payload
                .pointer("/messages/0/content")
                .and_then(Value::as_str),
            Some("custom system prompt")
        );
        assert_eq!(
            payload
                .get("temperature")
                .and_then(Value::as_f64)
                .map(|v| v as f32),
            Some(0.7)
        );
        assert_eq!(
            payload.get("max_tokens").and_then(Value::as_u64),
            Some(1024)
        );
        assert_eq!(payload.get("stream").and_then(Value::as_bool), Some(false));
    }

    #[test]
    fn sse_line_buffer_decodes_multibyte_chars_split_across_chunks() {
        // 「汉」UTF-8 为 3 字节(E6 B1 89):把 SSE 行切在它的第 1/2 字节之间,
        // 跨 chunk 解码必须不丢不乱(逐 chunk from_utf8_lossy 会产生替换字符)。
        let line = "data: {\"choices\":[{\"delta\":{\"content\":\"汉字\"}}]}\n";
        let bytes = line.as_bytes();
        let split = line.find('汉').expect("test line must contain 汉") + 1;

        let mut buffer = SseLineBuffer::default();
        assert!(
            buffer.push(&bytes[..split]).is_empty(),
            "no complete line yet"
        );
        let lines = buffer.push(&bytes[split..]);
        assert_eq!(lines.len(), 1);

        let mut content = String::new();
        let mut truncated = false;
        let mut saw_data_event = false;
        assert!(!process_sse_line(
            &lines[0],
            &mut content,
            &mut truncated,
            &mut saw_data_event
        ));
        assert_eq!(content, "汉字");
        assert!(saw_data_event);
        assert!(!truncated);
        assert!(buffer.finish().is_none());
    }

    #[test]
    fn sse_line_buffer_finish_returns_trailing_line_without_newline() {
        let mut buffer = SseLineBuffer::default();
        assert!(buffer.push(b"data: [DONE]").is_empty());
        let line = buffer.finish().expect("trailing line");
        let mut content = String::new();
        let mut truncated = false;
        let mut saw_data_event = false;
        assert!(
            process_sse_line(&line, &mut content, &mut truncated, &mut saw_data_event),
            "[DONE] must signal end of stream"
        );
        assert!(buffer.finish().is_none());
    }

    #[test]
    fn plain_json_body_is_not_counted_as_sse_data_event() {
        // 端点忽略 stream 参数返回普通 JSON:行不以 data: 开头,saw_data_event
        // 保持 false —— execute 据此报错,与「图无内容」(有 data 事件但无
        // content delta)区分。
        let mut content = String::new();
        let mut truncated = false;
        let mut saw_data_event = false;
        assert!(!process_sse_line(
            "{\"choices\":[{\"message\":{\"content\":\"x\"}}]}",
            &mut content,
            &mut truncated,
            &mut saw_data_event
        ));
        assert!(!saw_data_event, "plain JSON must not count as SSE data");
        assert!(content.is_empty());

        assert!(!process_sse_line(
            "data: {\"choices\":[{\"delta\":{}}]}",
            &mut content,
            &mut truncated,
            &mut saw_data_event
        ));
        assert!(saw_data_event, "data line must count even without delta");
    }

    #[test]
    fn non_streaming_body_parses_content_and_length_truncation() {
        let (content, truncated) = parse_non_streaming_body(
            r#"{"choices":[{"message":{"content":"abc"},"finish_reason":"length"}]}"#,
        )
        .expect("valid body");
        assert_eq!(content, "abc");
        assert!(truncated, "finish_reason length must mark truncated");

        let (content, truncated) = parse_non_streaming_body(
            r#"{"choices":[{"message":{"content":"abc"},"finish_reason":"stop"}]}"#,
        )
        .expect("valid body");
        assert_eq!(content, "abc");
        assert!(!truncated);

        assert!(parse_non_streaming_body("not json").is_err());
    }

    // 一次性 HTTP 测试服务器:接收一个请求,把请求体 JSON 经 oneshot 送回,
    // 然后以固定状态行 + body 响应并关闭连接。
    async fn serve_once(
        status_line: &str,
        body: String,
    ) -> (String, tokio::sync::oneshot::Receiver<Value>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test listener");
        let addr = listener.local_addr().expect("local addr");
        let (tx, rx) = tokio::sync::oneshot::channel();
        let status_line = status_line.to_string();
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let mut request: Vec<u8> = Vec::new();
            let mut buf = [0u8; 8192];
            let body_bytes = loop {
                let n = socket.read(&mut buf).await.expect("read request");
                if n == 0 {
                    return;
                }
                request.extend_from_slice(&buf[..n]);
                let Some(headers_end) = request.windows(4).position(|w| w == b"\r\n\r\n") else {
                    continue;
                };
                let headers = String::from_utf8_lossy(&request[..headers_end]).to_lowercase();
                let content_length = headers
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .and_then(|v| v.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                if request.len() >= headers_end + 4 + content_length {
                    break request[headers_end + 4..headers_end + 4 + content_length].to_vec();
                }
            };
            let _ = tx.send(serde_json::from_slice::<Value>(&body_bytes).unwrap_or(Value::Null));
            let response = format!(
                "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = socket.write_all(response.as_bytes()).await;
        });
        (format!("http://{addr}/v1"), rx)
    }

    fn workspace_with_image() -> (tempfile::TempDir, ToolContext) {
        let tmp = tempdir().expect("tempdir");
        std::fs::write(tmp.path().join("img.png"), b"tiny-png").expect("write image");
        let ctx = ToolContext::new(tmp.path().to_path_buf());
        (tmp, ctx)
    }

    fn result_json(result: ToolResult) -> Value {
        serde_json::from_str(&result.content).expect("tool result must be JSON")
    }

    #[tokio::test]
    async fn execute_streaming_accumulates_sse_deltas() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"你好\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"，世界\"}}]}\n\n",
            "data: [DONE]\n\n",
        )
        .to_string();
        let (base_url, request_rx) = serve_once("HTTP/1.1 200 OK", body).await;
        let (_tmp, ctx) = workspace_with_image();
        let mut config = fake_config();
        config.base_url = Some(base_url);
        config.default_prompt = Some("默认提示词覆盖".to_string());
        let tool = ImageAnalyzeTool::new(config);

        let result = tool
            .execute(json!({"image_path": "img.png"}), &ctx)
            .await
            .expect("streaming execute must succeed");
        let value = result_json(result);
        assert_eq!(
            value.get("analysis").and_then(Value::as_str),
            Some("你好，世界")
        );
        assert!(value.get("truncated").is_none());

        let request = request_rx.await.expect("server must receive request");
        assert_eq!(request.get("stream").and_then(Value::as_bool), Some(true));
        assert_eq!(
            request
                .pointer("/messages/1/content/0/text")
                .and_then(Value::as_str),
            Some("默认提示词覆盖"),
            "config.default_prompt must be used when input has no prompt"
        );
    }

    #[tokio::test]
    async fn execute_streaming_errors_when_endpoint_returns_plain_json() {
        // 非流式兜底:流正常结束但无任何 SSE data 事件且内容为空 → 报错,
        // 区分「端点忽略 stream 参数」与「图无内容」。
        let body = r#"{"choices":[{"message":{"content":"plain json"}}]}"#.to_string();
        let (base_url, _request_rx) = serve_once("HTTP/1.1 200 OK", body).await;
        let (_tmp, ctx) = workspace_with_image();
        let mut config = fake_config();
        config.base_url = Some(base_url);
        let tool = ImageAnalyzeTool::new(config);

        let err = tool
            .execute(json!({"image_path": "img.png"}), &ctx)
            .await
            .expect_err("plain JSON body in streaming mode must error");
        assert!(
            err.to_string().contains("no SSE data events"),
            "error must call out missing SSE data; got {err}"
        );
    }

    #[tokio::test]
    async fn execute_non_streaming_parses_plain_json_response() {
        let body =
            r#"{"choices":[{"message":{"content":"图片内容描述"},"finish_reason":"length"}]}"#
                .to_string();
        let (base_url, request_rx) = serve_once("HTTP/1.1 200 OK", body).await;
        let (_tmp, ctx) = workspace_with_image();
        let mut config = fake_config();
        config.base_url = Some(base_url);
        config.stream = Some(false);
        let tool = ImageAnalyzeTool::new(config);

        let result = tool
            .execute(json!({"image_path": "img.png"}), &ctx)
            .await
            .expect("non-streaming execute must succeed");
        let value = result_json(result);
        assert_eq!(
            value.get("analysis").and_then(Value::as_str),
            Some("图片内容描述")
        );
        assert_eq!(
            value.get("truncated").and_then(Value::as_bool),
            Some(true),
            "finish_reason length must mark truncated"
        );

        let request = request_rx.await.expect("server must receive request");
        assert_eq!(request.get("stream").and_then(Value::as_bool), Some(false));
    }

    #[tokio::test]
    async fn execute_retries_retryable_status_then_succeeds() {
        // 有限重试:503 + Retry-After: 0 → 同一工具再次请求即成功。
        // 第一次响应 503,第二次 200 SSE。
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test listener");
        let addr = listener.local_addr().expect("local addr");
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let responses = [
                "HTTP/1.1 503 Service Unavailable\r\nRetry-After: 0\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_string(),
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    "data: [DONE]\n\n".len() + "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n".len(),
                    "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n"
                ),
            ];
            for response in responses {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let mut buf = [0u8; 8192];
                let mut request: Vec<u8> = Vec::new();
                // 读完整请求(头 + body)再响应,避免客户端写 body 时被 RST。
                loop {
                    let n = socket.read(&mut buf).await.expect("read request");
                    if n == 0 {
                        break;
                    }
                    request.extend_from_slice(&buf[..n]);
                    let Some(headers_end) = request.windows(4).position(|w| w == b"\r\n\r\n")
                    else {
                        continue;
                    };
                    let headers = String::from_utf8_lossy(&request[..headers_end]).to_lowercase();
                    let content_length = headers
                        .lines()
                        .find_map(|l| l.strip_prefix("content-length:"))
                        .and_then(|v| v.trim().parse::<usize>().ok())
                        .unwrap_or(0);
                    if request.len() >= headers_end + 4 + content_length {
                        break;
                    }
                }
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });
        let (_tmp, ctx) = workspace_with_image();
        let mut config = fake_config();
        config.base_url = Some(format!("http://{addr}/v1"));
        let tool = ImageAnalyzeTool::new(config);

        let result = tool
            .execute(json!({"image_path": "img.png"}), &ctx)
            .await
            .expect("retryable 503 must be retried and then succeed");
        let value = result_json(result);
        assert_eq!(value.get("analysis").and_then(Value::as_str), Some("ok"));
    }
}
