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
// 4096 时由流式超时(300s)截断返回部分内容(truncated 标记),不会重试死循环。
const DEFAULT_VISION_MAX_OUTPUT_TOKENS: u32 = 4096;

pub struct ImageAnalyzeTool {
    config: VisionModelConfig,
    client: reqwest::Client,
}

impl ImageAnalyzeTool {
    #[must_use]
    pub fn new(config: VisionModelConfig) -> Self {
        // Pinvou fork:流式接收后不再设整体 timeout(reqwest 整体超时会在
        // 300s 掐断流,拿不到部分内容);总时长由 execute 的流式循环 300s
        // 控制,超时返回已累积内容(truncated 标记)。连接阶段仍限 30s 防挂死。
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
        // temperature 0.7 → 0.2(转写类任务低温度减少幻觉)。
        let mut payload = json!({
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an image content analyst. Describe the image accurately: \
                       1) the image type (screenshot / photo / document / chart); \
                       2) all visible text, transcribed verbatim in its original language; \
                       3) key visual elements and layout. Never fabricate details you cannot see."
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
            "temperature": 0.2
        });

        let token_limit_field = if Self::uses_max_completion_tokens(&self.config) {
            "max_completion_tokens"
        } else {
            "max_tokens"
        };
        payload[token_limit_field] = json!(DEFAULT_VISION_MAX_OUTPUT_TOKENS);
        // Pinvou fork:流式接收——超时(300s)时返回已生成的部分内容而非整请求失败。
        payload["stream"] = json!(true);

        payload
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
        // 类型/文字转写/元素布局三要素,与 system prompt 同向。
        let prompt = input
            .get("prompt")
            .and_then(|v| v.as_str())
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

        // Pinvou fork:流式接收(stream: true,见 request_payload)。总时长 300s
        // 上限;超时返回已累积的部分内容(truncated 标记),慢设备/长文档不会
        // 整请求失败触发重试死循环。连接失败/HTTP 错误仍直接报错。
        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&payload)
            .send()
            .await
            .map_err(|e| ToolError::execution_failed(format!("Vision API request failed: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            let error_text =
                sanitize_http_error_body(Some("Vision provider"), status.as_u16(), &error_text);
            return Err(ToolError::execution_failed(format!(
                "Vision API request failed: {error_text}"
            )));
        }

        // 流式累积:行缓冲解析 SSE(data: {json}),超时截断时保留已累积内容。
        let mut content = String::new();
        let mut truncated = false;
        let deadline = std::time::Instant::now() + Duration::from_secs(300);
        let mut buffer = String::new();
        let mut stream = response.bytes_stream();
        'stream: loop {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                truncated = true;
                break;
            }
            let chunk = match tokio::time::timeout(remaining, stream.next()).await {
                Ok(Some(Ok(chunk))) => chunk,
                Ok(Some(Err(e))) => {
                    return Err(ToolError::execution_failed(format!(
                        "Vision stream error: {e}"
                    )))
                }
                Ok(None) => break, // 流正常结束
                Err(_) => {
                    truncated = true;
                    break;
                }
            };
            buffer.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(pos) = buffer.find('\n') {
                let line = buffer[..pos].trim().to_string();
                buffer = buffer[pos + 1..].to_string();
                if !line.starts_with("data:") {
                    continue;
                }
                let data = line[5..].trim();
                if data == "[DONE]" {
                    break 'stream;
                }
                let Ok(event) = serde_json::from_str::<Value>(data) else {
                    continue;
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
                    truncated = true;
                }
            }
        }

        // 流式响应无 model 字段,回退配置值(与上游非流式行为一致)。
        let model = self.config.model.clone();

        let mut result = json!({
            "analysis": content,
            "model": model,
        });
        if truncated {
            result["truncated"] = json!(true);
        }

        ToolResult::json(&result)
            .map_err(|e| ToolError::execution_failed(format!("Failed to serialize result: {e}")))
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
}
