# Configuration

## Environment Variables

API keys are read directly from environment variables:

- `OPENAI_API_KEY`: API key for OpenAI models
- `ANTHROPIC_API_KEY`: API key for Anthropic models
- `GEMINI_API_KEY`: API key for Google models
- `HUGGINGFACE_API_KEY`: API key for HuggingFace models

## Rate Limit Configuration

Rate limits are configured in YAML files. The system will look for a `rate_limits.yaml` file inthe current working directory.

Example `rate_limits.yaml`:

```yaml
openai:
  gpt-4-vision-preview:
    requests_per_minute: 100
    tokens_per_minute: 100000

anthropic:
  claude-3-opus-20240229:
    requests_per_minute: 100
    tokens_per_minute: 100000
```

If no configuration file is found, the system will use default rate limits defined in the code.
