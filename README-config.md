# Configuration

## Environment Variables

API keys are read directly from environment variables:

- `OPENAI_API_KEY`: API key for OpenAI models
- `ANTHROPIC_API_KEY`: API key for Anthropic models
- `GOOGLE_API_KEY`: API key for Google models
- `MISTRAL_API_KEY`: API key for Mistral models
- `COHERE_API_KEY`: API key for Cohere models

## Rate Limit Configuration

Rate limits are configured in YAML files. The system will look for a `rate_limits.yaml` file in:

1. The current working directory
2. The `~/.vlm_autoeval/` directory
3. The package config directory

Example `rate_limits.yaml`:

```yaml
openai:
  gpt-4-vision-preview:
    requests_per_minute: 100
    tokens_per_minute: 100000
    concurrent_requests: 50
  gpt-4-turbo-preview:
    requests_per_minute: 100
    tokens_per_minute: 100000
    concurrent_requests: 50

anthropic:
  claude-3-opus-20240229:
    requests_per_minute: 100
    tokens_per_minute: 100000
    concurrent_requests: 50
  claude-3-sonnet-20240229:
    requests_per_minute: 100
    tokens_per_minute: 100000
    concurrent_requests: 50
```

If no configuration file is found, the system will use default rate limits defined in the code. 