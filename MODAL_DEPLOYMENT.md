# Deploying the VLM Policy Server with Modal

This guide explains how to deploy the VLM AutoEval Robot Benchmark server to [Modal](https://modal.com/) as a web endpoint.

## Prerequisites

1. Install Modal CLI and authenticate:
   ```bash
   pip install modal
   modal token new
   ```

2. Create API key secrets in Modal:
   ```bash
   # OpenAI API key (required)
   modal secret create openai-api-key OPENAI_API_KEY=sk-your-key-here

   # Optional: Add other provider API keys as needed
   # modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-your-key-here
   # modal secret create gemini-api-key GEMINI_API_KEY=your-key-here
   ```

3. Make sure your local package is installable:
   ```bash
   pip install -e .
   ```

## Deployment Steps

1. Configure rate limits (optional but recommended):
   ```bash
   # Copy the example rate limits configuration
   cp rate_limits.yaml.example rate_limits.yaml

   # Edit the file to adjust rate limits for your needs
   nano rate_limits.yaml
   ```

2. Deploy to Modal:
   ```bash
   modal deploy modal_server.py
   ```

3. After deployment, Modal will provide a URL for your endpoint, typically in this format:
   ```
   https://username--vlm-robot-policy-server-act.modal.run
   ```

4. Test your deployment with a sample request:
   ```bash
   curl -X POST "https://username--vlm-robot-policy-server-act.modal.run" \
     -H "Content-Type: application/json" \
     -d '{"image": "base64_encoded_image_here", "instruction": "pick up the red cube"}'
   ```

## Rate Limiting Configuration

The server uses rate limiting to avoid exceeding API provider quotas. If you deploy with a `rate_limits.yaml` file, it will be copied to the Modal container and used to configure rate limits.

Example rate limits configuration (`rate_limits.yaml`):

```yaml
# OpenAI models
openai:
  models:
    # GPT-4o mini (default model)
    gpt-4o-mini:
      requests_per_minute: 60  # 1 request per second
      tokens_per_minute: 100000  # Max tokens per minute
      concurrent_requests: 5  # Max concurrent requests
```

Each model can have these rate limit parameters:
- `requests_per_minute`: Maximum number of requests allowed per minute
- `tokens_per_minute`: Maximum number of tokens allowed per minute
- `concurrent_requests`: Maximum number of concurrent requests

## Environment Variables

You can configure the deployment with these environment variables:

- `VLM_MODEL`: The VLM model to use (default: "gpt-4o-mini")
- `CONCURRENCY_LIMIT`: Maximum number of concurrent requests (default: 2)
- `TIMEOUT`: Maximum request timeout in seconds (default: 300)

To set environment variables in Modal:

```bash
modal app update vlm-robot-policy-server --env VLM_MODEL=gpt-4o
```

## Local Development

You can run the server locally for development:

```bash
modal serve modal_server.py
```

This will start the server locally and provide endpoints at:
- http://localhost:8000/act (POST)
- http://localhost:8000/health (GET)
- http://localhost:8000/ (GET)

## Troubleshooting

1. **Missing API Keys**: Ensure you've set up all necessary API key secrets in Modal.

2. **Package Installation Issues**: If the package fails to install in Modal, check your `setup.py` and dependencies.

3. **Action Bounds File**: If you get errors about missing `action_bounds.json`, verify the file path and that it's being correctly copied to the Modal image.

4. **Rate Limiting**: If you encounter rate limiting issues with the LLM providers, check your `rate_limits.yaml` file and adjust the limits.

5. **Memory or Timeout Issues**: Adjust the `CONCURRENCY_LIMIT` and `TIMEOUT` environment variables if needed.

## Scaling and Costs

- **Modal Free Tier**: The free tier includes a limited number of compute hours per month, which should be sufficient for light usage.
- **Cost Management**: Monitor your Modal usage to avoid unexpected charges when using paid plans.
- **Scaling**: Modal automatically scales your application based on demand, but make sure your LLM provider limits are set accordingly.

## Security Considerations

- **API Keys**: Your API keys are securely stored in Modal's secret management system.
- **Request Validation**: Consider adding additional request validation if exposing your endpoint publicly.
- **Access Control**: For production usage, consider adding authentication to your Modal web endpoints.
