# Notify Slack with ZenML v3 Alerter

ZenML (v3)’s alerting system is built around the **Alerter** stack component, which lets you send automated messages (or “asks”) from within your pipelines. The most common use case is the **SlackAlerter**, which plugs directly into any ZenML stack and posts to a channel or prompts a user for approval.

**How alerting works in ZenML v3**

1. **Alerter components** live alongside your orchestrator, artifact store, experiment tracker, etc., in your stack definition.
2. During a pipeline run, you can fetch the active alerter via the client (e.g. `Client().active_stack.alerter`) and call:

   - `alerter.post("…")` to send a notification
   - `alerter.ask("…")` to ask a yes/no question and await a response (returns a boolean) ([docs.zenml.io][1])

3. Under the hood, ZenML serializes the message payload and hands it off to the Slack API (via the Slack SDK), using the credentials and channel you configured in the stack component’s settings.

---

**Registering the SlackAlerter in your local stack**

1. **Create a Slack App** in your workspace and grant it these OAuth scopes:

   - `chat:write`
   - `channels:read`
   - `channels:history`
     Then invite the app to your target channel. ([docs.zenml.io][1])

2. **Install the Slack integration** in your environment:

   ```bash
   zenml integration install slack -y
   ```

3. **Store your bot token securely** as a ZenML secret:

   ```bash
   zenml secret create slack_token --oauth_token=<YOUR_SLACK_BOT_TOKEN>
   ```

4. **Register the SlackAlerter** component, referencing that secret and your channel ID:

   ```bash
   zenml alerter register slack_alerter \
     --flavor=slack \
     --slack_token={{slack_token.oauth_token}} \
     --slack_channel_id=<SLACK_CHANNEL_ID>
   ```

   - The `{{secret_name.key}}` syntax ensures your token isn’t stored in plaintext. ([docs.zenml.io][1])

5. **Add it to your local stack** (replacing the placeholder stack name with your own):

   ```bash
   zenml stack register my_local_stack \
     --orchestrator=local_docker \
     --artifact_store=local \
     --experiment_tracker=local \
     --alerter=slack_alerter \
     --set
   ```

   Now any pipeline you run on `my_local_stack` can use `Client().active_stack.alerter.post()` or `.ask()` to send Slack notifications.

---

**Quick example usage in code**

```python
from zenml import pipeline, step
from zenml.client import Client

@step
def notify_step() -> None:
    client = Client()
    client.active_stack.alerter.post("✅ My pipeline step has completed!")

@pipeline(enable_cache=False)
def my_pipeline():
    notify_step()

if __name__ == "__main__":
    my_pipeline()
```

You’ll see “✅ My pipeline step has completed!” appear in your chosen Slack channel whenever that step runs.

[1]: https://docs.zenml.io/stacks/stack-components/alerters/slack "Slack Alerter | ZenML - Bridging the gap between ML & Ops"
[2]: https://docs.zenml.io/stacks/stack-components/alerters/discord "Discord Alerter | ZenML - Bridging the gap between ML & Ops"
