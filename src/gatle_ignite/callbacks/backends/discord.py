"""Discord webhook backend."""

import os

from gatle_ignite.callbacks.logging import Backend as BaseBackend
from gatle_ignite.dispatch import ConfigError


class Backend(BaseBackend):
    def __init__(self, cfg):
        super().__init__(cfg)
        try:
            import requests
        except ImportError as e:
            raise ConfigError(
                "logger_name includes 'discord' but requests is not installed. "
                "Install 'gatle-ignite[discord]' (or '.[discord]'), or drop 'discord' from logger_name."
            ) from e
        self._requests = requests

        self.url = cfg.get("discord_url", None) or os.environ.get("DISCORD_WEBHOOK_URL", "")
        if not self.url:
            raise ConfigError(
                "logger_name includes 'discord' but no webhook is configured. "
                "Set cfg.discord_url or the DISCORD_WEBHOOK_URL env var."
            )
        self.username = cfg.get("discord_username", "gatle")

    def _send(self, description, fields):
        payload = {
            "username": self.username,
            "embeds": [
                {
                    "title": self.cfg.name,
                    "description": description,
                    "fields": fields[:25],  # Discord caps embeds at 25 fields
                }
            ],
        }
        try:
            self._requests.post(self.url, json=payload, timeout=10)
        except Exception as e:  # noqa: BLE001 - logging must never kill a training run
            print(f"[discord] failed to send: {e}")

    def log(self, metrics, step=None, epoch=None):
        header = f"Epoch {epoch}" if epoch is not None else f"Step {step}"
        fields = [
            {"name": k, "value": f"{v:.5g}" if isinstance(v, float) else str(v), "inline": True}
            for k, v in sorted(metrics.items())
        ]
        self._send(f"Results for {header}", fields)

    def finish(self, failed=False):
        self._send("Run failed" if failed else "Run complete", [])
