from __future__ import annotations

import json
import logging
from typing import Any, Literal, Protocol
from urllib import error, request

import paho.mqtt.client as mqtt

from edge_vision.config import PublishConfig

logger = logging.getLogger(__name__)

PublishMode = Literal["none", "http", "mqtt"]


class Publisher(Protocol):
    def publish(self, event: dict[str, Any]) -> None: ...

    def close(self) -> None: ...


class NullPublisher:
    def publish(self, event: dict[str, Any]) -> None:
        return None

    def close(self) -> None:
        return None


class HttpPublisher:
    """POST each event as JSON. A dead endpoint is logged and does not stop the camera loop."""

    def __init__(self, url: str, timeout_s: float = 2.0) -> None:
        self._url = url
        self._timeout_s = timeout_s

    def publish(self, event: dict[str, Any]) -> None:
        payload = json.dumps(event).encode("utf-8")
        req = request.Request(
            self._url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self._timeout_s) as response:
                response.read()
        except error.URLError as exc:
            logger.warning("http publish failed: %s", exc)

    def close(self) -> None:
        return None


class MqttPublisher:
    """Publish JSON payloads on one topic. Connection errors surface at startup."""

    def __init__(
        self,
        host: str,
        port: int,
        topic: str,
        qos: int = 0,
        client: mqtt.Client | None = None,
    ) -> None:
        self._topic = topic
        self._qos = qos
        self._owns_client = client is None
        if client is None:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            client.connect(host, port, keepalive=30)
            client.loop_start()
        self._client = client

    def publish(self, event: dict[str, Any]) -> None:
        payload = json.dumps(event).encode("utf-8")
        try:
            self._client.publish(self._topic, payload, qos=self._qos)
        except Exception as exc:
            logger.warning("mqtt publish failed: %s", exc)

    def close(self) -> None:
        if not self._owns_client:
            return
        self._client.loop_stop()
        self._client.disconnect()
        self._owns_client = False


def build_publisher(cfg: PublishConfig, client: mqtt.Client | None = None) -> Publisher:
    mode = cfg.mode
    if mode == "none":
        return NullPublisher()
    if mode == "http":
        return HttpPublisher(cfg.http_url, timeout_s=cfg.http_timeout_s)
    if mode == "mqtt":
        return MqttPublisher(
            cfg.mqtt_host,
            cfg.mqtt_port,
            cfg.mqtt_topic,
            qos=cfg.mqtt_qos,
            client=client,
        )
    _never: never = mode
    raise ValueError(f"unsupported publish mode {_never}")
