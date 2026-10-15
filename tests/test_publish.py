import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from edge_vision.config import PublishConfig
from edge_vision.publish import HttpPublisher, MqttPublisher, build_publisher


class _Handler(BaseHTTPRequestHandler):
    body: dict[str, Any] = {}

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        _Handler.body = {
            "payload": json.loads(raw.decode("utf-8")),
            "path": self.path,
            "content_type": self.headers.get("Content-Type"),
        }
        self.send_response(204)
        self.end_headers()

    def log_message(self, fmt: str, *args: object) -> None:
        return None


def test_http_publisher_posts_json() -> None:
    _Handler.body = {}
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        publisher = HttpPublisher(f"http://127.0.0.1:{port}/events", timeout_s=2)
        publisher.publish({"event": "zone_enter", "track_id": 4})
        publisher.close()
    finally:
        server.shutdown()
        thread.join(timeout=2)
    assert _Handler.body["payload"] == {"event": "zone_enter", "track_id": 4}
    assert _Handler.body["path"] == "/events"
    assert _Handler.body["content_type"] == "application/json"


def test_http_failure_does_not_raise() -> None:
    publisher = HttpPublisher("http://127.0.0.1:9/events", timeout_s=0.5)
    publisher.publish({"event": "zone_enter"})
    publisher.close()


class _FakeMqtt:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bytes, int]] = []

    def publish(self, topic: str, payload: bytes, qos: int = 0) -> None:
        self.calls.append((topic, payload, qos))


def test_mqtt_publisher_uses_the_injected_client() -> None:
    client = _FakeMqtt()
    publisher = MqttPublisher("127.0.0.1", 1883, "edge/alerts", qos=1, client=client)  # type: ignore[arg-type]
    publisher.publish({"event": "zone_exit", "track_id": 2})
    publisher.close()
    topic, payload, qos = client.calls[0]
    assert topic == "edge/alerts"
    assert qos == 1
    assert json.loads(payload.decode("utf-8"))["track_id"] == 2


def test_null_publisher() -> None:
    publisher = build_publisher(PublishConfig(mode="none"))
    publisher.publish({"event": "zone_enter"})
    publisher.close()
