import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from researchloop import server

CONFIG = {"cloud": {"enabled": True, "base_url": "https://api.example.com/v1", "model": "m", "api_key": "sk-stored-secret"},
          "web": {"enabled": False}}


class ServerSecurityTests(unittest.TestCase):
    def setUp(self):
        self.api = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=self.api.serve_forever, daemon=True).start()
        self.addCleanup(self.api.server_close)
        self.addCleanup(self.api.shutdown)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.api.server_port, timeout=5)
        self.addCleanup(connection.close)
        connection.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        values = {"Host": f"127.0.0.1:{self.api.server_port}", "Content-Type": "application/json", **(headers or {})}
        data = json.dumps(body).encode() if body is not None else b""
        values.setdefault("Content-Length", str(len(data)))
        for key, value in values.items():
            connection.putheader(key, value)
        connection.endheaders(data)
        response = connection.getresponse()
        return response.status, response.read()

    def test_rebound_host_is_rejected_for_reads_and_writes(self):
        rebound = {"Host": f"evil.example:{self.api.server_port}", "Origin": f"http://evil.example:{self.api.server_port}"}
        self.assertEqual(self.request("GET", "/api/state", headers=rebound)[0], 403)
        with patch.object(server, "settings", return_value=CONFIG), patch.object(server.cloudllm, "probe") as probe:
            status, _ = self.request("POST", "/api/cloud/test", {"base_url": "https://evil.example/v1"}, rebound)
        self.assertEqual(status, 403)
        probe.assert_not_called()

    def test_local_host_names_are_accepted(self):
        for host in ("127.0.0.1", "localhost"):
            status, _ = self.request("GET", "/favicon.svg", headers={"Host": f"{host}:{self.api.server_port}"})
            self.assertEqual(status, 200, host)

    def test_negative_content_length_is_rejected_immediately(self):
        status, _ = self.request("POST", "/api/cloud/test", headers={"Content-Length": "-1"})
        self.assertEqual(status, 400)

    def test_stored_cloud_key_is_not_sent_to_a_new_base_url(self):
        with patch.object(server, "settings", return_value=CONFIG), \
                patch.object(server.cloudllm, "probe", return_value={"available": True}) as probe:
            self.request("POST", "/api/cloud/test", {"base_url": "https://other.example/v1"})
            self.assertFalse(probe.call_args.args[1])
            self.request("POST", "/api/cloud/test", {"base_url": "https://api.example.com/v1/"})
            self.assertEqual(probe.call_args.args[1], "sk-stored-secret")


if __name__ == "__main__":
    unittest.main()
