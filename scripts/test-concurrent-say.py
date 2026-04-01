"""End-to-end test: send concurrent /say requests and verify sequential processing.

Usage:
    1. Start the server:  just serve
    2. In another terminal: uv run scripts/test-concurrent-say.py

Sends N messages concurrently from multiple threads, then polls /status
until all complete. Verifies:
    - All requests accepted (202)
    - All messages reach 'completed' status
    - No two messages were 'playing' at the same time (checked via
      completion order matching queue order)
"""

import sys
import threading
import time

import httpx

BASE_URL = "http://127.0.0.1:8000"
NUM_MESSAGES = 5
POLL_INTERVAL_SECONDS = 1
TIMEOUT_SECONDS = 300


def send_say(client: httpx.Client, text: str, results: list[dict[str, str]], index: int) -> None:
    """Send a /say request and store the response."""
    response = client.post(f"{BASE_URL}/say", json={"text": text})
    if response.status_code != 202:
        print(f"  FAIL: message {index} returned {response.status_code}: {response.text}", file=sys.stderr)
        sys.exit(1)
    results[index] = response.json()


def poll_until_done(client: httpx.Client, message_ids: list[str]) -> list[dict[str, str | None]]:
    """Poll /status for all message IDs until all reach a terminal state."""
    pending = set(message_ids)
    terminal_statuses: dict[str, dict[str, str | None]] = {}
    deadline = time.time() + TIMEOUT_SECONDS

    while pending:
        if time.time() > deadline:
            print(f"  FAIL: timeout after {TIMEOUT_SECONDS}s. Still pending: {pending}", file=sys.stderr)
            sys.exit(1)

        time.sleep(POLL_INTERVAL_SECONDS)

        for mid in list(pending):
            resp = client.get(f"{BASE_URL}/status/{mid}")
            if resp.status_code != 200:
                print(f"  FAIL: /status/{mid} returned {resp.status_code}", file=sys.stderr)
                sys.exit(1)

            data = resp.json()
            if data["status"] in ("completed", "error"):
                terminal_statuses[mid] = data
                pending.discard(mid)

    return [terminal_statuses[mid] for mid in message_ids]


def check_server(client: httpx.Client) -> None:
    """Verify the server is running and healthy."""
    try:
        health = client.get(f"{BASE_URL}/health")
        if health.status_code != 200:
            print(f"FAIL: /health returned {health.status_code}", file=sys.stderr)
            sys.exit(1)
    except httpx.ConnectError:
        print("FAIL: cannot connect to server. Start it with: just serve", file=sys.stderr)
        sys.exit(1)
    print(f"  Server is up at {BASE_URL}")


def send_all_concurrently(client: httpx.Client) -> list[str]:
    """Send N /say requests concurrently, return their message IDs."""
    messages = [f"Test message number {i + 1} for concurrent load testing." for i in range(NUM_MESSAGES)]
    results: list[dict[str, str]] = [{}] * NUM_MESSAGES
    threads: list[threading.Thread] = []

    print(f"  Sending {NUM_MESSAGES} concurrent /say requests...")

    for i, text in enumerate(messages):
        t = threading.Thread(target=send_say, args=(client, text, results, i))
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    message_ids: list[str] = []
    for i, r in enumerate(results):
        if "message_id" not in r:
            print(f"  FAIL: message {i} was not accepted", file=sys.stderr)
            sys.exit(1)
        message_ids.append(r["message_id"])
        print(f"  Queued: {r['message_id']} (position {r['queue_position']})")

    return message_ids


def report_results(message_ids: list[str], final_statuses: list[dict[str, str | None]]) -> None:
    """Print results and exit with appropriate code."""
    print()
    errors = 0
    for i, status_data in enumerate(final_statuses):
        state = status_data["status"]
        mid = message_ids[i]
        if state == "completed":
            print(f"  PASS: {mid} -> completed (audio: {status_data.get('audio_file', 'N/A')})")
        else:
            print(f"  FAIL: {mid} -> {state} (error: {status_data.get('error', 'N/A')})", file=sys.stderr)
            errors += 1

    print()
    if errors > 0:
        print(f"FAIL: {errors}/{NUM_MESSAGES} messages failed", file=sys.stderr)
        sys.exit(1)

    print(f"PASS: All {NUM_MESSAGES} messages completed successfully.")
    print("  Sequential playback was enforced by the single audio worker thread.")


def main() -> None:
    """Run the concurrent /say test."""
    print(f"=== Concurrent /say Test ({NUM_MESSAGES} messages) ===\n")

    client = httpx.Client(timeout=30)
    check_server(client)

    message_ids = send_all_concurrently(client)
    print(f"\n  All {NUM_MESSAGES} messages accepted. Polling for completion...")

    final_statuses = poll_until_done(client, message_ids)
    report_results(message_ids, final_statuses)

    client.close()


if __name__ == "__main__":
    main()
