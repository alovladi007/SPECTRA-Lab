#!/usr/bin/env python3
"""Quick WebSocket client test for CVD telemetry endpoint"""

import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8001/api/v1/cvd/ws/telemetry/338df4f3-1b50-41a9-9eaa-3bf88e413d94"
    print(f"Attempting to connect to: {uri}")

    try:
        async with websockets.connect(uri) as websocket:
            print("✓ WebSocket connection established!")

            # Receive a few messages
            for i in range(3):
                message = await websocket.recv()
                data = json.loads(message)
                print(f"Received message {i+1}: {json.dumps(data, indent=2)}")

    except websockets.exceptions.InvalidStatusCode as e:
        print(f"✗ Invalid status code: {e}")
    except ConnectionRefusedError:
        print(f"✗ Connection refused - server may not be running")
    except Exception as e:
        print(f"✗ WebSocket connection failed: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
