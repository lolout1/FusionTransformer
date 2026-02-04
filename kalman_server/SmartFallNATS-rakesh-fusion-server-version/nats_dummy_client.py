#!/usr/bin/env python3
"""
Dummy sensor data generator + NATS load tester.

- Generates synthetic accelerometer payloads shaped like:
    [[[x, y, z], [x, y, z], ...]]
- Sends requests to NATS on subject m.<uuid>
- Waits for float response from inference server
- Measures latency per request

Usage examples:

# Single known user UUID, 100 messages, 0.2s interval
python nats_dummy_client.py \
    --uuid d53762d3-1451-4c8c-afca-116d263764b5 \
    --messages-per-user 100 \
    --interval 0.5

# 10 random dummy users, 50 messages each, 0.1s interval
python nats_dummy_client.py \
    --num-random-users 10 \
    --messages-per-user 50 \
    --interval 0.5
"""

import argparse
import asyncio
import json
import time
import uuid as uuidlib
from typing import List

import numpy as np
from nats.aio.client import Client as NATS
from nats.aio.errors import ErrTimeout


def generate_dummy_sequence(T: int = 120) -> list:
    """
    Generate dummy accelerometer data with shape (1, T, 3),

    Returns a Python list suitable for json.dumps:
        [[[x, y, z], [x, y, z], ...]]
    """
    t = np.arange(T, dtype=np.float32)

    # Simple synthetic 'motion' pattern with noise
    x = 6.0 + 2.0 * np.sin(2 * np.pi * t / 50.0) + np.random.normal(0, 0.5, size=T)
    y = -3.0 + 1.5 * np.sin(2 * np.pi * t / 40.0 + 0.7) + np.random.normal(0, 0.5, size=T)
    z = 9.0 + 2.5 * np.sin(2 * np.pi * t / 30.0 + 1.3) + np.random.normal(0, 0.5, size=T)

    seq = np.stack([x, y, z], axis=1)  # shape (T, 3)
    return [seq.tolist()]  # shape (1, T, 3) → [[[x,y,z], ...]]


async def run_user(
    nc: NATS,
    uuid_str: str,
    messages_per_user: int,
    interval: float,
    seq_len: int,
    timeout: float,
) -> None:
    """
    One logical "user": repeatedly send sensor payloads and print replies.
    """
    subject = f"m.{uuid_str}"
    print(f"[USER] UUID={uuid_str} subject={subject}")

    for i in range(messages_per_user):
        payload = generate_dummy_sequence(seq_len)
        data_bytes = json.dumps(payload).encode("utf-8")

        t0 = time.time()
        try:
            msg = await nc.request(subject, data_bytes, timeout=timeout)
            dt_ms = (time.time() - t0) * 1000.0

            raw = msg.data.decode("utf-8", errors="replace").strip()
            try:
                score = float(raw)
            except ValueError:
                score = None

            print(
                f"[{subject}] msg#{i+1:04d} "
                f"latency={dt_ms:.2f} ms "
                f"raw_reply='{raw}' "
                f"parsed_score={score}"
            )
        except ErrTimeout:
            dt_ms = (time.time() - t0) * 1000.0
            print(
                f"[{subject}] msg#{i+1:04d} TIMEOUT after {dt_ms:.2f} ms"
            )

        if interval > 0:
            await asyncio.sleep(interval)


async def main_async(args):
    nc = NATS()

    # Connect to NATS (plaintext)
    # e.g. nats://chocolatefrog@cssmartfall1.cose.txstate.edu:4224
    print(f"[NATS] Connecting to {args.server} ...")
    await nc.connect(
        servers=[args.server],
        max_reconnect_attempts=-1,
        reconnect_time_wait=0.5,
        connect_timeout=3.0,
    )
    print("[NATS] Connected.")

    # Build list of UUIDs to simulate
    uuids: List[str] = []

    if args.uuid:
        uuids.extend(args.uuid)

    if args.num_random_users > 0:
        for _ in range(args.num_random_users):
            uuids.append(str(uuidlib.uuid4()))

    if not uuids:
        raise SystemExit(
            "Error: You must provide at least one --uuid or --num-random-users > 0"
        )

    print(f"[LOAD] Simulating {len(uuids)} user(s)")

    # Create a task per user
    tasks = [
        asyncio.create_task(
            run_user(
                nc=nc,
                uuid_str=u,
                messages_per_user=args.messages_per_user,
                interval=args.interval,
                seq_len=args.sequence_len,
                timeout=args.timeout,
            )
        )
        for u in uuids
    ]

    # Wait for all to finish
    await asyncio.gather(*tasks, return_exceptions=False)

    print("[NATS] Closing connection ...")
    await nc.drain()
    await nc.close()
    print("[NATS] Closed.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dummy sensor data generator + NATS replicator"
    )

    parser.add_argument(
        "--server",
        default="nats://chocolatefrog@cssmartfall1.cose.txstate.edu:4224",
        help="NATS server URL (default: %(default)s)",
    )

    # You can pass multiple --uuid arguments if want explicit uuids
    parser.add_argument(
        "--uuid",
        action="append",
        help="User UUID to simulate (can be passed multiple times)",
    )

    parser.add_argument(
        "--num-random-users",
        type=int,
        default=0,
        help="Number of random UUID users to simulate (default: 0)",
    )

    parser.add_argument(
        "--messages-per-user",
        type=int,
        default=20,
        help="How many messages to send per user (default: 20)",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Interval in seconds between messages per user (default: 0.5)",
    )

    parser.add_argument(
        "--sequence-len",
        type=int,
        default=128, # 128 for tflite; 128 for pth; 
        help="Number of timesteps (T) in the dummy sequence (default: 128)",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="NATS request timeout in seconds (default: 2.0)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user.")


if __name__ == "__main__":
    main()
