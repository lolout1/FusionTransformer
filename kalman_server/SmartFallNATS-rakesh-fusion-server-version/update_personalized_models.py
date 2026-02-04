#!/usr/bin/env python3
import os
from typing import List, Optional

from couchbase.cluster import Cluster
from couchbase.auth import PasswordAuthenticator
from couchbase.options import ClusterOptions, QueryOptions

# ----------------- Config -----------------
CB_CONN_STR = "couchbase://cssmartfall1.cose.txstate.edu"
CB_USER = "Administrator"
CB_PASS = "iJ118+8!/nMFl\Lh9j<n"
CB_BUCKET_DATA = "smart-fall-data"
CB_BUCKET_BLOBS = "smart-fall-blobs"

MODELS_ROOT = "/home/su-kgv34/personalized_models"
DEFAULT_UUID = "default-lstm-2d"   # extra default uuid
DEFAULT_DIR_NAME = "default"       # directory name for the default model
DEFAULT_MODEL_NAME = "default"       # name for the default model

# ----------------- Helper functions -----------------


def get_uuids_with_enough_samples(cluster: Cluster) -> List[str]:
    """
    Query CB_BUCKET_DATA to get all uuids with count(uuid) > 10.
    Then append DEFAULT_UUID if not already present.
    """

    query_str = f"""
    SELECT d.uuid
    FROM `{CB_BUCKET_DATA}` AS d
    GROUP BY d.uuid
    HAVING COUNT(1) > 10
    """
    rows = cluster.query(query_str)
    uuids = [row["uuid"] for row in rows if row.get("uuid")]

    if DEFAULT_UUID not in uuids:
        uuids.append(DEFAULT_UUID)

    return uuids


def get_best_model_doc_id(cluster: Cluster, uuid: str) -> Optional[str]:
    """
    Get META().id of the best model (isBest = TRUE, highest version)
    for the given uuid from CB_BUCKET_BLOBS.
    """

    query_str = f"""
    SELECT META(b).id AS doc_id
    FROM `{CB_BUCKET_BLOBS}` AS b
    WHERE b.uuid = $uuid AND b.isBest = TRUE
    ORDER BY b.version DESC
    LIMIT 1
    """
    rows = cluster.query(
        query_str,
        QueryOptions(named_parameters={"uuid": uuid}),
    )
    for row in rows:
        return row.get("doc_id")
    return None


def get_parts_for_doc(cluster: Cluster, doc_id: str) -> List[str]:
    """
    Get parts[] for a given document id from CB_BUCKET_BLOBS.
    """

    query_str = f"""
    SELECT b.parts
    FROM `{CB_BUCKET_BLOBS}` AS b
    USE KEYS $doc_id
    LIMIT 1
    """
    rows = cluster.query(
        query_str,
        QueryOptions(named_parameters={"doc_id": doc_id}),
    )
    for row in rows:
        parts = row.get("parts") or []
        return parts
    return []

def blob_string_to_bytes(blob_str: str) -> bytes:
    """
    Convert a comma-separated string of integers ('12,34,255,...')
    back into a bytes object.
    """

    s = blob_str.strip()
    if not s:
        return b""

    # In case model stores like "[12,34,...]"
    if s[0] == '[' and s[-1] == ']':
        s = s[1:-1].strip()

    parts = s.split(',')
    try:
        return bytes(int(p) for p in parts if p.strip() != "")
    except ValueError as e:
        # Debug help: show a small prefix of the string
        print(f"[ERROR] Failed to parse blob string to bytes: {e}")
        print(f"        Sample blob content: {s[:200]}...")
        raise


def get_blob_for_pid(cluster: Cluster, pid: str) -> Optional[bytes]:
    """
    Get blob (stored as comma-separated byte values) for the given pid
    from CB_BUCKET_BLOBS and convert it back into bytes.
    """

    query_str = f"""
    SELECT b.blob
    FROM `{CB_BUCKET_BLOBS}` AS b
    USE KEYS $pid
    LIMIT 1
    """
    rows = cluster.query(
        query_str,
        QueryOptions(named_parameters={"pid": pid}),
    )

    for row in rows:
        blob = row.get("blob")
        if blob is None:
            return None

        # If it's already bytes
        if isinstance(blob, (bytes, bytearray)):
            return bytes(blob)

        # In case it's a comma-separated string of ints
        if isinstance(blob, str):
            try:
                return blob_string_to_bytes(blob)
            except Exception:
                return None

        print(f"[WARN] Unexpected blob type for pid={pid}: {type(blob)}")
        return None

    return None

def save_model_blob(uuid: str, blob: bytes) -> None:
    """
    Save blob to the appropriate directory under MODELS_ROOT.
    - For default uuid -> models/default/default-lstm-2d.tflite
    - For others      -> models/<uuid>/<uuid>.tflite
    Overwrites if the file already exists.
    """

    # Directory name logic
    dir_name = DEFAULT_DIR_NAME if uuid == DEFAULT_UUID else uuid
    uuid = DEFAULT_MODEL_NAME if uuid == DEFAULT_UUID else uuid
    out_dir = os.path.join(MODELS_ROOT, dir_name)
    os.makedirs(out_dir, exist_ok=True)

    # File name logic
    filename = f"{uuid}.tflite"
    out_path = os.path.join(out_dir, filename)

    with open(out_path, "wb") as f:
        f.write(blob)

    print(f"Saved model for uuid={uuid} to {out_path}")


# ----------------- Main script -----------------


def main():
    # Connect to Couchbase
    auth = PasswordAuthenticator(CB_USER, CB_PASS)
    cluster = Cluster(CB_CONN_STR, ClusterOptions(auth))

    # Ensure buckets are opened
    cluster.bucket(CB_BUCKET_DATA)
    cluster.bucket(CB_BUCKET_BLOBS)

    try:
        uuids = get_uuids_with_enough_samples(cluster)
        print(f"Found {len(uuids)} uuids (including default): {uuids}")

        for uuid in uuids:
            print(f"\nProcessing uuid: {uuid}")

            # 1) Get best model doc_id
            doc_id = get_best_model_doc_id(cluster, uuid)
            if not doc_id:
                print(f"  [WARN] No best model doc found for uuid={uuid}")
                continue
            print(f"  Best model doc_id: {doc_id}")

            # 2) Get parts[]
            parts = get_parts_for_doc(cluster, doc_id)
            if not parts:
                print(f"  [WARN] No parts[] found for doc_id={doc_id}")
                continue
            pid = parts[0]
            print(f"  Using pid={pid} (parts[0])")

            # 3) Get blob for pid
            blob = get_blob_for_pid(cluster, pid)
            if blob is None:
                print(f"  [WARN] No blob found for pid={pid}")
                continue

            # 4) Save blob to filesystem
            save_model_blob(uuid, blob)

    finally:
        cluster.close()


if __name__ == "__main__":
    main()
