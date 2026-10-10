"""
MongoDB client for storing and retrieving full call transcripts and conversation feeds.
Supports MongoDB Atlas (mongodb+srv) with connection resilience and Postgres fallback.
"""

import datetime as dt
import re
import urllib.parse
from typing import Any, Optional

import config

try:
    from pymongo import MongoClient, DESCENDING
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False


class MongoTranscriptStore:
    def __init__(self, uri: Optional[str] = None, db_name: Optional[str] = None):
        self.uri = uri or getattr(config, "MONGODB_URI", "")
        self.db_name = db_name or getattr(config, "MONGODB_DB_NAME", "voice_agent")
        self._client: Optional[Any] = None
        self._connected = False
        self._collection = None

        if self.uri and PYMONGO_AVAILABLE:
            self._init_connection()

    def _sanitize_uri(self, raw_uri: str) -> str:
        """Fixes unescaped '@' characters in MongoDB Atlas password strings."""
        if not raw_uri:
            return ""
        # Match mongodb+srv://username:password@cluster...
        match = re.match(r"^(mongodb(?:\+srv)?://)([^:]+):(.+)@([^@]+)$", raw_uri)
        if match:
            prefix, user, password, host_part = match.groups()
            # If password contains unencoded '@', url-encode it
            if "@" in password:
                password = urllib.parse.quote_plus(password)
            return f"{prefix}{user}:{password}@{host_part}"
        return raw_uri

    def _init_connection(self):
        try:
            clean_uri = self._sanitize_uri(self.uri)
            self._client = MongoClient(
                clean_uri,
                serverSelectionTimeoutMS=2500,
                connectTimeoutMS=2500,
                socketTimeoutMS=2500,
            )
            # Test ping quickly
            self._client.admin.command("ping")
            self._connected = True
            db = self._client[self.db_name]
            self._collection = db["call_transcripts"]
            # Ensure indexes
            self._collection.create_index([("call_id", 1)], unique=True)
            self._collection.create_index([("created_at", DESCENDING)])
            print(f"🍃 [mongodb] Connected successfully to Atlas database '{self.db_name}'.")
        except Exception as e:
            self._connected = False
            self._collection = None
            print(f"🍃 [mongodb] Connection notice (will retry / fallback to Postgres): {e}")

    @property
    def is_connected(self) -> bool:
        if self._connected and self._client:
            return True
        # Try reconnecting lazily
        if self.uri and PYMONGO_AVAILABLE:
            self._init_connection()
            return self._connected
        return False

    def save_call_transcript(
        self,
        call_id: str,
        caller_number: str = "Web",
        turns: Optional[list[dict[str, Any]]] = None,
        language: str = "gu-IN",
        duration_seconds: float = 0.0,
        status: str = "completed",
    ) -> bool:
        """
        Stores or updates a complete call transcript document in MongoDB.
        """
        if not call_id:
            return False

        turns_list = turns or []
        # Build plain text transcript for quick reading
        lines = []
        for t in turns_list:
            role = "User" if t.get("role") == "user" else "Priya"
            text = (t.get("text") or "").strip()
            if text:
                lines.append(f"{role}: {text}")
        full_text = "\n".join(lines)

        doc = {
            "call_id": call_id,
            "caller_number": caller_number or "Web",
            "language": language or "gu-IN",
            "duration_seconds": round(duration_seconds, 1),
            "status": status,
            "turn_count": len(turns_list),
            "turns": turns_list,
            "full_transcript": full_text,
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }

        if self.is_connected and self._collection is not None:
            try:
                self._collection.update_one(
                    {"call_id": call_id},
                    {
                        "$set": doc,
                        "$setOnInsert": {
                            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                        },
                    },
                    upsert=True,
                )
                print(f"🍃 [mongodb] Saved transcript for call {call_id} ({len(turns_list)} turns).")
                return True
            except Exception as e:
                print(f"🍃 [mongodb] Error saving transcript for {call_id}: {e}")

        return False

    def get_recent_transcripts(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Fetches the recent N call transcripts from MongoDB.
        """
        if not self.is_connected or self._collection is None:
            return []

        try:
            cursor = self._collection.find(
                {},
                {"_id": 0}  # omit MongoDB ObjectId for clean JSON serialization
            ).sort("created_at", DESCENDING).limit(limit)

            return list(cursor)
        except Exception as e:
            print(f"🍃 [mongodb] Error fetching recent transcripts: {e}")
            return []

    def get_transcript_by_call_id(self, call_id: str) -> Optional[dict[str, Any]]:
        """
        Fetches a single call transcript document by call_id from MongoDB.
        """
        if not self.is_connected or self._collection is None or not call_id:
            return None
        try:
            return self._collection.find_one({"call_id": call_id}, {"_id": 0})
        except Exception as e:
            print(f"🍃 [mongodb] Error fetching transcript for {call_id}: {e}")
            return None


# Global singleton instance
_shared_mongo_store: Optional[MongoTranscriptStore] = None


def get_shared_mongo_store() -> MongoTranscriptStore:
    global _shared_mongo_store
    if _shared_mongo_store is None:
        _shared_mongo_store = MongoTranscriptStore()
    return _shared_mongo_store
