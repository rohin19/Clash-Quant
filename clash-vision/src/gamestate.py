"""gamestate.py
Core opponent state tracking logic. Only this file is modified to avoid merge conflicts.

External model integration contract:
Call GameState.ingest_detections(predictions, frame_ts=None) each frame where
  predictions: list[dict] with keys (case-insensitive accepted for flexibility):
    - card OR class_name: str (normalized card name Title Case)
    - bbox OR bbox_xyxy: [x1, y1, x2, y2] in pixel coordinates
    - confidence: float (0..1)
    - (optional) frame: int | None (frame index)

Primary responsibilities:
 1. Maintain a rolling set of currently visible cards & last-seen times.
 2. Infer "play events" when a card newly appears (debounced so persistence
    across consecutive frames does not repeatedly count as a play).
 3. Simulate opponent elixir regeneration & spending when a card play is inferred.
 4. Reconstruct / maintain deck ordering & current hand (first 4 indices) in a
    best-effort way as cards are discovered.

Important assumptions / simplifications (MVP):
 - A card play is inferred when a detection for a card appears after it was not
   visible for at least PLAY_COOLDOWN_SEC seconds (prevents multi-frame duplicates).
 - If card unknown in deck we append it (deck discovery phase).
 - Hand cycling: when a card in the first 4 (hand) is played, it is moved to the
   back after queue shift emulating Clash Royale cycle (simplified when unknowns exist).
 - We do not attempt to handle mirrored / duplicate card variants.

Extend / integrate: The model provider can simply format outputs to the contract
above; no other code change needed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
import time
import math

############################
# Configuration Constants  #
############################

PLAY_COOLDOWN_SEC = 1.0          # Minimum time a card must be absent before a new detection counts as a play
VISIBILITY_TTL_SEC = 2.0         # Remove a card from visible set if not re-seen for this long
MIN_CONFIDENCE = 0.30            # Ignore detections below this confidence
ELIXIR_INCREMENT_INTERVAL = 0.28 # Normal elixir tick (adds 0.1)
ELIXIR_DOUBLE_TIME = 120.0       # Seconds after start when double elixir begins
MAX_ELIXIR = 10.0
FRAME_TIME_FALLBACK = 1/15       # Used if frame timestamp not provided to approximate timing

# Basic elixir cost map (extend as needed)
CARD_ELIXIR_COST: Dict[str, int] = {
    "Giant": 5, "Knight": 3, "Bomber": 3, 
    "Hog Rider": 4, "Dart Goblin": 3, "Mini Pekka": 4,
    "Baby Dragon": 4, "Valkyrie": 4
}


@dataclass
class CardDetection:
    card_name: str
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    frame: Optional[int] = None
    first_seen: float = 0.0
    last_seen: float = 0.0


class GameState:
    def __init__(self):
        # Visibility tracking
        self.visible_cards: Dict[str, CardDetection] = {}
        # Deck & play tracking
        self.deck: List[Optional[str]] = [None] * 8   # indices 0-3 hand, 4-7 queue
        self.play_history: List[str] = []
        self.last_played: Optional[str] = None
        # Elixir
        self.elixir_opponent: float = 0.0
        self.last_elixir_update: float = time.time()
        self.match_start_time: float = self.last_elixir_update
        # Predictions & helpers
        self.next_prediction: List[str] = []  # placeholder for future forecasting
        self._last_absence_times: Dict[str, float] = {}  # last time we stopped seeing a card
        self._last_play_time: Dict[str, float] = {}      # last time a card was considered played

    # ------------------------ Public API ------------------------ #
    def ingest_detections(self, detections: List[Dict[str, Any]], frame_ts: Optional[float] = None):
        """Main entry point called each frame with raw model detections.

        detections: list of dicts with keys (loose schema):
           card|class_name : str
           bbox|bbox_xyxy  : list[float] length 4 (x1,y1,x2,y2)
           confidence      : float
           frame           : (optional) int
        frame_ts: optional explicit timestamp (seconds). If None, uses time.time().
        """
        now = frame_ts if frame_ts is not None else time.time()
        self._update_elixir(now)

        # Normalize & filter
        norm = self._normalize_detections(detections)
        # Deduplicate per card keep highest confidence
        reduced = self._dedupe_keep_best(norm)
        # Update visibility map
        self._update_visibility(reduced, now)
        # Determine new play events
        plays = self._infer_plays(reduced, now)
        # Apply deck & elixir effects of plays
        for card in plays:
            self._apply_play(card, now)
        # Housekeeping remove stale visibility
        self._prune_visibility(now)

    def get_state(self) -> Dict[str, Any]:
        """Return current state summary (serializable)."""
        visible_export = [
            {
                "card": cd.card_name,
                "bbox": cd.bbox,
                "confidence": round(cd.confidence, 4),
                "last_seen": cd.last_seen,
            } for cd in self.visible_cards.values()
        ]
        return {
            "visible_cards": visible_export,
            "elixir_opponent": round(self.elixir_opponent, 2),
            "last_played": self.last_played,
            "play_history": self.play_history[-20:],  # recent window
            "deck": self.deck,
            "current_hand": self.deck[:4],
            "queue": self.deck[4:],
            "next_prediction": self.next_prediction,
            "match_time": round(time.time() - self.match_start_time, 2)
        }

    def reset(self):
        self.__init__()

    # --------------------- Internal Helpers --------------------- #
    def _normalize_detections(self, raw: List[Dict[str, Any]]) -> List[CardDetection]:
        out: List[CardDetection] = []
        for d in raw:
            if not isinstance(d, dict):
                continue
            name = d.get("card") or d.get("class_name") or d.get("name")
            if not name:
                continue
            bbox = d.get("bbox") or d.get("bbox_xyxy")
            if not bbox or len(bbox) != 4:
                continue
            conf = float(d.get("confidence", 0.0))
            if conf < MIN_CONFIDENCE:
                continue
            name_norm = self._normalize_card_name(name)
            out.append(CardDetection(
                card_name=name_norm,
                bbox=[float(x) for x in bbox],
                confidence=conf,
                frame=d.get("frame"),
                first_seen=0.0,
                last_seen=0.0,
            ))
        return out

    def _dedupe_keep_best(self, dets: List[CardDetection]) -> List[CardDetection]:
        best: Dict[str, CardDetection] = {}
        for d in dets:
            prev = best.get(d.card_name)
            if prev is None or d.confidence > prev.confidence:
                best[d.card_name] = d
        return list(best.values())

    def _update_visibility(self, dets: List[CardDetection], now: float):
        for d in dets:
            existing = self.visible_cards.get(d.card_name)
            if existing:
                existing.bbox = d.bbox
                existing.confidence = d.confidence
                existing.last_seen = now
            else:
                d.first_seen = now
                d.last_seen = now
                self.visible_cards[d.card_name] = d

    def _infer_plays(self, dets: List[CardDetection], now: float) -> List[str]:
        plays: List[str] = []
        current_names = {d.card_name for d in dets}
        # Mark absence times for cards that disappeared
        for name, cd in list(self.visible_cards.items()):
            if name not in current_names and (now - cd.last_seen) > 0.0:
                self._last_absence_times[name] = now
        # A play is a card that is in dets AND either never seen before or absent long enough
        for d in dets:
            last_play_t = self._last_play_time.get(d.card_name, 0.0)
            absence_t = self._last_absence_times.get(d.card_name, None)
            # If never played yet -> treat first appearance as play (deck discovery)
            if last_play_t == 0.0 and d.card_name not in self.play_history:
                plays.append(d.card_name)
                continue
            # If card was absent for cooldown -> consider a new play
            if absence_t is not None and (now - absence_t) >= PLAY_COOLDOWN_SEC and (now - last_play_t) >= PLAY_COOLDOWN_SEC:
                plays.append(d.card_name)
        return plays

    def _apply_play(self, card_name: str, now: float):
        cost = CARD_ELIXIR_COST.get(card_name)
        if cost is not None and self.elixir_opponent >= cost:
            self.elixir_opponent = max(0.0, self.elixir_opponent - cost)
        self.last_played = card_name
        self.play_history.append(card_name)
        self._last_play_time[card_name] = now
        self._last_absence_times.pop(card_name, None)
        self._integrate_into_deck(card_name)

    def _integrate_into_deck(self, card: str):
        # If card already placed in deck hand (0-3) treat as cycle
        if card in self.deck[:4]:
            hand_index = self.deck.index(card)
            played_card = self.deck[hand_index]
            # Promote first queue card (index 4) if exists
            if self.deck[4] is not None:
                self.deck[hand_index] = self.deck[4]
                for i in range(4, 7):
                    self.deck[i] = self.deck[i + 1]
                self.deck[7] = played_card
            else:
                # If queue unknown, move played to end & leave gap
                self.deck[hand_index] = None
                # shift existing queue entries (if any) forward
                for i in range(4, 7):
                    self.deck[i] = self.deck[i + 1]
                self.deck[7] = played_card
            self._compact_deck()
            return
        # If card is already somewhere else (queue) do nothing (duplicate play ignored)
        if card in self.deck:
            return
        # Insert new card at first empty slot; if none, rotate
        try:
            empty_idx = self.deck.index(None)
            self.deck[empty_idx] = card
        except ValueError:
            # All filled: emulate cycling: shift left and append
            for i in range(7):
                self.deck[i] = self.deck[i + 1]
            self.deck[7] = card
        self._compact_deck()

    def _compact_deck(self):
        # Keep order but ensure no None gaps in hand region if queue has cards
        hand = self.deck[:4]
        queue = self.deck[4:]
        # Pull from queue to fill None in hand
        for i in range(4):
            if hand[i] is None:
                # find first non-None in queue
                for j, qv in enumerate(queue):
                    if qv is not None:
                        hand[i] = qv
                        queue[j] = None
                        break
        self.deck = hand + queue

    def _prune_visibility(self, now: float):
        for name in list(self.visible_cards.keys()):
            cd = self.visible_cards[name]
            if (now - cd.last_seen) > VISIBILITY_TTL_SEC:
                self.visible_cards.pop(name, None)
                self._last_absence_times[name] = now

    def _update_elixir(self, now: float):
        elapsed = now - self.last_elixir_update
        if elapsed <= 0:
            return
        match_elapsed = now - self.match_start_time
        double_time = match_elapsed >= ELIXIR_DOUBLE_TIME
        interval = ELIXIR_INCREMENT_INTERVAL / (2 if double_time else 1)
        ticks = int(elapsed / interval)
        if ticks > 0:
            self.elixir_opponent = min(MAX_ELIXIR, self.elixir_opponent + 0.1 * ticks)
            self.last_elixir_update += ticks * interval

    def _normalize_card_name(self, raw: str) -> str:
        # Basic normalization (Title Case & strip)
        return raw.strip().title()


# -------------------------- Example (Manual Test) -------------------------- #
if __name__ == "__main__":  # simple manual sanity test
    gs = GameState()
    # Simulate first frame
    gs.ingest_detections([
        {"card": "Giant", "bbox": [10,10,50,90], "confidence": 0.9},
        {"card": "Goblin", "bbox": [60,15,90,85], "confidence": 0.82},
    ])
    print("After frame 1", gs.get_state())
    time.sleep(0.5)
    # Same cards again (should NOT double-count plays due to cooldown)
    gs.ingest_detections([
        {"card": "Giant", "bbox": [12,12,52,92], "confidence": 0.88},
    ])
    print("After frame 2", gs.get_state())
    # Simulate disappearance & reappearance
    time.sleep(1.2)
    gs.ingest_detections([
        {"card": "Giant", "bbox": [14,14,54,94], "confidence": 0.9},
    ])
    print("After frame 3 (giant replay)", gs.get_state())
