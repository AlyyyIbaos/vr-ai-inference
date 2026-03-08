from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Dict, Optional, Tuple
import time
import threading


@dataclass
class _State:
    # vote buffer of "suspicious events" (0/1)
    votes: Deque[int] = field(default_factory=lambda: deque(maxlen=5))
    last_seen: float = field(default_factory=time.time)

    # latched alarm state
    active: bool = False

    # holding + cooldown
    hold_until: float = 0.0
    cooldown_until: float = 0.0

    # last values (debug)
    last_prob: float = 0.0
    last_risk: float = 0.0
    last_event: int = 0
    updates: int = 0


class StrongCATEngine:
    """
    Strong Context-Aware Thresholding for runtime proctoring.

    Core ideas:
    1) Convert probability (and optional risk) into a binary "event" using ON/OFF thresholds (hysteresis).
    2) Use K-of-N persistence voting on events.
    3) Latch active with hold_seconds to prevent flicker.
    4) Apply cooldown to avoid rapid re-triggering.
    5) TTL cleanup to prevent memory growth.
    Thread-safe.

    Use:
      out = cat.update(session_id, prob, risk=None, now=time.time())
      out["cat_active"] is your stable decision.
    """

    def __init__(
        self,
        n: int = 5,
        k_on: int = 3,
        k_off: int = 1,
        prob_on: float = 0.72,
        prob_off: float = 0.60,
        hold_seconds: float = 2.0,
        cooldown_seconds: float = 1.0,
        session_ttl_seconds: int = 60 * 30,
        clamp_prob: Tuple[float, float] = (0.01, 0.99),

        # Optional risk gating (0..1). If you don't use risk, keep risk_* None.
        risk_on: Optional[float] = None,
        risk_off: Optional[float] = None,
        use_prob_or_risk: bool = True,
    ):
        if n < 1:
            raise ValueError("n must be >= 1")
        if not (1 <= k_on <= n):
            raise ValueError("k_on must be in [1, n]")
        if not (0 <= k_off < k_on):
            raise ValueError("k_off must be in [0, k_on-1] for hysteresis")
        if prob_off > prob_on:
            raise ValueError("prob_off must be <= prob_on")

        self.n = int(n)
        self.k_on = int(k_on)
        self.k_off = int(k_off)

        lo, hi = clamp_prob
        self.prob_on = float(max(lo, min(hi, prob_on)))
        self.prob_off = float(max(lo, min(hi, prob_off)))

        self.hold_seconds = float(max(0.0, hold_seconds))
        self.cooldown_seconds = float(max(0.0, cooldown_seconds))
        self.session_ttl_seconds = int(max(60, session_ttl_seconds))

        self.risk_on = risk_on
        self.risk_off = risk_off
        self.use_prob_or_risk = bool(use_prob_or_risk)

        self._sessions: Dict[str, _State] = {}
        self._lock = threading.Lock()

    def cleanup(self, now: Optional[float] = None) -> int:
        if now is None:
            now = time.time()
        removed = 0
        with self._lock:
            dead = [sid for sid, st in self._sessions.items()
                    if (now - st.last_seen) > self.session_ttl_seconds]
            for sid in dead:
                del self._sessions[sid]
                removed += 1
        return removed

    def _get(self, session_id: str, now: float) -> _State:
        st = self._sessions.get(session_id)
        if st is None:
            st = _State(votes=deque(maxlen=self.n))
            self._sessions[session_id] = st
        st.last_seen = now
        return st

    def _event_from_prob_risk(self, st: _State, prob: float, risk: Optional[float]) -> Tuple[int, str]:
        """
        Converts (prob, risk) into a binary event with hysteresis:
        - If currently not active -> event=1 when prob>=prob_on (or risk>=risk_on)
        - If currently active     -> event=1 when prob>=prob_off (or risk>=risk_off)
        This reduces flicker around threshold.
        """
        # Determine which threshold to use based on current active state
        p_thr = self.prob_off if st.active else self.prob_on

        # If using risk gating:
        if risk is not None and self.risk_on is not None and self.risk_off is not None:
            r_thr = self.risk_off if st.active else self.risk_on

            p_hit = prob >= p_thr
            r_hit = risk >= r_thr

            if self.use_prob_or_risk:
                ev = 1 if (p_hit or r_hit) else 0
                reason = f"OR gate (p>={p_thr:.3f}={p_hit}, r>={r_thr:.3f}={r_hit})"
            else:
                ev = 1 if (p_hit and r_hit) else 0
                reason = f"AND gate (p>={p_thr:.3f}={p_hit}, r>={r_thr:.3f}={r_hit})"

            return ev, reason

        # Prob-only mode
        ev = 1 if prob >= p_thr else 0
        reason = f"prob_only (p>={p_thr:.3f}={ev==1})"
        return ev, reason

    def update(
        self,
        session_id: str,
        prob_cheat: float,
        risk: Optional[float] = None,
        now: Optional[float] = None,
    ) -> Dict[str, object]:
        if now is None:
            now = time.time()

        p = float(prob_cheat)
        r = float(risk) if risk is not None else None

        with self._lock:
            st = self._get(session_id, now)
            st.updates += 1
            st.last_prob = p
            st.last_risk = 0.0 if r is None else r

            # Apply cooldown: if in cooldown and not active, be stricter (optional behavior)
            in_cooldown = (now < st.cooldown_until) and (not st.active)

            # Determine event (0/1) with hysteresis
            event, event_reason = self._event_from_prob_risk(st, p, r)

            # If in cooldown, require event to be very strong by forcing event=0 unless prob is clearly high
            # (prevents spammy toggling if the signal is jittery)
            if in_cooldown:
                if p < self.prob_on + 0.05:  # small margin (tune if needed)
                    event = 0
                    event_reason += " + cooldown_strict"

            st.last_event = event
            st.votes.append(event)
            votes_sum = sum(st.votes)

            # HOLD: if active and still holding, stay active regardless of votes
            if st.active and self.hold_seconds > 0 and now < st.hold_until:
                cat_active = True
                transition = "hold_active"
            else:
                # Transition logic with persistence + hysteresis (k_on, k_off)
                if not st.active:
                    if votes_sum >= self.k_on:
                        st.active = True
                        cat_active = True
                        transition = "activate"
                        if self.hold_seconds > 0:
                            st.hold_until = now + self.hold_seconds
                    else:
                        cat_active = False
                        transition = "remain_inactive"
                else:
                    # turn off only when votes fall to <= k_off
                    if votes_sum <= self.k_off:
                        st.active = False
                        cat_active = False
                        transition = "deactivate"
                        st.hold_until = 0.0
                        if self.cooldown_seconds > 0:
                            st.cooldown_until = now + self.cooldown_seconds
                    else:
                        cat_active = True
                        transition = "remain_active"
                        if self.hold_seconds > 0:
                            # refresh hold window when staying active
                            st.hold_until = max(st.hold_until, now + self.hold_seconds)

            hold_remaining = max(0.0, st.hold_until - now) if cat_active else 0.0
            cooldown_remaining = max(0.0, st.cooldown_until - now) if (not cat_active) else 0.0

        # pred is the *raw* threshold decision at prob_on (not hysteresis)
        pred_raw = 1 if p >= self.prob_on else 0

        return {
            "pred_raw": int(pred_raw),
            "event": int(event),
            "event_reason": event_reason,
            "votes_sum": int(votes_sum),
            "n": self.n,
            "k_on": self.k_on,
            "k_off": self.k_off,
            "prob_on": self.prob_on,
            "prob_off": self.prob_off,
            "risk_on": self.risk_on,
            "risk_off": self.risk_off,
            "cat_active": int(cat_active),
            "transition": transition,
            "hold_remaining_s": float(hold_remaining),
            "cooldown_remaining_s": float(cooldown_remaining),
        }

