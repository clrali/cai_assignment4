"""A skeleton of an agent: replay recordings, time every stage and write the trace."""

import argparse
import json
import math
import os
import re
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from urllib.request import ProxyHandler, Request, build_opener

from voice_kit import ASR_FINAL, LLM_COMPLETE, LLM_FIRST_TOKEN, ReplaySession
from voice_kit import now_nanoseconds
from voice_kit.audio import read_wav

import onnxruntime as ort
from faster_whisper import WhisperModel
from piper import PiperVoice
from piper.config import PiperConfig

# Replace this with the sample rate of your TTS, in hertz.
TTS_SAMPLE_RATE_HZ = 22050

_runtime = None

ANSWER_PROMPT = """You are the spoken assistant for Hartwell Community Center.
Reply in one or two short plain-text sentences, without markdown or reasoning.
Use ONLY the supplied task result and knowledge base for facts and actions.
If the result contains 'say', repeat that message exactly. Never claim an action succeeded unless
its result confirms it. Repeat acted-on durations, labels, room, floor, day, time and name.
For proposal_only ask for a yes; for no_match explain the incompatible constraints and
suggest the closest alternative. Ask just one question for missing information.
Answer both parts of a lookup and admit when facts are absent. Zero remaining seconds
means a timer has expired. Repeat reminder times as supplied by the user.
"""

PLAN_PROMPT = """Interpret the latest request as one flat JSON action object.
Actions: timer_set, timer_change, timer_extend, timer_cancel, timer_status, timer_list,
reminder_set, reminder_change, reminder_cancel, reminder_list, booking, lookup, clarify.
A request to find or reserve a room is booking, even if phrased as a question. A short
reply to a booking question is booking. Questions about a room's floor, capacity or
equipment are lookup, as are venue hours, prices, services and policies.
Timer fields: label (timer name), duration_seconds (minutes multiplied by 60; hours by 3600).
Use timer_status for how much time remains, timer_list for listing, timer_extend to add time.
Use reminder_set for a request to be reminded of a task at a clock time, never timer_set.
Reminder fields: label, task, time. For follow-ups, use the pending question and state.
If no duration is stated for a new timer, omit duration_seconds; Python will ask for it.
If no clock time is stated for a new reminder, omit time; Python will ask for it.
Only extract values, not facts or actions. Do not calculate answers to venue questions.
For clarify use question.
Omit unknown fields. Never invent a duration, reminder time, or successful action.
Return JSON only. State is context, not a new command.
"""

BOOKING_PROMPT = """Extract only meeting requirements explicitly stated in the user's text.
Return JSON with these fields, using null for anything not stated:
people: integer headcount; day: lowercase weekday; time: HH:MM (convert AM/PM or noon);
period: morning/afternoon/evening/any only if explicitly stated;
features: list of required equipment or quietness, including TV screens and quiet rooms;
feature_mode: add for extra requirements, replace for replacement/waiving all, remove
for equipment no longer needed; name: the PERSON to book under.
Use the supplied canonical feature vocabulary for paraphrases. Preserve unsupported
features so the search can explain no match. An empty feature list means explicitly
no equipment; otherwise absent equipment is null. If AM/PM is unspecified use the
12-hour reading; do not guess. Do not infer a name, headcount, day, time, or equipment.
The pending question only identifies what a short fragment refers to. Do not answer it.
"""


def mentioned_rooms(kb, transcript):
    return [r["id"] for r in kb["rooms"] if any(
        re.search(r"\b" + re.escape(alias) + r"\b", transcript, re.I)
        for alias in (r["id"], r["name"]))]


def booking_intent(transcript, state):
    """High-confidence domain cues; unfamiliar wording falls back to the LLM.

    These rules use linguistic intent and the live KB, never recording names or
    whole-utterance matches. Slot values are extracted separately from current text.
    """
    text = transcript.lower().replace("’", "'")
    active = state.pending.get("action") == "booking"
    named = mentioned_rooms(state.kb, text)
    correction = re.search(r"\b(actually|instead|change|switch|make|what if|rather|same)\b", text)
    request = re.search(r"\b(need|want|like|book|reserve|find|get|looking for)\b", text)
    fresh = re.search(r"\b(?:a|an|another|new) (?:[\w-]+ ){0,4}(?:room|space|hall|studio)\b", text)
    if request and fresh and not correction:
        return "new"
    if re.search(r"\b(timers?|remind|reminders?)\b", text):
        return None
    if any(re.search(r"\b" + re.escape(alias.lower()) + r"\b", text)
           for item in [*state.kb.get("areas", []), *state.kb.get("services", [])]
           for alias in (item["id"].replace("_", " "), item["name"])):
        return None
    # A question about a room ("Does Cedar Room have a projector?") is a lookup, not a booking change, so let the LLM planner route it.
    referent = named or re.search(r"\b(?:that|this|the) room\b", text) or (active and len(text.split()) <= 5 and re.search(r"\bit\b", text))
    if referent and re.search(r"\b(floor|where|capacity|hold|holds|equipment|features|available|have|has)\b", text) and not request and not correction:
        return None
    if active:
        if state.pending.get("awaiting") == "features" and re.search(r"\b(equipment|features|none|nothing)\b", text):
            return "update"
        if (state.pending.get("awaiting") != "features" and re.fullmatch(r"(?:no(?: thanks| thank you)?|not yet)[.! ]*", text)) or (re.search(r"\b(no|not|don't|do not|wait)\b", text) and re.search(r"\b(book|reserve|go ahead)\b", text)):
            return "decline"
        if re.search(r"^(?:yes|yeah|yep|okay|ok|sure|go ahead)\b|\b(?:book|reserve) (?:it|that|the room)\b|\b(?:sounds good|is fine|that works)\b", text):
            return "confirm"
        feature = any(re.search(r"\b" + re.escape(f) + r"\b", text)
                      for room in state.kb["rooms"] for f in room["features"])
        if correction or re.search(r"\b(need|require|add|remove|drop)\b", text) or (feature and (request or len(text.split()) <= 5)) or named or len(text.split()) <= 3 or re.search(r"\b(people|attendees|participants|of us|under the name)\b", text):
            return "update"
    return None


def plan_schema(knowledge_base, transcript, booking=False):
    """Typed output schema; missing booking values are explicit nulls."""
    nullable = {"type": ["string", "null"]}
    if booking:
        properties = {
            "people": {"type": ["integer", "null"], "minimum": 1},
            "day": {"enum": [None, *knowledge_base["opening_hours"]]},
            "time": nullable, "period": {"enum": [None, "morning", "afternoon", "evening", "any"]},
            "features": {"type": ["array", "null"], "items": {"type": "string"}},
            "feature_mode": {"enum": [None, "add", "replace", "remove"]}, "name": nullable,
        }
    else:
        properties = {
            "action": {"enum": ["timer_set", "timer_change", "timer_extend", "timer_cancel", "timer_status",
                                "timer_list", "reminder_set", "reminder_change", "reminder_cancel",
                                "reminder_list", "booking", "lookup", "clarify"]},
            "label": nullable, "task": nullable, "time": nullable, "question": nullable,
            "duration_seconds": {"type": ["number", "null"]},
        }
    return {"type": "object", "properties": properties,
            "required": list(properties) if booking else ["action"], "additionalProperties": False}


def spoken_time(value):
    hour, minute = map(int, value.split(":"))
    if minute == 0 and hour in (0, 12):
        return "midnight" if hour == 0 else "noon"
    numbers = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
               "fourteen fifteen sixteen seventeen eighteen nineteen").split()
    def words(n):
        return numbers[n] if n < 20 else ("", "", "twenty", "thirty", "forty", "fifty")[n // 10] + (
            " " + numbers[n % 10] if n % 10 else "")
    clock = words(hour % 12 or 12)
    if minute:
        clock += " " + ("oh " if minute < 10 else "") + words(minute)
    return clock + (" in the morning" if hour < 12 else " in the afternoon" if hour < 17 else " in the evening")


def joined(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1] if items else ""


def floor_name(floor):
    return "the " + {1: "first", 2: "second", 3: "third"}[floor] + " floor" if floor in (1, 2, 3) else f"floor {floor}"


def normalize_feature(feature):
    feature = feature.strip().lower()
    aliases = {"quiet room": "quiet", "quiet space": "quiet", "white board": "whiteboard",
               "tv": "tv screen", "television": "tv screen", "television screen": "tv screen",
               "video conferencing kit": "video call kit", "video conferencing": "video call kit"}
    return aliases.get(feature, feature)


def feature_phrase(features):
    return joined("a quiet setting" if f == "quiet" else "a " + f for f in features)


def run_session(recordings_directory: Path, output_directory: Path) -> None:
    """Replay every recording as one turn and write the trace."""
    global TTS_SAMPLE_RATE_HZ
    recordings = sorted(recordings_directory.glob("*.wav"))
    if (output_directory / "trace.jsonl").exists():
        raise FileExistsError("Use a fresh output directory so traces are not appended together.")
    _runtime.warm_up(read_wav(recordings[0]))
    TTS_SAMPLE_RATE_HZ = _runtime.sample_rate
    session = ReplaySession(
        sorted(recordings_directory.glob("*.wav")),
        output_directory,
        sink_sample_rate_hz=TTS_SAMPLE_RATE_HZ,
    )

    for turn in session.turns():
        received_audio = bytearray()

        try:
            # The shim feeds recording at real time.
            turn.start(on_chunk=received_audio.extend)

            # Stands in for releasing push-to-talk key.
            turn.end_of_speech_reached.wait()

            transcript = _runtime.transcribe(bytes(received_audio))

            # The moment the final transcript is available.
            turn.stamp(ASR_FINAL)

            turn.extras["transcript"] = transcript
            if not transcript:
                raise ValueError("Empty transcription")
            _runtime.turn = turn
            _runtime.transcript = transcript

            # Call your LLM here: stamp LLM_FIRST_TOKEN when the first token of the answer arrives,
            # and LLM_COMPLETE when the answer is finished.
            # Hand each sentence to your TTS as soon as it is complete, and pass every chunk it
            # produces to turn.sink.write(chunk).
            # Here your_asr and reply_text stand for your own code.

            turn.extras["reply"] = _runtime.respond(_runtime.transcript, _runtime.turn)

        except Exception as exc:
            turn.extras["error"] = f"{type(exc).__name__}: {exc}"
            print(f"{turn.turn_id}: {turn.extras['error']}", file=sys.stderr)
        finally:
            _runtime.turn = None
            # Waits for replay to end and writes the record.
            turn.finish()


class TaskState:
    """Keep deadlines and booking decisions in Python, outside the LLM conversation."""

    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.timers, self.reminders, self.pending = {}, {}, {}
        self.last_timer = self.last_reminder = None
        self.booking, self.bookings = {}, []
        self.proposal = None
        self.booking_confirmed = False
        self.alternatives = []

    def snapshot(self):
        now = now_nanoseconds()
        return {"timers": {k: {"remaining_seconds": max(0, math.ceil((v - now) / 1e9))}
                           for k, v in self.timers.items()},
                "reminders": self.reminders, "last_timer": self.last_timer,
                "last_reminder": self.last_reminder, "pending_request": self.pending,
                "booking_constraints": self.booking, "proposal": self.proposal,
                "booking_confirmed": self.booking_confirmed, "alternatives": self.alternatives,
                "bookings": self.bookings}

    def apply(self, plan):
        if not isinstance(plan, dict) or not isinstance(plan.get("action"), str):
            raise ValueError("I could not understand that request. Please rephrase it.")
        for field in ("label", "task", "room", "name"):
            if plan.get(field) is not None and (not isinstance(plan[field], str) or not plan[field].strip()):
                raise ValueError("Please give a nonempty name or label.")
        action = plan["action"]
        if self.pending.get("action") == action:
            plan = {**self.pending, **{k: v for k, v in plan.items() if v is not None}}
        if action.startswith("timer_"):
            return self.timer(action, plan)
        if action.startswith("reminder_"):
            return self.reminder(action, plan)
        if action == "booking":
            return self.book(plan)
        if action == "lookup":
            # The room under discussion, so "what floor is it on?" can be answered.
            selected_room = (self.proposal or (self.alternatives[0] if self.alternatives else None)
                             or (self.bookings[-1] if self.bookings else None))
            return {"knowledge_base": self.kb, "selected_room": selected_room,
                    "rule": "Use only these facts; admit absent answers."}
        if action == "clarify" and plan.get("question"):
            return {"say": plan["question"]}
        return {"say": "Please tell me the timer, reminder, venue question, or room booking you need."}

    def missing(self, plan, question):
        self.pending = plan.copy()
        return {"say": question}

    @staticmethod
    def target(items, label, last):
        if label:
            if label not in items:
                raise ValueError(f"There is no item named {label}. Which one do you mean?")
            return label
        if last in items:
            return last
        if len(items) == 1:
            return next(iter(items))
        raise ValueError("Which one do you mean?" if items else "There are none set.")

    def timer(self, action, plan):
        now, label = now_nanoseconds(), plan.get("label")
        if action == "timer_list":
            return {"timers": self.snapshot()["timers"]}
        if action != "timer_set":
            label = self.target(self.timers, label, self.last_timer)
        if action == "timer_cancel":
            del self.timers[label]
            self.pending = {}
            return {"say": f"Canceled the {label} timer."}
        if action == "timer_status":
            return {"label": label, "remaining_seconds": max(0, math.ceil((self.timers[label] - now) / 1e9))}
        if action not in ("timer_set", "timer_change", "timer_extend"):
            raise ValueError("Please ask to set, change, extend, cancel, check, or list timers.")

        duration = plan.get("duration_seconds")
        if duration is None:
            return self.missing(plan, "How long should the timer run?" if action == "timer_set"
                                else "What duration should I use?")
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Please give a positive duration in seconds or minutes.")
        if not label:
            number = 1
            while f"timer {number}" in self.timers:
                number += 1
            label = f"timer {number}"
        if action == "timer_set" and label in self.timers:
            return {"say": f"A timer named {label} already exists. Should I change it?"}

        start = max(now, self.timers[label]) if action == "timer_extend" else now
        self.timers[label] = start + round(duration * 1e9)
        self.last_timer, self.pending = label, {}
        return {"action": action, "label": label, "duration_seconds": duration,
                "remaining_seconds": math.ceil((self.timers[label] - now) / 1e9)}

    def reminder(self, action, plan):
        if action == "reminder_list":
            return {"reminders": self.reminders}
        label = plan.get("label")
        if action in ("reminder_cancel", "reminder_change"):
            label = self.target(self.reminders, label, self.last_reminder)
        if action == "reminder_cancel":
            del self.reminders[label]
            self.pending = {}
            return {"say": f"Canceled the reminder to {label}."}
        if action not in ("reminder_set", "reminder_change"):
            raise ValueError("Please ask to set, change, cancel, or list reminders.")
        old = self.reminders.get(label, {})
        task, when = plan.get("task") or old.get("task"), plan.get("time") or old.get("time")
        if not task:
            return self.missing(plan, "What should I remind you to do?")
        if not when:
            return self.missing(plan, "At what time should I remind you?")
        label = label or task
        self.reminders[label] = {"task": task, "time": when}
        self.last_reminder, self.pending = label, {}
        return {"action": action, **self.reminders[label]}

    def free_slots(self, room):
        return {day: [time for time in times if not any(
            b["room_id"] == room["id"] and b["day"] == day and b["time"] == time
            for b in self.bookings)] for day, times in room["free_hour_starts"].items()}

    def no_booking_match(self, b, alternatives):
        # Prefer a room that fits the group when two alternatives each break one constraint.
        alternatives.sort(key=lambda r: (len(r["unmet_constraints"]), r["capacity"] < b["people"],
                                         len(r["missing_features"]), abs(r["capacity"] - b["people"])))
        self.alternatives = alternatives[:2]
        requirements = f"for {b['people']} people" + (
            " with " + feature_phrase(b["features"]) if b["features"] else "")
        when = ""
        if b.get("day"):
            when = " on " + b["day"].capitalize()
        if b.get("time"):
            when += " at " + spoken_time(b["time"])
        elif b.get("period") and b["period"] != "any":
            when += " in the " + b["period"]
        first = f"No room {requirements} is available{when}."
        if b.get("room"):
            selected = next(r for r in self.kb["rooms"] if r["id"] == b["room"])
            first = f"{selected['name']} cannot meet your requirements {requirements}{when}."
        if not alternatives:
            say = first + " Which requirement would you like to change?"
        else:
            alt = alternatives[0]
            desc = f"{alt['room']} holds {alt['capacity']}"
            if alt["capacity"] < b["people"]:
                desc += " and has " + feature_phrase(alt["features"])
                question = "would a smaller group work?"
                if alt["missing_features"]:
                    desc += " but lacks " + joined(alt["missing_features"])
                    question = "would a smaller group without " + joined(alt["missing_features"]) + " work?"
            elif alt["missing_features"]:
                desc += " but lacks " + joined(alt["missing_features"])
                question = "would that work without " + joined(alt["missing_features"]) + "?"
            else:
                if b["features"]:
                    desc += " and has " + feature_phrase(b["features"])
                free = alt["free_hour_starts"]
                if b.get("day") and free.get(b["day"]):
                    times = free[b["day"]]
                    slot = min(times, key=lambda t: abs(int(t[:2]) * 60 + int(t[3:]) -
                               (int(b["time"][:2]) * 60 + int(b["time"][3:])))) if b.get("time") else times[0]
                    alt["suggested_day"], alt["suggested_time"] = b["day"], slot
                    desc += f", but its free hour on {b['day'].capitalize()} starts at {spoken_time(slot)}"
                    question = "would that time work?"
                else:
                    days = [d.capitalize() for d in self.kb["opening_hours"] if free.get(d)]
                    desc += ", but " + ("is free on " + joined(days) if days else "has no free hours")
                    question = "would another day work?" if days else "would another room work?"
            say = first + " " + desc + "; " + question
        return {"status": "no_match", "constraints": b.copy(), "closest_alternatives": self.alternatives,
                "say": say}

    def book(self, plan):
        constraints = ("day", "time", "time_ambiguous", "period", "people", "features", "room")
        updates = {k: plan[k] for k in (*constraints, "name") if plan.get(k) is not None}
        if "people" in updates and (type(updates["people"]) is not int or updates["people"] <= 0):
            raise ValueError("How many people need the room?")
        for field in ("features", "add_features", "remove_features"):
            if field in plan and (not isinstance(plan[field], list) or
                                  not all(isinstance(f, str) and f.strip() for f in plan[field])):
                raise ValueError("Which equipment or feature do you need?")
        for field in ("confirmed", "new_booking", "accept_alternative", "time_ambiguous"):
            if field in plan and type(plan[field]) is not bool:
                raise ValueError("Please clarify whether you want to book or change the room.")
        if "day" in updates:
            if not isinstance(updates["day"], str) or updates["day"].lower().strip() not in self.kb["opening_hours"]:
                raise ValueError("Which day of the week do you need?")
            updates["day"] = updates["day"].lower().strip()
        if "time" in updates:
            value = updates["time"]
            if not isinstance(value, str) or not re.fullmatch(r"(?:[01]?\d|2[0-3]):[0-5]\d", value):
                raise ValueError("At what time do you need the room, including morning or afternoon?")
            updates["time"] = value.zfill(5)
            updates["time_ambiguous"] = plan.get("time_ambiguous", False)
        periods = {"morning": (0, 12), "afternoon": (12, 17), "evening": (17, 24), "any": (0, 24)}
        if "period" in updates and (not isinstance(updates["period"], str) or updates["period"] not in periods):
            raise ValueError("Do you prefer morning, afternoon, or evening?")
        if "room" in updates:
            room = next((r for r in self.kb["rooms"] if updates["room"].strip().lower() in
                         (r["id"].lower(), r["name"].lower())), None)
            if room is None:
                raise ValueError("I do not have that room in the schedule. Which room do you mean?")
            updates["room"] = room["id"]

        previous = self.booking.copy()
        b = {} if plan.get("new_booking") else self.booking.copy()
        proposal = None if plan.get("new_booking") else self.proposal
        consent = False if plan.get("new_booking") else self.booking_confirmed
        if plan.get("accept_alternative") and not plan.get("new_booking") and self.alternatives:
            alt = next((a for a in self.alternatives if a["room_id"] == updates.get("room")), None)
            if alt is None and "room" not in updates:
                alt = self.alternatives[0]
            if alt:
                # Only relax the feature tradeoff actually disclosed in the last reply.
                if alt is self.alternatives[0]:
                    b["features"] = [f for f in b.get("features", []) if f not in alt["missing_features"]]
                    if alt.get("suggested_time") and not any(k in updates for k in ("day", "time", "period")):
                        b.update(day=alt["suggested_day"], time=alt["suggested_time"], time_ambiguous=False)
                b["room"] = alt["room_id"]
            proposal, consent = None, False
        if "period" in updates and "time" not in updates and not b.get("time_ambiguous"):
            b.pop("time", None)
            b.pop("time_ambiguous", None)
        if "time" in updates and "period" not in updates:
            b.pop("period", None)
        b.update(updates)
        if "features" in b:
            b["features"] = sorted({normalize_feature(f) for f in b["features"]})
        if "add_features" in plan or "remove_features" in plan:
            features = set(b.get("features", []))
            features.update(normalize_feature(f) for f in plan.get("add_features", []))
            features.difference_update(normalize_feature(f) for f in plan.get("remove_features", []))
            b["features"] = sorted(features)
        changed = any(previous.get(k) != b.get(k) for k in constraints if k != "room")
        if changed and "room" not in updates and not plan.get("accept_alternative"):
            b.pop("room", None)  # Search again rather than pinning an earlier selection.
        if previous.get("room") != b.get("room"):
            changed |= proposal is None or b.get("room") != proposal["room_id"]
        if changed:
            proposal, consent = None, False
        self.booking, self.proposal, self.booking_confirmed = b, proposal, consent
        self.pending = {"action": "booking"}
        self.alternatives = []
        for field, question in (("people", "How many people?"),
                                ("features", "Do you need any equipment or a quiet room?")):
            if field not in b:
                return self.missing({"action": "booking", "awaiting": field}, question)

        matches, alternatives = [], []
        for room in self.kb["rooms"]:
            unmet = []
            if b.get("room") and b["room"] != room["id"]:
                unmet.append("different room")
            if room["capacity"] < b["people"]:
                unmet.append(f"capacity is {room['capacity']}, need {b['people']}")
            missing = sorted(set(b["features"]) - set(room["features"]))
            if missing:
                unmet.append("missing " + ", ".join(missing))
            free = self.free_slots(room)
            slots = free.get(b.get("day"), [])
            low, high = periods.get(b.get("period", "any"), (0, 24))
            if b.get("time"):
                choices = [b["time"]]
                if b.get("time_ambiguous"):
                    hour, minute = map(int, b["time"].split(":"))
                    choices = [f"{hour % 12:02d}:{minute:02d}", f"{hour % 12 + 12:02d}:{minute:02d}"]
                slots = [t for t in slots if t in choices and low <= int(t[:2]) < high]
            else:
                slots = [t for t in slots if low <= int(t[:2]) < high]
            if b.get("day") and not slots:
                unmet.append("unavailable at the requested day/time")
            details = {"room_id": room["id"], "room": room["name"], "floor": room["floor"],
                       "capacity": room["capacity"], "features": room["features"]}
            if not unmet:
                for slot in slots or [None]:
                    matches.append({**details, "day": b.get("day"), "time": slot})
            else:
                alternatives.append({**details, "unmet_constraints": unmet,
                                     "missing_features": missing, "free_hour_starts": free})
        if b.get("time_ambiguous"):
            times = {m["time"] for m in matches if m["time"]}
            if len(times) == 1 and b.get("day"):
                b["time"], b["time_ambiguous"] = times.pop(), False
            else:
                return self.missing({"action": "booking", "awaiting": "time"},
                                    "Do you mean in the morning or afternoon?")
        if not matches:
            self.proposal, self.booking_confirmed = None, False
            return self.no_booking_match(b, alternatives)
        if not b.get("day"):
            return self.missing({"action": "booking", "awaiting": "day"}, "Which day do you need the room?")
        if not b.get("time") and not b.get("period"):
            return self.missing({"action": "booking", "awaiting": "time"}, "At what time, or which part of the day?")
        matches.sort(key=lambda m: (m["capacity"], m["time"], m["room_id"]))
        if self.proposal in matches and not changed:
            if plan.get("confirmed") is True:
                self.booking_confirmed = True
            elif plan.get("confirmed") is False:
                self.booking_confirmed = False
                return {"say": "I have not booked it. What would you like to change?"}
            if self.booking_confirmed:
                if not b.get("name"):
                    return self.missing({"action": "booking", "awaiting": "name"}, "What name should I book under?")
                booked = {**self.proposal, "name": b["name"]}
                self.bookings.append(booked)
                self.booking, self.proposal, self.booking_confirmed = {}, None, False
                self.pending = {}
                return {"status": "booked", **booked,
                        "say": f"Booked {booked['room']} on {floor_name(booked['floor'])}, "
                               f"{booked['day'].capitalize()} at {spoken_time(booked['time'])}, for {booked['name']}."}
        self.proposal = matches[0]
        p = self.proposal
        equipment = " and has " + feature_phrase(b["features"]) if b["features"] else ""
        return {"status": "proposal_only", **p,
                "say": f"{p['room']} on {floor_name(p['floor'])} is free on {p['day'].capitalize()} "
                       f"at {spoken_time(p['time'])}; it holds {p['capacity']}{equipment}. Shall I book it?"}


class LocalLLM:
    """Read streaming chat completions from the user's local vLLM server."""

    def __init__(self, settings):
        self.settings = settings
        self.url = settings.llm_base_url.rstrip("/") + "/chat/completions"
        self.opener = build_opener(ProxyHandler({}))

    def stream(self, messages, turn=None, json_mode=False, schema=None, exact_text=None):
        payload = {"model": self.settings.llm_model, "messages": messages, "stream": True,
                   "temperature": 0, "seed": self.settings.seed, "max_tokens": self.settings.max_tokens}
        if schema is not None:
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "action_plan", "schema": schema}}
        elif json_mode:
            payload["response_format"] = {"type": "json_object"}
        if exact_text is not None:
            # Still stream real spoken-answer tokens, preserving the timing contract.
            # The decoder must emit the grounded response, not reinterpret its question.
            payload["structured_outputs"] = {"choice": [exact_text]}
        request = Request(self.url, data=json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
        finished = False
        with self.opener.open(request, timeout=self.settings.request_timeout) as response:
            for line in response:
                line = line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if "error" in event:
                    raise RuntimeError(f"Local LLM error: {event['error']}")
                if not event.get("choices"):
                    continue
                choice = event["choices"][0]
                delta = choice.get("delta", {})
                if delta.get("reasoning_content") or delta.get("reasoning"):
                    if turn is not None:
                        turn.extras.setdefault("first_reasoning_token_nanoseconds", now_nanoseconds())
                if delta.get("content"):
                    yield delta["content"]
                reason = choice.get("finish_reason")
                if reason:
                    if reason != "stop":
                        raise RuntimeError(f"LLM ended with {reason!r}; increase max_tokens in main() if truncated.")
                    finished = True
                    break
        if not finished:
            raise RuntimeError("The LLM stream ended without a completion event.")


class VoiceAgent:
    """Model helpers used by the original your_asr/reply_text/run_session scaffold."""

    def __init__(self, args):
        hf_cache = Path("/storage/ice1/0/4/lheck6/CS6604_ECE6256/hf_cache")
        self.llm = LocalLLM(args)

        np.random.seed(args.seed)
        asr_cache = hf_cache / "hub/models--Systran--faster-whisper-small.en"
        tts_cache = hf_cache / "hub/models--rhasspy--piper-voices"
        asr_model = asr_cache / "snapshots" / (asr_cache / "refs/main").read_text().strip()
        tts_snapshot = tts_cache / "snapshots" / (tts_cache / "refs/main").read_text().strip()
        tts_model = tts_snapshot / "en/en_US/lessac/medium/en_US-lessac-medium.onnx"
        
        self.asr = WhisperModel(str(asr_model), device="cuda",
                                compute_type="float16", cpu_threads=args.cpu_count,
                                num_workers=1, local_files_only=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = args.cpu_count, 1
        self.tts = PiperVoice(
            session=ort.InferenceSession(str(tts_model), sess_options=options, providers=["CPUExecutionProvider"]),
            config=PiperConfig.from_dict(json.loads(Path(str(tts_model) + ".json").read_text())))
        self.sample_rate = self.tts.config.sample_rate
        self.state = TaskState(json.loads(args.knowledge_base.read_text(encoding="utf-8")))
        self.history = []

    def transcribe(self, audio_bytes):
        samples = np.frombuffer(audio_bytes, dtype="<i2").astype(np.float32) / 32768.0
        segments, _ = self.asr.transcribe(samples, language="en", beam_size=1, temperature=0,
                                          vad_filter=False, condition_on_previous_text=False)
        return " ".join(segment.text.strip() for segment in segments).strip()

    def audio_chunks(self, text):
        for chunk in self.tts.synthesize(text):
            yield chunk.audio_int16_bytes

    def warm_up(self, audio_bytes):
        self.transcribe(audio_bytes)
        list(self.llm.stream([{"role": "user", "content": 'Return JSON: {"action": "lookup"}'}], json_mode=True))
        list(self.llm.stream([{"role": "user", "content": "Say ready."}], exact_text="Ready."))
        list(self.llm.stream([{"role": "system", "content": PLAN_PROMPT},
                             {"role": "user", "content": "What are the opening hours?"}],
                            schema=plan_schema(self.state.kb, "What are the opening hours?")))
        list(self.llm.stream([{"role": "system", "content": BOOKING_PROMPT},
                             {"role": "user", "content": "Ready."}],
                            schema=plan_schema(self.state.kb, "Ready.", booking=True)))
        for _ in self.audio_chunks("Ready."):
            pass

    def interpret(self, transcript, turn):
        intent = booking_intent(transcript, self.state)
        if intent is None:
            messages = [{"role": "system", "content": PLAN_PROMPT + "\nState: " + json.dumps(self.state.snapshot())},
                        *self.history[-6:], {"role": "user", "content": transcript}]
            plan = json.loads("".join(self.llm.stream(messages, turn, json_mode=True,
                                                    schema=plan_schema(self.state.kb, transcript))))
            if plan.get("action") != "booking":
                return plan
            intent = "update" if self.state.booking else "new"
        # A booking turn: ask the LLM for slots, then keep only values said in this turn.
        vocabulary = sorted({f for r in self.state.kb["rooms"] for f in r["features"]})
        awaiting = self.state.pending.get("awaiting") if intent != "new" else None
        messages = [{"role": "system", "content": BOOKING_PROMPT + "\nFeature vocabulary: " + json.dumps(vocabulary) +
                     "\nPending field: " + str(awaiting)}, {"role": "user", "content": transcript}]
        slots = json.loads("".join(self.llm.stream(messages, turn, json_mode=True,
                                                 schema=plan_schema(self.state.kb, transcript, booking=True))))
        plan = {"action": "booking", "new_booking": intent == "new"}
        plan.update({k: v for k, v in slots.items() if v is not None and k != "feature_mode"})
        text = transcript.lower().replace("’", "'")
        # Reject default-filled slots: each value must have evidence in this turn.
        number_words = r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)"
        number = r"(?:\d+|" + number_words + r")(?:[ -]" + number_words + r")*"
        headcount = re.search(r"\b(" + number + r")\s+(?:people|persons?|attendees|participants|guests|delegates|members|of us)\b", text)
        if not headcount:
            headcount = re.search(r"\b(?:group|party|team) of (" + number + r")\b", text)
        if not headcount:
            headcount = re.search(r"\bfor (" + number + r")(?=\s+(?:on|with|at)\b|[.,?!]|$)", text)
        if not headcount and awaiting == "people":
            headcount = re.search(r"\b(" + number + r")\b", text)
        if not headcount:
            plan.pop("people", None)
        if plan.get("day") and plan["day"] not in text:
            plan.pop("day")
        if plan.get("period") and plan["period"] not in text:
            plan.pop("period")
        clock_stated = re.search(r"\b(?:at|from|around|by)\s+(?:about\s+)?" + number + r"|\b(noon|midnight)\b|\d(?::\d{2})?\s*[ap]\.?m\.?|\b(?:half|quarter) (?:past|to)\b", text)
        if not clock_stated and not (awaiting == "time" and re.search(number, text)):
            plan.pop("time", None)
        # Explicit canonical words are evidence even if the model missed one.
        literal_features = [f for f in vocabulary if re.search(r"\b" + re.escape(f) + r"\b", text)]
        features = plan.pop("features", None)
        if literal_features:
            features = sorted(set(features or []) | set(literal_features))
        waive = (re.search(r"\b(no (?:special )?equipment|none|nothing|no features|without equipment|(?:don't|do not) need (?:any |special )?equipment)\b", text) or
                 (awaiting == "features" and re.fullmatch(r"no(?: thanks| thank you)?[.! ]*", text)))
        if features:
            features = sorted({normalize_feature(f) for f in features})
        if features or waive:
            mode = slots.get("feature_mode") or "add"
            if re.search(r"\b(instead|replace|rather than|only need|only want)\b", text):
                mode = "replace"
            elif re.search(r"\b(remove|drop|without|don't need|do not need|no longer need)\b", text):
                mode = "remove"
            if waive:
                mode, features = "replace", []
            field = {"add": "add_features", "replace": "features", "remove": "remove_features"}[mode]
            plan[field] = features
        # A person's name must actually occur in the current utterance.
        name_context = (awaiting == "name" or re.search(r"\b(name|under|I'm|I am)\b", transcript, re.I) or
                        (intent == "confirm" and re.search(r"\bfor\b", text)))
        if plan.get("name") and (not name_context or plan["name"].casefold() not in transcript.casefold()):
            plan.pop("name")
        if plan.get("time"):
            explicit_period = re.search(r"\b(morning|afternoon|evening|tonight|noon|midnight)\b|\b[ap]\.?m\.?\b", text)
            hour, minute = map(int, plan["time"].split(":"))
            if re.search(r"\b(afternoon|evening|tonight)\b|\bp\.?m\.?\b", text) and hour < 12:
                hour += 12
            elif re.search(r"\b(morning)\b|\ba\.?m\.?\b", text) and hour == 12:
                hour = 0
            plan["time"] = f"{hour:02d}:{minute:02d}"
            explicit_24h = re.search(r"\b(?:at|from)\s+(?:1[3-9]|2[0-3])\b|\b(?:1[3-9]|2[0-3]):[0-5]\d\b", text)
            plan["time_ambiguous"] = not bool(explicit_period or explicit_24h)
        if intent == "confirm":
            if self.state.alternatives and not self.state.proposal:
                plan["accept_alternative"] = True
            else:
                plan["confirmed"] = True
        elif intent == "decline":
            plan["confirmed"] = False
        named = mentioned_rooms(self.state.kb, transcript)
        if len(named) == 1:
            plan["room"] = named[0]
        return plan

    def respond(self, transcript, turn):
        try:
            plan = self.interpret(transcript, turn)
            turn.extras["task_plan"] = plan
            result = self.state.apply(plan)
        except (ValueError, TypeError, KeyError) as exc:
            result = {"say": str(exc) if isinstance(exc, ValueError) and not isinstance(exc, json.JSONDecodeError)
                      else "I could not understand that request. Please rephrase it."}
        turn.extras["task_result"] = result

        # Second stream generates the *spoken* answer; timing it separately avoids
        # falsely labeling internal JSON tokens as the first token of the answer.
        if "say" in result:
            messages = [{"role": "system", "content": "Repeat the supplied text exactly. Do not answer its "
                         "questions or add any other words."}, {"role": "user", "content": result["say"]}]
        else:
            messages = [{"role": "system", "content": ANSWER_PROMPT},
                        {"role": "user", "content": json.dumps({"request": transcript, "result": result})}]
        def speak(sentence):
            for chunk in self.audio_chunks(sentence):
                turn.sink.write(chunk)

        pieces, buffer = [], ""
        # One worker keeps sentences in order while the main thread receives LLM tokens.
        # The context waits for all audio writes before run_session can finish the turn.
        with ThreadPoolExecutor(max_workers=1) as speaker:
            pending_audio = []
            for token in self.llm.stream(messages, turn, exact_text=result.get("say")):
                if not pieces and not token.strip():
                    continue
                if not pieces:
                    turn.stamp(LLM_FIRST_TOKEN)
                pieces.append(token)
                buffer += token
                turn.extras["reply"] = "".join(pieces).strip()
                # Feed first full sentence to Piper while the LLM continues.
                while True:
                    match = re.search(r'[.!?]["\u201d]?\s+', buffer)
                    if not match:
                        break
                    pending_audio.append(speaker.submit(speak, buffer[:match.end()].strip()))
                    buffer = buffer[match.end():]
            if not pieces:
                raise RuntimeError("LLM returned no spoken reply")
            turn.stamp(LLM_COMPLETE)
            if buffer.strip():
                pending_audio.append(speaker.submit(speak, buffer.strip()))
            for audio in pending_audio:
                audio.result()  # Propagate TTS errors so failed turns are recorded as failures.
        answer = "".join(pieces).strip()
        self.history.extend([{"role": "user", "content": transcript},
                             {"role": "assistant", "content": answer}])
        self.history = self.history[-6:]
        return answer


def main():
    global _runtime
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", ""))
    parser.add_argument("--llm-base-url", default=os.environ.get("LLM_BASE_URL", ""))
    parser.add_argument("--cpu-count", type=int, default=4)
    parser.add_argument("--recordings-directory", type=Path, default=Path("recordings"))
    parser.add_argument("--output-directory", type=Path, default=Path("run_output"))
    args = parser.parse_args()

    settings = SimpleNamespace(
        recordings_directory=args.recordings_directory,
        output_directory=args.output_directory,
        llm_model=args.llm_model.strip(),
        llm_base_url=args.llm_base_url.strip(),
        knowledge_base=Path(__file__).resolve().parent / "toy_kb/knowledge_base.json",
        asr_compute_type="float16",
        cpu_count=args.cpu_count,
        max_tokens=256,
        seed=42,
        request_timeout=120,
    )
    
    _runtime = VoiceAgent(settings)
    run_session(settings.recordings_directory, settings.output_directory)


if __name__ == "__main__":
    main()