"""Meme presets: each one is a motion + sound recipe that becomes a long single-paragraph LTX-2.5 prompt.

LTX-2.5 was trained on long captions describing the shot, the motion, the light and the sound, so every template
says what moves, how the camera behaves and what is heard. `{subject}` is the user's description of who is in the
photo (or a neutral fallback); `{line}` is the text for presets that take one.
"""
from __future__ import annotations

FALLBACK_SUBJECT = "the main subject of the photo"
KEEP_LOOK = ("The framing, lighting, colours and identity of {subject} stay exactly as in the original photo, "
             "with natural, realistic motion.")
# Group photos: one person performs the meme, everyone else stays put. Big motion on every face is what makes the
# model invent new faces; small motion on the others keeps them recognisable.
GROUP_SUBJECT = "the person in the centre of the photo"
GROUP_KEEP = ("Everyone else in the photo stays in their place with only small, natural movements, smiling and "
              "glancing at {subject}. Every person keeps exactly their own face, hair, body and clothes from the "
              "original photo, nobody new appears and nobody leaves the frame. The camera stays steady with no zoom "
              "or cut, and the lighting and background stay exactly as in the photo.")

PRESETS: list[dict] = [
    {"id": "dramatic_zoom", "name": "Dramatic Zoom", "emoji": "🎬", "tag": "dun dun DUNNN",
     "colors": ["#ff3d6e", "#7a1fff"],
     "prompt": "The camera suddenly crash-zooms in on the face of {subject}, who slowly turns to stare straight into "
               "the lens with an intense, deadly serious expression. The light dims and a cold spotlight falls on the "
               "face. A loud, cheesy orchestral sting plays, three heavy brass hits, dun, dun, dunnn, followed by a "
               "low rumble."},
    {"id": "vibing", "name": "Vibing", "emoji": "🎧", "tag": "lo-fi head bob",
     "colors": ["#00c2ff", "#0047ff"],
     "prompt": "{subject} closes their eyes and bobs their head rhythmically to the beat with a small satisfied "
               "smile, shoulders swaying side to side, completely in the zone. The camera stays still with a gentle "
               "handheld sway. A chill lo-fi hip hop beat with a mellow bass line and soft vinyl crackle plays "
               "throughout."},
    {"id": "sitcom_laugh", "name": "Sitcom Laugh", "emoji": "😂", "tag": "studio audience",
     "colors": ["#ffd000", "#ff7a00"],
     "prompt": "{subject} bursts into uncontrollable laughter, head thrown back, shoulders shaking, wiping a tear "
               "from one eye, then trying and failing to regain composure. The camera holds a steady medium shot. A "
               "loud studio audience laugh track roars and claps, layered over the real laughter."},
    {"id": "side_eye", "name": "Side Eye", "emoji": "👀", "tag": "suspicious glance",
     "colors": ["#a6ff00", "#00a86b"],
     "prompt": "{subject} slowly shifts their eyes sideways toward the camera without moving their head, one eyebrow "
               "rising in deep suspicion, lips pressed tight. The camera pushes in very slowly. Tense plucked "
               "pizzicato strings tiptoe in the background, ending on a single comedic low note."},
    {"id": "sad_violin", "name": "Sad Violin", "emoji": "🎻", "tag": "single tear",
     "colors": ["#6a8dff", "#2b2d6e"],
     "prompt": "The lower lip of {subject} trembles and a single tear rolls slowly down their cheek as they look "
               "down, the colours becoming slightly cooler and greyer. The camera slowly pushes in. A melancholic "
               "solo violin plays a slow, overly dramatic melody, with soft rain pattering against a window."},
    {"id": "explosion", "name": "Explosion Behind", "emoji": "💥", "tag": "cool guys don't look",
     "colors": ["#ff8a00", "#e52e04"],
     "prompt": "A huge fiery explosion erupts in the background behind {subject}, throwing orange light, sparks and "
               "smoke across the scene, while {subject} stays completely calm, unbothered, not looking back. Hair "
               "and clothes ripple from the shockwave. A deep thunderous boom, crackling fire and a heroic electric "
               "guitar riff."},
    {"id": "say_it", "name": "Make Them Say It", "emoji": "🗣️", "tag": "your line, lip-synced",
     "colors": ["#ff4fd8", "#8b00ff"], "needs_line": True, "line_label": "What do they say?",
     "line_placeholder": "I'm not arguing, I'm just explaining why I'm right",
     "prompt": "{subject} looks straight into the camera and clearly says, \"{line}\", with expressive, natural lip "
               "movements perfectly matching the words, eyebrows and head moving with the speech. The camera holds a "
               "steady close-up. The voice is clear and close to the microphone, with a quiet room tone behind it."},
    {"id": "victory_dance", "name": "Victory Dance", "emoji": "🕺", "tag": "goofy celebration",
     "colors": ["#00ffa3", "#00a3ff"],
     "prompt": "{subject} breaks into an energetic, goofy victory dance, arms pumping, shoulders shimmying, grinning "
               "from ear to ear. The camera follows with a playful handheld bounce. An upbeat funky track with a "
               "slapping bass line and a crowd cheering and whistling."},
    {"id": "mind_blown", "name": "Mind Blown", "emoji": "🤯", "tag": "jaw on the floor",
     "colors": ["#ff2fa0", "#ffb800"],
     "prompt": "The eyes of {subject} go wide and their jaw drops as both hands rise slowly to their head in total "
               "disbelief. The camera does a fast, shaky zoom in on the face. A whoosh, a big cartoon boom and a "
               "crowd going, ooooh."},
    {"id": "slowmo_hero", "name": "Slow-mo Hero", "emoji": "🦸", "tag": "epic wind & choir",
     "colors": ["#ffcf3d", "#b8860b"],
     "prompt": "In dramatic slow motion, strong wind blows through the hair and clothes of {subject}, who gazes into "
               "the distance with heroic determination as golden light flares across the lens. The camera slowly "
               "tilts up from a slightly low angle. An epic choir and thundering war drums build to a crescendo."},
    {"id": "shook", "name": "Shook", "emoji": "😱", "tag": "record scratch",
     "colors": ["#ff5757", "#8c52ff"],
     "prompt": "{subject} suddenly gasps and flinches backward in shock, eyes wide and hands raised, then freezes "
               "completely still. The camera jolts with the movement. A loud vinyl record scratch, then total "
               "awkward silence with a single cricket chirping."},
    {"id": "smolder", "name": "Smolder", "emoji": "😏", "tag": "smooth sax",
     "colors": ["#c86dd7", "#3023ae"],
     "prompt": "{subject} slowly raises one eyebrow and breaks into a confident, smouldering smirk, giving a tiny "
               "knowing nod at the camera. Warm, soft light glows on the face as the camera gently pushes in. A "
               "smooth, sultry saxophone riff plays over a soft jazz groove."},
    {"id": "custom", "name": "Custom", "emoji": "✍️", "tag": "write your own",
     "colors": ["#5b5b6b", "#1e1e28"], "custom": True,
     "prompt": "{custom}"},
]
BY_ID = {p["id"]: p for p in PRESETS}


class PresetError(ValueError):
    pass


def public() -> list[dict]:
    """What the browser needs to draw the preset grid (no prompt text)."""
    keys = ("id", "name", "emoji", "tag", "colors", "needs_line", "line_label", "line_placeholder", "custom")
    return [{k: p[k] for k in keys if k in p} for p in PRESETS]


def build_prompt(preset_id: str, subject: str = "", line: str = "", custom: str = "", group: bool = False) -> str:
    p = BY_ID.get(preset_id)
    if p is None:
        raise PresetError(f"unknown preset '{preset_id}'")
    subject = " ".join(subject.split())[:200] or (GROUP_SUBJECT if group else FALLBACK_SUBJECT)
    keep = (GROUP_KEEP if group else KEEP_LOOK).format(subject=subject)
    line = " ".join(line.replace('"', "'").split())[:300]
    custom = " ".join(custom.split())[:3000]
    if p.get("needs_line") and not line:
        raise PresetError(f"'{p['name']}' needs the line they should say")
    if p.get("custom"):
        if len(custom) < 20:
            raise PresetError("describe the motion and the sound in at least a sentence")
        return f"{custom} {keep}"
    body = p["prompt"].format(subject=subject, line=line)
    body = body[0].upper() + body[1:]                    # "{subject}" may open the sentence
    return f"{body} {keep}"
