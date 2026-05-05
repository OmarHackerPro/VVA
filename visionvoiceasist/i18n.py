"""Azerbaijani language pack — message templates + COCO-class translations.

User-facing text is centralised here so it can be reviewed by a native speaker
and swapped for other languages later.
"""

from __future__ import annotations

# COCO class translations (corrections applied vs. previous codebase).
AZ_LABELS: dict[str, str] = {
    "person": "insan", "bicycle": "velosiped", "car": "maşın",
    "motorcycle": "motosiklet", "airplane": "təyyarə", "bus": "avtobus",
    "train": "qatar", "truck": "yük maşını", "boat": "qayıq",
    "traffic light": "svetofor", "fire hydrant": "yanğın kranı",
    "stop sign": "dayan nişanı", "parking meter": "parkomat",
    "bench": "skamya", "bird": "quş", "cat": "pişik", "dog": "it",
    "horse": "at", "sheep": "qoyun", "cow": "inək", "elephant": "fil",
    "bear": "ayı", "zebra": "zebra", "giraffe": "zürafə",
    "backpack": "çanta", "umbrella": "çətir", "handbag": "əl çantası",
    "tie": "qalstuk", "suitcase": "çamadan", "frisbee": "frizbi",
    "skis": "xizək", "snowboard": "snoubord", "sports ball": "top",
    "kite": "çərpələng", "baseball bat": "beysbol çubuğu",
    "baseball glove": "beysbol əlcəyi", "skateboard": "skeytbord",
    "surfboard": "sörf taxtası", "tennis racket": "tennis raketi",
    "bottle": "butulka", "wine glass": "şərab stəkanı",
    "cup": "fincan", "fork": "çəngəl", "knife": "bıçaq", "spoon": "qaşıq",
    "bowl": "kasa", "banana": "banan", "apple": "alma",
    "sandwich": "sendviç", "orange": "portağal", "broccoli": "brokoli",
    "carrot": "kök", "hot dog": "hot-dog", "pizza": "pizza",
    "donut": "donut", "cake": "tort", "chair": "stul", "couch": "divan",
    "potted plant": "çiçək", "bed": "yataq", "dining table": "masa",
    "toilet": "tualet", "tv": "televizor", "laptop": "noutbuk",
    "mouse": "siçan", "remote": "pult", "keyboard": "klaviatura",
    "cell phone": "telefon", "microwave": "mikrodalğalı soba",
    "oven": "soba", "toaster": "toster", "sink": "lavabo",
    "refrigerator": "soyuducu", "book": "kitab", "clock": "saat",
    "vase": "vaza", "scissors": "qayçı", "teddy bear": "oyuncaq ayı",
    "hair drier": "saç qurutma", "toothbrush": "diş fırçası",
}

SURFACES: frozenset[str] = frozenset({
    "dining table", "bed", "couch", "chair", "bench", "desk", "shelf",
})

EMERGENCY_LABELS: frozenset[str] = frozenset({
    "person", "car", "motorcycle", "truck", "bus", "bicycle",
})


# ── Message templates (callable so .format errors surface in tests) ─────────
def az_label(label_eng: str) -> str:
    """Return Azerbaijani translation, falling back to English."""
    return AZ_LABELS.get(label_eng, label_eng)


def position_label(cx: int, frame_w: int) -> str:
    """Return spatial position phrase based on horizontal centre.

    Properly diacriticised — a previous bug spelled these as
    'Saginizdə' / 'Qarsinizda'.
    """
    if cx < frame_w // 3:
        return "Solunuzda"
    if cx > 2 * frame_w // 3:
        return "Sağınızda"
    return "Qarşınızda"


def distance_label(area_pct: float) -> str:
    """Return distance phrase from object area percentage."""
    if area_pct > 0.45:
        return "çox yaxın (< 1 m)"
    if area_pct > 0.18:
        return "yaxın (1–2 m)"
    if area_pct > 0.06:
        return "orta (2–4 m)"
    return "uzaqda (4+ m)"


# ── Standard messages ─────────────────────────────────────────────────────
class Messages:
    """Static message strings and templates."""

    GREETING = "Salam! VisionVoiceAsist aktiv oldu."
    GREETING_BATTERY = "Batareya {pct} faizdir."
    AI_ONLINE = "Onlayn süni intellekt aktiv."
    AI_OFFLINE = "Offline rejim — yerli süni intellekt aktiv."
    AI_DISABLED = "Süni intellekt deaktivdir."

    SHUTDOWN = "VisionVoiceAsist bağlanır. Sağlam qalın!"
    OPEN_PATH = "Qarşınızda açıq yol var, maneə aşkarlanmadı."
    CLEAR_TEXT = "Yazı oxuyuram: {text}"
    APPROACH = "DİQQƏT! {label} sürətlə yaxınlaşır! Dayanın!"
    CRITICAL_PROXIMITY = "KRİTİK TƏHLÜKƏ! {label} çox yaxındır, dərhal dayanın!"
    FALLEN_PERSON = "DİQQƏT! {position} yıxılmış insan görünür!"

    PIT_STAIRS = "DİQQƏT! Qarşınızda pilləkən var! Dərhal dayanın!"
    PIT_OBSTACLE = "Xəbərdarlıq! Döşəmədə maneə var."
    PIT_THRESHOLD = "Diqqət! Döşəmə kəskin dəyişir — astana ola bilər."

    BATTERY_CRITICAL = "KRİTİK! Batareya yalnız {pct} faizdir! Dərhal şarj edin!"
    BATTERY_WARN = "Xəbərdarlıq: Batareya {pct} faizdir. Şarj etməyi unutmayın."

    CAMERA_LOST = "Kamera əlaqəsi kəsildi! Cihazı yoxlayın."
    NETWORK_LOST = "İnternet yoxdur, yerli süni intellekt işə düşdü."
    NETWORK_BACK = "İnternet bərpa olundu, onlayn rejim aktivdir."
    DEGRADED = "Sistem qismən işləyir — bəzi modullar əlçatmazdır."
    TTS_FAILED = "Səs sistemində xəta var."

    TRAFFIC_LIGHT_RED = "Svetofor QIRMIZIdır — DAYANIN."
    TRAFFIC_LIGHT_YELLOW = "Svetofor SARIdır — hazırlaşın."
    TRAFFIC_LIGHT_GREEN = "Svetofor YAŞILdır — keçə bilərsiniz."
    TRAFFIC_LIGHT_OFF = "Svetofor sönülüdür."

    SCENE_ON_SURFACE_ONE = "{surface} üzərində {item} var"
    SCENE_ON_SURFACE_FEW = "{surface} üzərində {items} və {last} var"
    SCENE_ON_SURFACE_MANY = "{surface} üzərində {n} əşya var"
    SCENE_CROWD = "Ətrafınızda {n} nəfər var — izdiham"
    SCENE_TWO_PEOPLE = "Ətrafınızda 2 nəfər var"
    SCENE_BUSY = "Mühit çox əşya dolu — {n} əşya aşkarlandı, ehtiyatlı olun"
