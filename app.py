# app.py
import os
import time
import traceback
from datetime import datetime, timezone, timedelta
from collections import Counter
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

# ML / NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# YouTube API
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# ------------------------------------------------------
# LOAD API KEY
# ------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    raise RuntimeError("Set YOUTUBE_API_KEY in .env file")

app = Flask(__name__, template_folder="templates")

MAX_CHANNELS = 5
SHORTS_MAX_SECONDS = 60
BATCH_SIZE = 50
RECENT_DAYS = 30


# ------------------------------------------------------
# HELPERS
# ------------------------------------------------------
def get_youtube_service():
    return build("youtube", "v3", developerKey=API_KEY, static_discovery=False)


def short_number(n):
    try:
        n = int(n)
    except:
        return str(n or "0")
    if n < 1000:
        return str(n)
    value = float(n)
    for unit in ["", "K", "M", "B", "T"]:
        if abs(value) < 1000:
            if unit == "":
                return f"{int(value)}"
            if value >= 100:
                return f"{int(round(value))}{unit}"
            return f"{round(value, 1)}{unit}"
        value /= 1000.0
    return f"{round(value,1)}T"


def human_date(iso):
    if not iso:
        return "N/A"
    try:
        if iso.endswith("Z"):
            dt = datetime.fromisoformat(iso.replace("Z","+00:00"))
        else:
            dt = datetime.fromisoformat(iso)
        return dt.strftime("%B %d, %Y — %I:%M %p")
    except:
        return "N/A"


def parse_iso8601_duration_to_seconds(dur: str) -> int:
    if not dur:
        return 0
    dur = dur.replace("PT", "")
    num = ""
    sec = 0
    for ch in dur:
        if ch.isdigit():
            num += ch
        else:
            if num == "":
                continue
            if ch == "H":
                sec += int(num) * 3600
            elif ch == "M":
                sec += int(num) * 60
            elif ch == "S":
                sec += int(num)
            num = ""
    return sec


def extract_channel_from_input(inp: str):
    s = (inp or "").strip()
    if not s:
        return None, None

    # Raw channel ID
    if s.startswith("UC") and len(s) > 20:
        return s, f"https://www.youtube.com/channel/{s}"

    if "youtube.com" in s:
        if "/channel/" in s:
            cid = s.split("/channel/")[1].split("/")[0].split("?")[0]
            return cid, f"https://www.youtube.com/channel/{cid}"
        base = s.split("?")[0].rstrip("/")
        return None, base

    if s.startswith("@"):
        return None, f"https://www.youtube.com/{s}"

    return None, s


# ------------------------------------------------------
# YOUTUBE API HELPERS
# ------------------------------------------------------
def fetch_channel_basic(youtube, channel_id):
    resp = youtube.channels().list(
        part="snippet,statistics,contentDetails",
        id=channel_id
    ).execute()

    items = resp.get("items", [])
    if not items:
        return None

    it = items[0]
    sn = it.get("snippet", {})
    st = it.get("statistics", {})
    cd = it.get("contentDetails", {})

    profile = (sn.get("thumbnails") or {}).get("high", {}).get("url")
    subs = int(st.get("subscriberCount", 0)) if st.get("subscriberCount") else 0

    return {
        "channel_id": channel_id,
        "title": sn.get("title"),
        "profile_pic": profile,
        "subscribers": subs,
        "uploads_playlist": cd.get("relatedPlaylists", {}).get("uploads")
    }


def resolve_handle_to_channel_id(youtube, canonical_url_or_handle):
    try:
        q = canonical_url_or_handle.strip().split("/")[-1]
        resp = youtube.search().list(
            part="snippet",
            q=q,
            type="channel",
            maxResults=1
        ).execute()
        items = resp.get("items", [])
        if items:
            return items[0]["snippet"]["channelId"]
    except:
        return None
    return None


def fetch_all_video_ids_from_playlist(youtube, playlist):
    ids = []
    next_page = None

    while True:
        resp = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist,
            maxResults=50,
            pageToken=next_page
        ).execute()

        for it in resp.get("items", []):
            vid = it["contentDetails"].get("videoId")
            if vid:
                ids.append(vid)

        next_page = resp.get("nextPageToken")
        if not next_page:
            break

    return ids


def fetch_videos_metadata(youtube, ids):
    out = []
    for i in range(0, len(ids), BATCH_SIZE):
        batch = ids[i:i+BATCH_SIZE]
        resp = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=",".join(batch)
        ).execute()

        for v in resp.get("items", []):
            sn = v.get("snippet", {})
            st = v.get("statistics", {})
            cd = v.get("contentDetails", {})

            dur_sec = parse_iso8601_duration_to_seconds(cd.get("duration", ""))

            out.append({
                "id": v.get("id"),
                "title": sn.get("title", ""),
                "description": sn.get("description", "") or "",
                "publishedAt": sn.get("publishedAt"),
                "views": int(st.get("viewCount", 0)),
                "duration_seconds": dur_sec,
            })

    return out


# ------------------------------------------------------
# ORIGINALITY ENGINE
# ------------------------------------------------------
def metadata_originality_analysis(videos_meta):
    N = len(videos_meta)
    titles = [v["title"].strip() for v in videos_meta]
    descs = [v["description"].strip() for v in videos_meta]
    durations = [v["duration_seconds"] for v in videos_meta]

    desc_counts = Counter(descs)
    desc_dup_ratio = sum(1 for d in descs if d and desc_counts[d] > 1) / max(1, N)
    empty_desc_ratio = sum(1 for d in descs if not d or len(d) < 10) / max(1, N)

    shorts_ratio = sum(1 for d in durations if d <= SHORTS_MAX_SECONDS) / max(1, N)

    avg_dur = sum(durations) / max(1, N)
    std_dur = (sum((x-avg_dur)**2 for x in durations) / max(1, N)) ** 0.5
    dur_uniformity = 1 - (std_dur / (avg_dur + 1))

    score = 100
    score -= 20 * (desc_dup_ratio + 0.5 * empty_desc_ratio)
    score -= 15 * shorts_ratio
    score -= 10 * (1 - dur_uniformity)

    final_score = int(max(0, min(100, score)))

    explanation = []
    if desc_dup_ratio > 0.2:
        explanation.append("Duplicate descriptions detected.")
    if empty_desc_ratio > 0.4:
        explanation.append("Many videos have empty descriptions.")
    if shorts_ratio > 0.85:
        explanation.append("Mostly shorts content — reused content risk.")
    if dur_uniformity < 0.3:
        explanation.append("Durations too uniform — templated content.")
    if not explanation:
        explanation.append("No reused-content signals detected.")

    return final_score, explanation, {
        "desc_dup_ratio": desc_dup_ratio,
        "empty_desc_ratio": empty_desc_ratio,
        "shorts_ratio": shorts_ratio,
        "dur_uniformity": dur_uniformity
    }


# ------------------------------------------------------
# FULL CHANNEL ANALYZER
# ------------------------------------------------------
def analyze_channel_full(channel_input):
    youtube = get_youtube_service()

    cid, canonical = extract_channel_from_input(channel_input)
    if not cid:
        cid = resolve_handle_to_channel_id(youtube, canonical or channel_input)
        if not cid:
            return {"error": "Could not resolve channel ID"}

    ch = fetch_channel_basic(youtube, cid)
    if not ch:
        return {"error": "Channel not found"}

    uploads = ch["uploads_playlist"]
    video_ids = fetch_all_video_ids_from_playlist(youtube, uploads) if uploads else []
    videos_meta = fetch_videos_metadata(youtube, video_ids)

    # Split shorts & long videos
    shorts = [v for v in videos_meta if v["duration_seconds"] <= SHORTS_MAX_SECONDS]
    longs  = [v for v in videos_meta if v["duration_seconds"] > SHORTS_MAX_SECONDS]

    shorts_count = len(shorts)
    long_count = len(longs)

    total_shorts_views = sum(v["views"] for v in shorts)
    total_long_views   = sum(v["views"] for v in longs)

    # Last uploads
    def latest(vlist):
        if not vlist:
            return "N/A"
        try:
            return human_date(max(vlist, key=lambda x: x["publishedAt"])["publishedAt"])
        except:
            return "N/A"

    last_short = latest(shorts)
    last_video = latest(longs)

    # 30-day stats
    cut = datetime.now(timezone.utc) - timedelta(days=RECENT_DAYS)
    shorts_30 = []
    longs_30 = []

    for v in videos_meta:
        try:
            dt = datetime.fromisoformat(v["publishedAt"].replace("Z","+00:00"))
        except:
            continue

        if dt >= cut:
            if v["duration_seconds"] <= SHORTS_MAX_SECONDS:
                shorts_30.append(v)
            else:
                longs_30.append(v)

    shorts_30_count = len(shorts_30)
    longs_30_count  = len(longs_30)

    shorts_30_views = sum(v["views"] for v in shorts_30)
    longs_30_views  = sum(v["views"] for v in longs_30)

    # Average durations
    def avg_duration(lst):
        return sum(v["duration_seconds"] for v in lst) / len(lst) if lst else 0

    avg_short = avg_duration(shorts)
    avg_long  = avg_duration(longs)

    avg_short_human = f"{int(avg_short//60)}m {int(avg_short%60)}s" if avg_short else "N/A"
    avg_long_human  = f"{int(avg_long//60)}m {int(avg_long%60)}s" if avg_long else "N/A"

    # Frequency
    def freq(count):
        if count == 0:
            return "Rarely uploads"
        per_week = (count / RECENT_DAYS) * 7
        if per_week < 1:
            days = int(7 / per_week)
            return f"1 upload every {days} days"
        return f"{per_week:.1f} uploads/week"

    shorts_frequency = freq(shorts_30_count)
    videos_frequency = freq(longs_30_count)

    uploads_per_week_recent = round((shorts_30_count + longs_30_count) / RECENT_DAYS * 7, 2)

    # Originality
    originality_score, originality_explanation, originality_signals = metadata_originality_analysis(videos_meta)

    return {
        "channel_id": ch["channel_id"],
        "title": ch["title"],
        "profile_pic": ch["profile_pic"],
        "subscribers": ch["subscribers"],
        "subscribers_fmt": short_number(ch["subscribers"]),
        "monetization": "UNKNOWN — YouTube does not show publicly",

        "originality_score": originality_score,
        "originality_explanation": originality_explanation,
        "originality_signals": originality_signals,

        "shorts_count": shorts_count,
        "videos_count": long_count,
        "videos_scanned": len(videos_meta),

        "total_shorts_views_lifetime": total_shorts_views,
        "total_long_views_lifetime": total_long_views,

        "total_shorts_views_lifetime_fmt": short_number(total_shorts_views),
        "total_long_views_lifetime_fmt": short_number(total_long_views),

        "shorts_30d_count": shorts_30_count,
        "videos_30d_count": longs_30_count,

        "total_shorts_views_30d": shorts_30_views,
        "total_long_views_30d": longs_30_views,

        "total_shorts_views_30d_fmt": short_number(shorts_30_views),
        "total_long_views_30d_fmt": short_number(longs_30_views),

        "shorts_frequency_text": shorts_frequency,
        "videos_frequency_text": videos_frequency,
        "uploads_per_week_recent": uploads_per_week_recent,

        "avg_short_duration_human": avg_short_human,
        "avg_long_duration_human": avg_long_human,

        "last_uploaded_short": last_short,
        "last_uploaded_video": last_video
    }


# ------------------------------------------------------
# ROUTES
# ------------------------------------------------------
@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/preview", methods=["POST"])
def preview():
    try:
        data = request.get_json()
        if not data or "input" not in data:
            return jsonify({"success": False, "error": "Invalid request"}), 400

        inp = data["input"].strip()
        youtube = get_youtube_service()

        cid, canonical = extract_channel_from_input(inp)
        if not cid:
            cid = resolve_handle_to_channel_id(youtube, canonical or inp)
        if not cid:
            return jsonify({"success": False, "error": "Channel not found"}), 404

        ch = fetch_channel_basic(youtube, cid)
        if not ch:
            return jsonify({"success": False, "error": "Channel not found"}), 404

        return jsonify({
            "success": True,
            "channel_id": ch["channel_id"],
            "title": ch["title"],
            "profile_pic": ch["profile_pic"],
            "subscribers": ch["subscribers"],
            "subscribers_fmt": short_number(ch["subscribers"])
        })

    except Exception as e:
        print("Preview Error:", e)
        return jsonify({"success": False, "error": "Server error"}), 500


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    try:
        data = request.get_json()
        inputs = data.get("channels", [])

        results = []
        errors = []

        for c in inputs[:MAX_CHANNELS]:
            report = analyze_channel_full(c)
            if "error" in report:
                errors.append({"channel": c, "error": report["error"]})
            else:
                results.append(report)

        return jsonify({"success": True, "results": results, "errors": errors})

    except Exception as e:
        print("Analyze Error:", e)
        return jsonify({"success": False, "error": "Server error"}), 500


# ------------------------------------------------------
# FOOTER PAGES
# ------------------------------------------------------
@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


@app.route("/terms")
def terms():
    return render_template("terms.html")


@app.route("/disclaimer")
def disclaimer():
    return render_template("disclaimer.html")


# ------------------------------------------------------
# RUN
# ------------------------------------------------------
if __name__ == "__main__":
    print("Starting YouTube Live Analyzer...")
    app.run(debug=True, host="127.0.0.1", port=5000)
