import os
import json
import logging
import pandas as pd
from PIL import Image
from moviepy.editor import VideoFileClip
from multiprocessing import Pool, cpu_count
import datetime

# Constants
FPS = 5  # Fixed FPS as confirmed
FRAME_INTERVAL = 1 / FPS

# Paths
EXCEL_PATH = r"/Volumes/T9/EndonasalAR/Endonasal RA - Chronological Video Blocks.xlsx"
VIDEO_DIR = r"/Volumes/T9/EndonasalAR/Structured Patients Image"
OUTPUT_DIR = r"/Volumes/T9/EndonasalAR/Splitted Images"

# Setup logging
def setup_logging(log_path):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

# Convert time formats to total seconds
def time_to_seconds(t):
    if isinstance(t, (int, float)):
        return float(t)
    if pd.isna(t):
        return None
    if isinstance(t, datetime.time):
        return t.minute * 60 + t.second
    s = str(t).strip()
    parts = s.split(':')
    try:
        if len(parts) == 2:
            minutes = int(parts[0]) if parts[0] else 0
            seconds = int(parts[1]) if parts[1] else 0
            return minutes * 60 + seconds
        elif len(parts) == 3:
            hours = int(parts[0]) if parts[0] else 0
            minutes = int(parts[1]) if parts[1] else 0
            seconds = int(parts[2]) if parts[2] else 0
            return hours * 3600 + minutes * 60 + seconds
        else:
            return float(s) if s else None
    except Exception as e:
        logging.warning(f"⚠️ Failed to parse time '{t}': {e}")
        return None

# Build timeline from Excel, adjust for gaps
def build_timeline(df):
    timeline = {}
    for i in range(0, len(df)-1, 2):
        tr = df.iloc[i]
        lr = df.iloc[i+1]
        patient = str(tr['Patient']).strip().upper()
        video = str(tr['Video File Name']).strip()
        key = (patient, video)
        blocks = []
        cols = df.columns[2:]
        for j in range(0, len(cols)-1, 2):
            label = lr[cols[j]]
            start = tr[cols[j]]
            end = tr[cols[j+1]]
            if pd.isna(label) or pd.isna(start) or pd.isna(end):
                continue
            lbl = str(label).strip().replace(' ', '_').upper()
            st = time_to_seconds(start)
            en = time_to_seconds(end)
            if st is None or en is None:
                continue
            blocks.append([lbl, start, end, st, en])

        # Adjust blocks for gaps
        adjusted_blocks = []
        for k in range(len(blocks)):
            current = blocks[k]
            adjusted_blocks.append(current)
            if k < len(blocks) - 1:
                next_blk = blocks[k+1]
                if current[4] < next_blk[3]:
                    gap = next_blk[3] - current[4]
                    if 0.1 < gap <= 1.0:
                        logging.info(f"🔧 Small gap detected between {current[0]} and {next_blk[0]} by {FRAME_INTERVAL:.2f}s")
                        current[4] += FRAME_INTERVAL  # Extend current end
                        next_blk[3] -= FRAME_INTERVAL  # Shrink next start
                    elif gap > 1.0:
                        logging.warning(f"⚠️ Ignored large gap ({gap:.2f}s) between {current[0]} and {next_blk[0]}")
        timeline[key] = adjusted_blocks
    return timeline

# Process a single video: extract frames per block
def process_video(args):
    patient, video, path, out_root, blocks = args
    summary, missing = [], []
    counters = {}
    try:
        clip = VideoFileClip(path)
        duration = clip.duration
        logging.info(f"🎞️ Video {video}: {duration:.2f}s")
    except Exception as e:
        logging.error(f"❌ Cannot open {path}: {e}")
        missing.append(path)
        return summary, missing

    for lbl, raw_start, raw_end, st, en in blocks:
        if st >= en:
            logging.warning(f"⚠️ Skipping {lbl}: start>=end")
            continue
        if st >= duration:
            logging.warning(f"⏭️ Skipping {lbl}: start {st}s beyond {duration:.2f}s")
            continue
        if en > duration:
            logging.warning(f"⚠️ Trimming {lbl} end {en}s to {duration:.2f}s")
            en = duration

        key = f"{patient}_{video}_{lbl}"
        idx = counters.get(key, 0) + 1
        counters[key] = idx
        name = lbl if idx==1 else f"{lbl}_{idx}"

        out_dir = os.path.join(out_root, patient, video, name)
        os.makedirs(out_dir, exist_ok=True)
        logging.info(f"🧩 Extract {name}: {st:.2f}s→{en:.2f}s")

        total = int((en-st)*FPS)
        sub = clip.subclip(st, en)
        count = 0
        for i in range(total):
            t = i / FPS
            try:
                frame = sub.get_frame(t)
                fname = f"{video}_{name}_{i+1:04d}.jpg"
                Image.fromarray(frame).save(os.path.join(out_dir, fname))
                count += 1
            except Exception as ex:
                logging.warning(f"⚠️ Frame {i} at {t:.2f}s failed: {ex}")
        try: sub.reader.close()
        except: pass
        try: sub.audio.reader.close_proc()
        except: pass

        summary.append({
            'Video File': video,
            'Block': name,
            'Start Time': raw_start,
            'End Time': raw_end,
            'Duration(s)': en-st,
            'Frames': count
        })
        with open(os.path.join(out_dir,'metadata.json'),'w') as mf:
            json.dump({'start_sec':st,'end_sec':en,'frames':count}, mf, indent=2)

    clip.close()
    return summary, missing

# Main execution
def main():
    log_file = os.path.join(OUTPUT_DIR, 'process.log')
    setup_logging(log_file)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_excel(EXCEL_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    rename = {c:'Patient' for c in df.columns if 'patient' in c}
    rename.update({c:'Video File Name' for c in df.columns if 'video' in c and 'name' in c})
    df.rename(columns=rename, inplace=True)
    df['Patient'] = df['Patient'].ffill()

    timeline = build_timeline(df)
    tasks = []
    for root, _, files in os.walk(VIDEO_DIR):
        for f in files:
            if not f.lower().endswith('.mp4') or f.startswith('._'):
                continue
            vf = os.path.splitext(f)[0]
            pat = vf.split('_')[0].upper()
            key = (pat, vf)
            if key in timeline:
                logging.info(f"✅ Matched {key}")
                tasks.append((pat, vf, os.path.join(root, f), OUTPUT_DIR, timeline[key]))

    if not tasks:
        logging.warning("⚠️ No videos matched")
        return

    procs = min(cpu_count(), len(tasks))
    logging.info(f"🔧 Using {procs} processes")
    with Pool(procs) as pool:
        results = pool.map(process_video, tasks)

    from collections import defaultdict
    summary = defaultdict(list)
    missing = []
    for logs, miss in results:
        for entry in logs:
            summary[entry['Video File'].split('_')[0].upper()].append(entry)
        missing.extend(miss)

    for pat, logs in summary.items():
        od = os.path.join(OUTPUT_DIR, pat)
        os.makedirs(od, exist_ok=True)
        pd.DataFrame(logs).to_csv(os.path.join(od, f"{pat}_block_summary.csv"), index=False)

    if missing:
        with open(os.path.join(OUTPUT_DIR, 'missing_videos.txt'), 'w') as mf:
            mf.write("\n".join(missing))
        logging.warning(f"⚠️ Missing videos logged: {len(missing)}")

    logging.info("✅ All done")

if __name__ == '__main__':
    main()
