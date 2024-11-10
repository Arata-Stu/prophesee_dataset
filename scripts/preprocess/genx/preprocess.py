import os
import h5py
import hdf5plugin
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, get_context
import yaml
import argparse
import re
import cv2  # OpenCVを使用して解像度変更

# BBOXデータタイプ
BBOX_DTYPE = np.dtype({
    'names': ['t', 'x', 'y', 'w', 'h', 'class_id', 'track_id', 'class_confidence'],
    'formats': ['<i8', '<f4', '<f4', '<f4', '<f4', '<u4', '<u4', '<f4'],
    'offsets': [0, 8, 12, 16, 20, 24, 28, 32],
    'itemsize': 40
})

def find_event_and_bbox_files(sequence_dir, mode):
    """指定されたディレクトリ内でイベントデータとBBOXファイルを検索し、`gen1`と`gen4`で処理を分ける"""
    files = os.listdir(sequence_dir)
    event_file = None
    bbox_file = None
    if mode == 'gen1':
        event_pattern = r'_td\.dat\.h5$'
        bbox_pattern = r'_bbox\.npy$'
    elif mode == 'gen4':
        event_pattern = r'_td\.h5$'
        bbox_pattern = r'_bbox\.npy$'
    else:
        raise ValueError(f"Unknown mode: {mode}")

    for file in files:
        if re.search(event_pattern, file):
            event_file = file
        elif re.search(bbox_pattern, file):
            bbox_file = file

    if not event_file or not bbox_file:
        raise FileNotFoundError(f"イベントまたはBBOXファイルが見つかりません: {sequence_dir}")
    return event_file, bbox_file

def create_event_frame(slice_events, frame_shape, downsample=False):
    height, width = frame_shape
    frame = np.ones((height, width, 3), dtype=np.uint8) * 114  # グレーバックグラウンド

    # オン・オフイベントのマスクを作成
    off_events = (slice_events['p'] == -1)
    on_events = (slice_events['p'] == 1)

    # オンイベントを赤、オフイベントを青に割り当て
    frame[slice_events['y'][off_events], slice_events['x'][off_events]] = np.array([0, 0, 255], dtype=np.uint8)
    frame[slice_events['y'][on_events], slice_events['x'][on_events]] = np.array([255, 0, 0], dtype=np.uint8)

    # downsampleがTrueの場合、解像度を半分にする
    if downsample:
        frame = cv2.resize(frame, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)
        
    return frame

def process_sequence(args):
    data_dir, output_dir, split, seq, tau_ms, delta_t_ms, frame_shape, mode, downsample = args
    print(f"Processing : {seq} in split: {split}")

    tau_us = tau_ms * 1000
    delta_t_us = delta_t_ms * 1000

    # 出力ディレクトリの作成
    seq_output_dir = os.path.join(output_dir, split, seq, f"tau={tau_ms}_dt={delta_t_ms}")
    os.makedirs(seq_output_dir, exist_ok=True)

    sequence_dir = os.path.join(data_dir, split, seq)
    event_file, bbox_file = find_event_and_bbox_files(sequence_dir, mode)

    # データファイルパスの取得
    event_path = os.path.join(sequence_dir, event_file)
    bbox_path = os.path.join(sequence_dir, bbox_file)

    # イベントデータの読み込み
    with h5py.File(event_path, 'r') as f:
        events = {
            't': f['events']['t'][:],
            'x': f['events']['x'][:],
            'y': f['events']['y'][:],
            'p': f['events']['p'][:]
        }

    # BBOXデータの読み込み
    if os.path.exists(bbox_path):
        detections = np.load(bbox_path)
    else:
        detections = np.array([], dtype=BBOX_DTYPE)

    events['t'] = events['t'].astype(np.float64)
    detections['t'] = detections['t'].astype(np.float64)

    # 開始時刻の調整
    start_time = max(events['t'][0], 10000)  # 10000 usから開始

    # ウィンドウの中心時刻を定義
    window_times = np.arange(start_time, events['t'][-1], tau_us)

    # メイン処理
    for t in window_times:
        data_start = t - delta_t_us
        data_end = t

        # イベントデータのスライス（delta_tの範囲でスライス）
        start_idx = np.searchsorted(events['t'], data_start)
        end_idx = np.searchsorted(events['t'], data_end)

        slice_events = {
            't': events['t'][start_idx:end_idx],
            'x': events['x'][start_idx:end_idx],
            'y': events['y'][start_idx:end_idx],
            'p': events['p'][start_idx:end_idx],
        }

        # ラベルの取得（tau_ms基準で取得）
        if detections.size > 0:
            label_mask = (detections['t'] >= (t - tau_us / 2)) & (detections['t'] < (t + tau_us / 2))
            slice_detections = detections[label_mask]

            unique_detections = {}
            for det in slice_detections:
                track_id = det['track_id']
                if track_id not in unique_detections or det['t'] > unique_detections[track_id]['t']:
                    unique_detections[track_id] = det

            labels = [
                {
                    't': det['t'],
                    'x': det['x'] // 2 if downsample else det['x'],  # downsampleがTrueの場合に座標を半分に
                    'y': det['y'] // 2 if downsample else det['y'],
                    'w': det['w'] // 2 if downsample else det['w'],
                    'h': det['h'] // 2 if downsample else det['h'],
                    'class_id': det['class_id'],
                    'class_confidence': det['class_confidence'],
                    'track_id': det['track_id']
                }
                for det in unique_detections.values()
            ]
        else:
            labels = []

        # フレームの作成
        event_frame = create_event_frame(slice_events, frame_shape, downsample=downsample)

        # 出力ファイルの保存
        output_file = os.path.join(seq_output_dir, f"{int(data_start)}_to_{int(data_end)}.npz")
        if os.path.exists(output_file):
            continue

        np.savez_compressed(output_file, event=event_frame, labels=labels)

    print(f"Completed processing sequence: {seq} in split: {split}")

def main(config):
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    num_processors = config.get("num_processors", cpu_count())
    tau_ms = config["tau_ms"]
    delta_t_ms = config["delta_t_ms"]
    frame_shape = tuple(config["frame_shape"])
    mode = config.get("mode", "gen1")
    downsample = config.get("downsample", False)  # デフォルトはFalse

    splits = ['train', 'test', 'val']
    sequences = [(split, seq) for split in splits for seq in os.listdir(os.path.join(input_dir, split)) if os.path.isdir(os.path.join(input_dir, split, seq))]

    with tqdm(total=len(sequences), desc="Processing sequences") as pbar:
        with get_context('spawn').Pool(processes=num_processors) as pool:
            args_list = [(input_dir, output_dir, split, seq, tau_ms, delta_t_ms, frame_shape, mode, downsample) for split, seq in sequences]
            for _ in pool.imap_unordered(process_sequence, args_list):
                pbar.update()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process gen1/gen4 dataset with configuration file")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration YAML file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)
