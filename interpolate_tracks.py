import numpy as np
import json


def interpolate(track):
    interpolated_tracks = []
    interpolated_tracks.append(track[0])
    for prv, nxt in zip(track[:-1], track[1:]):
        assert prv[0] < nxt[0], (prv[0], nxt[0])
        vprv = np.array(prv[1])
        vnxt = np.array(nxt[1])
        vdif = vnxt - vprv
        time_diff = nxt[0] - prv[0]
        for i in range(prv[0] + 1, nxt[0]):
            elapsed_time = i - prv[0]
            interpolated_tracks.append((i, (vprv + (vdif * elapsed_time / time_diff)).tolist()))
        interpolated_tracks.append(nxt)
    return interpolated_tracks


def get_reader(filename):
    annotations_reader = open(filename, 'r')
    while True:
        annotation_txt = annotations_reader.readline()
        try:
            annotations = json.loads(annotation_txt)
        except json.JSONDecodeError:
            break

        yield annotations
    annotations_reader.close()

for rr in [2, 4, 8, 16]:
    reader = get_reader(f'jnc00.mp4.sorted.rr{rr}.tracks.jsonl')
    trajectories = {}


    def flush(trajectories, curr_idx, fp):
        removed_tids = []
        for tid, track in trajectories.items():
            if curr_idx is not None and track[-1][0] + 60 > curr_idx:
                continue
            interpolated_track = interpolate(track)
            if len(interpolated_track) < 5:
                print('Interpolating1', tid, len(track), len(interpolated_track), 'frames')
            json.dump([tid, interpolated_track], fp)
            fp.write('\n')
            removed_tids.append(tid)

        print('Removed', len(removed_tids), 'trajectories')
        for tid in removed_tids:
            del trajectories[tid]
    fp = open(f'jnc00.mp4.sorted.rr{rr}.interpolated.tracks.jsonl', 'w')

    index = 0
    for fid, ann in reader:
        print(fid)
        for tid, *bbox in ann:
            if tid not in trajectories:
                trajectories[tid] = []
            trajectories[tid].append((fid, bbox))
        
        if index % 100 == 0:
            flush(trajectories, fid, fp)
        if index > 5000:
            break
        index += 1
        
    flush(trajectories, None, fp)
    assert len(trajectories) == 0
    fp.close()