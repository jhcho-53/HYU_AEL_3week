"""3주차 클릭 라벨링 노트북(notebooks/week3_label_train.ipynb)의 도우미.

rosbag2 를 ROS 없이 읽고(`rosbags`), real_ipm_lab 과 같은 경로로 BEV 를 만들고, Colab 에서 클릭으로 라벨을 찍고,
camsim_lab 과 같은 형식의 데이터셋(images/*.png + labels.csv) 을 씀. `dataset.DiskDataset` 이 그대로 읽음.

BEV 는 real.py 의 함수로 만듦: ost.yaml 로 왜곡 보정 -> `real.camsim_h_i2g` -> `real.bev_from_camera`.
카메라 자세(pitch, 높이)는 real_ipm_lab 3장이 bag 에서 추정한 값을 숫자로 받음 (그 노트북을 먼저 돌릴 필요 없음).
"""
import base64
import csv
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from . import real
from .real import IMAGE_TOPIC, LABEL_HEADER

MIN_ACCEPTED = 5          # 노트북이 20 % 를 검증용으로 떼므로 최소 한 장은 남아야 함
SAMPLE = Path(__file__).resolve().parents[1]/'examples'/'week3_bag'
LABELER_JS = Path(__file__).with_name('week3_labeler.js')
# examples/week3_bag 의 카메라: real_ipm_lab 3장이 원본 bag(run_train2_part1) 에서 추정한 값 (docs/real_ipm_manual.md 2절)
SAMPLE_PITCH_DEG, SAMPLE_HEIGHT_M, SAMPLE_LANE_WIDTH_M = -4.25, 0.178, 0.8
# cv_bridge 와 같은 변환 (OpenCV 는 Bayer 패턴 이름을 한 행 아래 기준으로 붙임)
BAYER = {'bayer_rggb8': cv2.COLOR_BayerBG2BGR, 'bayer_bggr8': cv2.COLOR_BayerRG2BGR,
         'bayer_gbrg8': cv2.COLOR_BayerGR2BGR, 'bayer_grbg8': cv2.COLOR_BayerGB2BGR}
COLOUR = {'bgr8': (3, None), 'rgb8': (3, cv2.COLOR_RGB2BGR), 'bgra8': (4, cv2.COLOR_BGRA2BGR),
          'rgba8': (4, cv2.COLOR_RGBA2BGR), 'mono8': (1, cv2.COLOR_GRAY2BGR)}


def to_bgr(msg):
    """sensor_msgs/Image (rosbags 가 역직렬화한 것) -> uint8 BGR 영상."""
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    data = np.asarray(msg.data, np.uint8)[:h * step].reshape(h, step)
    if msg.encoding in BAYER:
        return cv2.cvtColor(np.ascontiguousarray(data[:, :w]), BAYER[msg.encoding])
    if msg.encoding in COLOUR:
        channels, code = COLOUR[msg.encoding]
        image = np.ascontiguousarray(data[:, :w * channels]).reshape(h, w, channels) if channels > 1 else data[:, :w]
        return image.copy() if code is None else cv2.cvtColor(np.ascontiguousarray(image), code)
    raise ValueError(f'{msg.encoding}: 지원하지 않는 영상 인코딩 (bayer_*8, bgr8, rgb8, bgra8, rgba8, mono8만 됨).')


def _reader(bag):
    """bag 폴더, 또는 그 안의 .db3/.mcap 파일 하나의 rosbags Reader (Colab 파일 창은 폴더가 아니라 파일을 올림)."""
    from rosbags.rosbag2 import Reader
    path = Path(bag)
    if not path.exists():
        raise ValueError(f'{path}: 없는 경로. Colab 파일 창에 올린 파일은 /content/ 아래에 있음 (예: /content/my_run_0.db3).')
    if path.is_dir() and not (path/'metadata.yaml').is_file():
        raise ValueError(f'{path}: metadata.yaml이 없는 폴더. bag 폴더나 그 안의 .db3 파일 하나를 지정할 것.')
    return Reader(path)


def bag_summary(bag):
    """bag(폴더 또는 .db3 하나)의 파일, 길이, 토픽: `ros2 bag info` 가 보여 주는 것."""
    path = Path(bag)
    with _reader(path) as reader:
        return dict(files=sorted(p.name for p in path.iterdir()) if path.is_dir() else [path.name],
                    duration_s=reader.duration / 1e9,
                    topics=[dict(topic=c.topic, type=c.msgtype, count=c.msgcount) for c in reader.connections])


def sampler(every_s):
    """keep(stamp_ns) -> every_s 간격마다 첫 번째 stamp 에서 True.

    조금 이른 프레임(지터)도 받음: 간격의 10 % 까지, 단 지금까지 본 카메라 프레임 간격의 절반을 넘지 않게.
    그래서 40 Hz 녹화는 0.5 s, 1.0 s, ... 영상을 남기고, 이미 2 Hz 로 줄인 bag 은 전부 남김.
    """
    step = int(round(every_s * 1e9))
    if step <= 0:
        raise ValueError('every_s는 0보다 커야 함.')
    due = previous = None
    interval = step

    def keep(stamp):
        nonlocal due, previous, interval
        if previous is not None:
            interval = min(interval, stamp - previous)
        previous = stamp
        slack = min(step // 10, interval // 2)
        if due is not None and stamp < due - slack:
            return False
        due = stamp + step if due is None else due + step
        while due - slack <= stamp:               # 녹화가 끊긴 구간이 지나친 단계는 건너뜀
            due += step
        return True
    return keep


def read_frames(bag, every_s=.5, topic=IMAGE_TOPIC):
    """카메라 시각(header stamp) 기준 every_s 초마다 한 장, [{'stamp_ns', 'bgr'}].

    앞 영상보다 늦지 않은 stamp(반복, 시계 되감김)는 세어 두고 건너뜀. 그런 영상이 대부분인 bag 은 오류.
    """
    from rosbags.typesys import Stores, get_typestore
    store, keep = get_typestore(Stores.ROS2_HUMBLE), sampler(every_s)
    frames, last, skipped, total = [], None, 0, 0
    with _reader(bag) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            raise ValueError(f'{topic} 토픽이 bag에 없음. 있는 토픽: {sorted({c.topic for c in reader.connections})}')
        for connection, _, raw in reader.messages(connections=connections):
            msg = store.deserialize_cdr(raw, connection.msgtype)
            stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            total += 1
            if last is not None and stamp <= last:
                skipped += 1
                continue
            last = stamp
            if keep(stamp):
                frames.append(dict(stamp_ns=stamp, bgr=to_bgr(msg)))
    if skipped * 2 > total:
        raise ValueError(f'{topic} 영상 {total}개 중 {skipped}개의 시각(header stamp)이 앞 영상보다 늦지 않음. '
                         '녹화 중에 시계가 바뀌었거나 카메라 드라이버가 시각을 찍지 않은 bag임.')
    if skipped:
        print(f'영상 시각(header stamp)이 앞 영상과 같거나 거꾸로 간 메시지 {skipped}개는 건너뜀.')
    if not frames:
        raise ValueError(f'{topic}에 영상 메시지가 없음.')
    return frames


class Camera:
    """bag 을 녹화한 카메라: ost.yaml(K, D) + real_ipm_lab 이 추정한 자세. real_ipm_lab 과 같은 BEV 와 자동 라벨을 냄."""

    def __init__(self, ost, cfg, pitch_deg, height_m, offset_x_m=0.0):
        self.K, self.D, self.size = real.load_ost(ost)
        self.maps = real.undistort_maps(self.K, self.D, self.size)
        self.cfg, self.pitch_deg, self.height_m, self.offset_x_m = cfg, float(pitch_deg), float(height_m), float(offset_x_m)
        self.H_i2g = real.camsim_h_i2g(self.K, self.size, cfg, self.pitch_deg, self.height_m, self.offset_x_m)   # 모델 입력
        self.H_full = real.ground_homography(self.K, self.pitch_deg, self.height_m, self.offset_x_m)          # 원본 해상도
        self.v_min = real.horizon_row(self.K, self.pitch_deg) + 15        # real_ipm_lab 3장의 V_MARGIN 과 같음

    def undistort(self, bgr):
        h, w = bgr.shape[:2]
        if (w, h) != tuple(self.size):
            raise ValueError(f'영상 {w}x{h}이 ost.yaml의 {self.size[0]}x{self.size[1]}과 다름. '
                             '이 bag을 녹화한 카메라 설정의 ost.yaml을 쓸 것.')
        return cv2.remap(bgr, *self.maps, interpolation=cv2.INTER_LINEAR)

    def bev(self, bgr):
        """모델 입력 BEV (380 x 300). real_ipm_lab 4장과 같은 경로."""
        return real.bev_from_camera(self.undistort(bgr), self.H_i2g, self.cfg)

    def auto_label(self, bgr, lane_width_m):
        """real_ipm_lab 5장의 자동 라벨 (x, y) m. 차선 둘을 못 찾으면 nan."""
        return real.label_frame(self.undistort(bgr), self.H_full, lane_width_m, self.cfg.waypoints.ahead_m, self.v_min)['wp']


def make_bevs(frames, camera):
    return [camera.bev(f['bgr']) for f in frames]


def auto_labels(frames, camera, lane_width_m):
    return np.array([camera.auto_label(f['bgr'], lane_width_m) for f in frames], dtype=float).reshape(-1, 2)


def ring_point(point, ahead_m, bev):
    """클릭한 점을 후륜축 중심 ahead_m 원 위로 옮김: 실차 라벨의 정의.

    시뮬 정답은 중심선을 따라 ahead_m 호길이 앞의 점. 실차는 이 원과 차선 가운데가 만나는 점을 씀. 둘은 몇 cm,
    주로 차선 방향으로 다름.
    """
    p = np.asarray(point, dtype=float)
    if p.shape != (2,) or not np.isfinite(p).all():
        raise ValueError('waypoint는 유한한 (x, y) 미터 좌표여야 함.')
    r = float(np.hypot(*p))
    if r < 1e-6:
        raise ValueError('후륜축에서 떨어진 곳을 클릭할 것.')
    wp = p * (float(ahead_m) / r)
    if not (bev['x_range_m'][0] <= wp[0] <= bev['x_range_m'][1] and bev['y_range_m'][0] <= wp[1] <= bev['y_range_m'][1]):
        raise ValueError('waypoint가 BEV 영상 범위 밖임. 차선 가운데가 안 보이면 이 프레임은 제외할 것.')
    return wp


def png_url(image):
    ok, data = cv2.imencode('.png', image)
    if not ok:
        raise ValueError('PNG 인코딩 실패')
    return 'data:image/png;base64,' + base64.b64encode(data.tobytes()).decode()


class Labeler:
    """read_frames 의 프레임별 BEV 에 클릭 라벨: ahead_m 원 위의 점 하나, 또는 'rejected'.

    바뀔 때마다 프레임 stamp 와 BEV 의 해시와 함께 JSON 으로 저장: 셀을 다시 실행해도 클릭이 남고,
    다른 bag, every_s, 카메라 자세면 새로 시작함.
    """

    def __init__(self, frames, bevs, cfg, path):
        if len(frames) != len(bevs):
            raise ValueError(f'프레임 {len(frames)}장과 BEV {len(bevs)}장의 수가 다름. BEV 셀부터 다시 실행할 것.')
        self.bevs, self.path = bevs, Path(path)
        self.stamps = [int(f['stamp_ns']) for f in frames]
        digest = hashlib.sha1()
        for bev in bevs:
            digest.update(np.ascontiguousarray(bev))
        self.bevs_sha1 = digest.hexdigest()
        self.ahead_m = float(cfg.waypoints.ahead_m)
        self.bev = dict(x_range_m=list(cfg.bev.x_range_m), y_range_m=list(cfg.bev.y_range_m),
                        resolution_m=float(cfg.bev.resolution_m))
        self.labels = [dict(status='unlabeled', waypoint_m=None) for _ in bevs]
        if self.path.exists():
            saved = json.loads(self.path.read_text())
            if isinstance(saved, dict) and (saved.get('stamps'), saved.get('bevs_sha1')) == (self.stamps, self.bevs_sha1):
                self.labels = saved['labels']

    def counts(self):
        statuses = [label['status'] for label in self.labels]
        return {s: statuses.count(s) for s in ('accepted', 'rejected', 'unlabeled')}

    def frame(self, index):
        index = self._index(index)
        return dict(index=index, n=len(self.bevs), image=png_url(self.bevs[index]), label=self.labels[index],
                    counts=self.counts(), statuses=[label['status'] for label in self.labels])

    def save(self, index, status, x=None, y=None):
        index = self._index(index)
        if status not in ('accepted', 'rejected'):
            raise ValueError("status는 'accepted' 또는 'rejected'여야 함.")
        point = ring_point([x, y], self.ahead_m, self.bev).tolist() if status == 'accepted' else None
        self.labels[index] = dict(status=status, waypoint_m=point)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(dict(stamps=self.stamps, bevs_sha1=self.bevs_sha1, labels=self.labels), indent=1))
        return dict(label=self.labels[index], counts=self.counts(), statuses=[label['status'] for label in self.labels])

    def _index(self, index):
        index = int(index)
        if not 0 <= index < len(self.bevs):
            raise ValueError(f'프레임 번호는 0~{len(self.bevs) - 1}임.')
        return index

    def html(self, prefix='camsim.week3.'):
        config = dict(n=len(self.bevs), ahead_m=self.ahead_m, bev=self.bev, scale=2, prefix=prefix)
        return f"""<div id="w3"><style>
#w3{{font:15px system-ui,sans-serif}} #w3 .bar{{display:flex;gap:8px;align-items:center;margin:6px 0;flex-wrap:wrap}}
#w3 button{{font:inherit;padding:6px 12px;border:1px solid #8a97ad;border-radius:5px;background:#eef2f8;cursor:pointer}}
#w3 .accept{{background:#d4f0e2}} #w3 .reject{{background:#f6dcdc}} #w3 canvas{{border:1px solid #8a97ad;cursor:crosshair;touch-action:none}}
#w3 #msg{{color:#9a5a00;min-height:22px}}</style>
<div class="bar"><button id="prev">← 이전</button><b id="pos"></b><button id="next">다음 →</button>
<button id="todo">다음 미작업</button><span id="counts"></span></div>
<canvas id="cv"></canvas>
<div class="bar"><button id="accept" class="accept">승인 (Enter)</button><button id="reject" class="reject">제외 (X)</button>
<span>이 프레임: <b id="state"></b></span></div><div id="msg"></div></div>
<script>const CONFIG={json.dumps(config)};
async function call(name,args){{const r=await google.colab.kernel.invokeFunction(CONFIG.prefix+name,args,{{}});
const d=r.data['application/json'];if(d.error)throw Error(d.error);return d}}</script>
<script>{LABELER_JS.read_text()}</script>"""

    def show(self, prefix='camsim.week3.'):
        """Colab 전용: 화면이 부르는 콜백 둘을 등록하고 이 셀 출력에 라벨링 화면을 그림."""
        from IPython.display import HTML, display
        try:
            from google.colab import output
        except ImportError:
            print('클릭 라벨링 화면은 Colab에서만 뜸 (google.colab 없음). Colab에서 이 노트북을 열 것.')
            return
        output.register_callback(prefix + 'frame', _answer(self.frame))
        output.register_callback(prefix + 'save', _answer(self.save))
        display(HTML(self.html(prefix)))


def _answer(method):
    """method 를 Colab 콜백으로: 결과를 JSON 으로, 예외는 화면이 보여 주는 {'error': 메시지} 로."""
    from IPython.display import JSON

    def callback(*args):
        try:
            return JSON(method(*args))
        except ValueError as exc:                  # 학생에게 쓴 우리 검사
            return JSON(dict(error=str(exc)))
        except Exception as exc:                   # 그 밖의 것: 화면은 traceback 을 못 보여 주니 이름이라도
            return JSON(dict(error=f'{type(exc).__name__}: {exc}'))
    return callback


def write_dataset(bevs, labels, out_dir, prefix='frame'):
    """승인한 프레임 -> out_dir/images/*.png + out_dir/labels.csv: camsim 의 DiskDataset 이 읽는 폴더.

    실차 녹화에는 차의 위치가 없어서 x, y, theta 는 nan (real_ipm_lab 의 LabelWriter 와 같음). 학습은 wp_x, wp_y 만 씀.
    """
    if len(labels) != len(bevs):
        raise ValueError(f'BEV {len(bevs)}장과 라벨 {len(labels)}개의 수가 다름. 라벨링 셀부터 다시 실행할 것.')
    accepted = [i for i, label in enumerate(labels) if label['status'] == 'accepted']
    if len(accepted) < MIN_ACCEPTED:
        raise ValueError(f'승인한 프레임이 {len(accepted)}장임. 20%를 검증용으로 떼고 학습하려면 '
                         f'{MIN_ACCEPTED}장 이상 점을 찍고 승인(Enter)할 것.')
    out = Path(out_dir)
    if out.exists():
        if any(p.name not in ('images', 'labels.csv') for p in out.iterdir()):
            raise ValueError(f'{out}에 데이터셋이 아닌 파일이 있어 지우지 않았음. 다른 폴더 이름을 쓸 것.')
        shutil.rmtree(out)
    (out/'images').mkdir(parents=True)
    rows = []
    for i in accepted:
        name, (x, y) = f'{prefix}_{i:04d}.png', labels[i]['waypoint_m']
        cv2.imwrite(str(out/'images'/name), bevs[i])
        rows.append([name, 'nan', 'nan', 'nan', f'{x:.4f}', f'{y:.4f}'])
    with (out/'labels.csv').open('w', newline='') as stream:
        csv.writer(stream, lineterminator='\n').writerows([LABEL_HEADER, *rows])
    return len(rows)
