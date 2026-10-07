"""camsim 환경 점검. 뭐가 깔려 있고 어떤 모드를 돌릴 수 있는지 보여줌.

표준 라이브러리만 쓰므로 아무것도 설치 안 된 상태에서도 돎.

    python camsim/scripts/check_env.py
"""
import importlib.util
import platform
import sys

# 모듈 이름 -> pip 패키지 이름
CORE = [("numpy", "numpy"), ("cv2", "opencv-python-headless"), ("yaml", "pyyaml"),
        ("PIL", "pillow"), ("torch", "torch"), ("torchvision", "torchvision")]
SIM = [("gym", "gym==0.19.0"), ("f110_gym", "f110_gym (이 레포)")]
LAB = [("matplotlib", "matplotlib"), ("pandas", "pandas"),
       ("imageio_ffmpeg", "imageio-ffmpeg"), ("onnx", "onnx"), ("onnxruntime", "onnxruntime")]

# 모드 -> 필요한 모듈
MODES = [
    ("데이터 생성",      ["numpy", "cv2", "yaml", "PIL", "torch"]),
    ("학습 (resnet18)",  ["numpy", "cv2", "yaml", "PIL", "torch", "torchvision"]),
    ("젯슨용 ONNX",      ["torch", "torchvision", "onnx", "onnxruntime"]),
    ("폐루프 시뮬",      ["numpy", "cv2", "yaml", "PIL", "torch", "gym", "f110_gym"]),
    ("주행 영상 재생",   ["imageio_ffmpeg"]),
    ("노트북 그래프",    ["matplotlib", "pandas"]),
]


NEED_VERSION = {"gym": "0.19.0"}             # f110_gym 이 옛 API 라 이 버전이어야 함


def version(name):
    if importlib.util.find_spec(name) is None:
        return None
    try:
        mod = __import__(name)
    except Exception as e:                       # 설치는 됐는데 import 가 깨지는 경우 (libGL 등)
        return f"!{type(e).__name__}"
    v = getattr(mod, "__version__", "?")
    if name in NEED_VERSION and v != NEED_VERSION[name]:
        return f"!{v} (need {NEED_VERSION[name]})"
    return v


def platform_name():
    try:
        with open("/proc/device-tree/model") as f:
            if "jetson" in f.read().lower():
                return "Jetson"
    except OSError:
        pass
    if importlib.util.find_spec("google.colab") is not None:
        return "Colab"
    return platform.system()


def main():
    print(f"플랫폼 {platform_name()}  |  Python {platform.python_version()}  |  {sys.executable}")
    print()
    have = {}
    for group, items in (("코어", CORE), ("시뮬", SIM), ("노트북", LAB)):
        for mod, pkg in items:
            v = version(mod)
            have[mod] = v is not None and not str(v).startswith("!")
            mark = "OK" if have[mod] else ("버전 불일치" if v and "need" in str(v) else "import 실패" if v else "없음")
            print(f"  {group:6s} {mod:16s} {str(v or '-'):12s} {mark}")
    print()
    print("가능한 것")
    missing_any = []
    for name, need in MODES:
        missing = [m for m in need if not have.get(m)]
        print(f"  {name:18s} {'O' if not missing else 'X  (' + ', '.join(missing) + ' 필요)'}")
        missing_any += missing
    if missing_any:
        print()
        print("Colab 설치:  notebooks/camsim_lab.ipynb 의 첫 코드 셀 실행")
        print("로컬 기본 의존성:  python -m pip install -r camsim/requirements.txt")
        print("PyTorch, torchvision, gym 0.19, f110_gym 은 Colab 노트북이 따로 확인·설치함")
    return 0


if __name__ == "__main__":
    sys.exit(main())
