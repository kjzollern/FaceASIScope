import argparse
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import blobconverter
import cv2
import depthai as dai
from depthai_nodes import ImgDetectionsExtended
from depthai_nodes.node import ImgDetectionsBridge, ParsingNeuralNetwork
from depthai_nodes.node.utils import generate_script_content

REQ_WIDTH, REQ_HEIGHT = (1024, 768)
FPS = 14
FACEMESH_INPUT_SIZE = 192

LM = {
    "EYE_L": 263,
    "EYE_R": 33,
    "ALA_L": 98,
    "ALA_R": 327,
    "MOUTH_L": 61,
    "MOUTH_R": 291,
    "ZYGOMA_L": 234,
    "ZYGOMA_R": 454,
    "GLABELLA": 10,
    "MENTON": 152,
}

COLORS_BGR = [
    (80, 80, 255),
    (255, 160, 40),
    (255, 60, 60),
    (80, 255, 80),
    (255, 100, 200),
]

FONT = cv2.FONT_HERSHEY_SIMPLEX
RESULTS_DIR = Path(__file__).parent / "captures"
RESULTS_DIR.mkdir(exist_ok=True)


class Toggles:
    def __init__(self) -> None:
        self.mode = 1  # 1=reference-line, 2=pair-line
        self.show_hline = True
        self.show_landmarks = True


TOG = Toggles()


def _parse_facemesh_points(msg: dai.NNData, xmin: float, ymin: float, xmax: float, ymax: float) -> List[Tuple[float, float]]:
    tensor = msg.getFirstTensor(dequantize=True)
    if tensor is None:
        return []

    flat = tensor.flatten().tolist()
    if len(flat) < 2:
        return []

    if len(flat) >= 468 * 3:
        flat = flat[: 468 * 3]
        stride = 3
    else:
        usable = len(flat) - (len(flat) % 2)
        flat = flat[:usable]
        stride = 2

    xs = flat[0::stride]
    ys = flat[1::stride]
    max_val = max(max(xs, default=0.0), max(ys, default=0.0))
    scale = FACEMESH_INPUT_SIZE if max_val > 2.0 else 1.0

    width = max(xmax - xmin, 1e-6)
    height = max(ymax - ymin, 1e-6)

    points: List[Tuple[float, float]] = []
    for raw_x, raw_y in zip(xs, ys):
        norm_x = float(raw_x) / scale
        norm_y = float(raw_y) / scale
        points.append((xmin + norm_x * width, ymin + norm_y * height))
    return points


def _compute_axioms_metrics(points: Sequence[Tuple[float, float]]) -> Dict | None:
    if not points or max(LM.values()) >= len(points):
        return None

    def p(name: str) -> Tuple[float, float]:
        return points[LM[name]]

    eye_l = p("EYE_L")
    eye_r = p("EYE_R")
    xmid = (eye_l[0] + eye_r[0]) / 2.0

    pairs = [
        ("EYE_L", "EYE_R"),
        ("ALA_L", "ALA_R"),
        ("MOUTH_L", "MOUTH_R"),
        ("ZYGOMA_L", "ZYGOMA_R"),
    ]

    mids: List[float] = []
    ais: List[float] = []
    dists: List[Tuple[float, float, Tuple[float, float], Tuple[float, float]]] = []
    for left_name, right_name in pairs:
        p_left = p(left_name)
        p_right = p(right_name)
        d_left = abs(p_left[0] - xmid)
        d_right = abs(p_right[0] - xmid)
        ais.append(abs((d_right - d_left) / (d_right + d_left + 1e-6)))
        mids.append(abs(p_right[0] - p_left[0]) / 2.0)
        dists.append((d_left, d_right, p_left, p_right))

    ai_overall = float(sum(ais) / max(len(ais), 1))
    fa_hor = float(sum(abs(mids[i] - mids[j]) for i in range(len(mids)) for j in range(i)))

    mouth_l = p("MOUTH_L")
    mouth_r = p("MOUTH_R")
    inc1 = (eye_l[1] - eye_r[1]) / (eye_l[0] - eye_r[0] + 1e-6)
    inc2 = (mouth_l[1] - mouth_r[1]) / (mouth_l[0] - mouth_r[0] + 1e-6)
    angle = abs(math.atan(abs((inc1 - inc2) / (1 + inc1 * inc2 + 1e-6))))

    verdict = "balanced" if ai_overall < 0.07 else "asymmetry detected"
    return {
        "AI_overall": ai_overall,
        "FA_hor": fa_hor,
        "Angle": angle,
        "Verdict": verdict,
        "Pairs": dists,
        "xmid": xmid,
    }


def _clamp_point(x: float, y: float, w: int, h: int) -> Tuple[int, int]:
    return (max(0, min(int(x), w - 1)), max(0, min(int(y), h - 1)))


def _draw_reference_lines(frame, metrics, points, color):
    h, w = frame.shape[:2]
    xmid = max(0, min(int(metrics["xmid"]), w - 1))
    cv2.line(frame, (xmid, 0), (xmid, h - 1), (180, 180, 180), 1)

    if TOG.show_hline:
        eye_l = points[LM["EYE_L"]]
        eye_r = points[LM["EYE_R"]]
        eye_y = max(0, min(int((eye_l[1] + eye_r[1]) / 2.0), h - 1))
        cv2.line(frame, (0, eye_y), (w - 1, eye_y), (150, 150, 150), 1)

    for d_left, d_right, p_left, p_right in metrics["Pairs"]:
        l = _clamp_point(p_left[0], p_left[1], w, h)
        r = _clamp_point(p_right[0], p_right[1], w, h)
        cv2.line(frame, l, (xmid, l[1]), color, 1)
        cv2.line(frame, (xmid, r[1]), r, color, 1)
        cv2.putText(frame, f"dL={d_left:.1f}", (l[0] + 5, max(0, l[1] - 5)), FONT, 0.4, color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"dR={d_right:.1f}", (max(0, r[0] - 60), max(0, r[1] - 5)), FONT, 0.4, color, 1, cv2.LINE_AA)


def _draw_pair_lines(frame, metrics, color):
    for _d_left, _d_right, p_left, p_right in metrics["Pairs"]:
        l = (int(p_left[0]), int(p_left[1]))
        r = (int(p_right[0]), int(p_right[1]))
        mid = ((l[0] + r[0]) // 2, (l[1] + r[1]) // 2)
        cv2.line(frame, l, r, color, 1)
        cv2.circle(frame, mid, 2, (255, 255, 255), -1)
        cv2.putText(frame, "m", (mid[0] + 3, max(0, mid[1] - 3)), FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Angle={math.degrees(metrics['Angle']):.2f} deg", (10, 20), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_summary_panel(frame, metrics, face_id, color):
    h, w = frame.shape[:2]
    panel_w = 250
    x0 = max(5, w - panel_w - 5)
    y0 = 5

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + 118), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + 118), color, 1)

    lines = [
        f"Face #{face_id + 1}",
        f"AI={metrics['AI_overall']:.3f}",
        f"FA={metrics['FA_hor']:.2f}",
        f"Angle={math.degrees(metrics['Angle']):.2f} deg",
        f"Verdict: {metrics['Verdict']}",
        "M=Mode L=Landmarks H=HLine",
    ]

    y = y0 + 16
    for text in lines:
        if "balanced" in text:
            txt_color = (0, 255, 0)
        elif "asymmetry" in text:
            txt_color = (80, 80, 255)
        else:
            txt_color = (255, 255, 255)
        cv2.putText(frame, text, (x0 + 8, y), FONT, 0.45, txt_color, 1, cv2.LINE_AA)
        y += 16


def _add_watermark(frame):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"faceme OAK Axioms  {ts}", (10, frame.shape[0] - 10), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def _render_frame(frame, detections: ImgDetectionsExtended, landmarks: List[dai.NNData]):
    if not detections or not hasattr(detections, "detections"):
        cv2.putText(frame, "No detections message", (20, 40), FONT, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return frame

    if len(detections.detections) == 0:
        cv2.putText(frame, "No face detected", (20, 40), FONT, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        _add_watermark(frame)
        return frame

    for face_id, (det, lm_msg) in enumerate(zip(detections.detections, landmarks)):
        color = COLORS_BGR[face_id % len(COLORS)]
        xmin, ymin, xmax, ymax = det.rotated_rect.getOuterRect()

        p0 = _clamp_point(xmin, ymin, frame.shape[1], frame.shape[0])
        p1 = _clamp_point(xmax, ymax, frame.shape[1], frame.shape[0])
        cv2.rectangle(frame, p0, p1, color, 1)

        points = _parse_facemesh_points(lm_msg, xmin, ymin, xmax, ymax)
        if not points:
            continue

        if TOG.show_landmarks:
            for x, y in points:
                px, py = _clamp_point(x, y, frame.shape[1], frame.shape[0])
                cv2.circle(frame, (px, py), 1, (255, 220, 120), -1)
            for idx in LM.values():
                if idx < len(points):
                    px, py = _clamp_point(points[idx][0], points[idx][1], frame.shape[1], frame.shape[0])
                    cv2.circle(frame, (px, py), 2, color, -1)

        metrics = _compute_axioms_metrics(points)
        if metrics is None:
            continue

        if TOG.mode == 1:
            _draw_reference_lines(frame, metrics, points, color)
        else:
            _draw_pair_lines(frame, metrics, color)

        _draw_summary_panel(frame, metrics, face_id, color)

    mode_text = "Mode: Reference Lines" if TOG.mode == 1 else "Mode: Pair Lines"
    cv2.putText(frame, mode_text, (10, 32), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    _add_watermark(frame)
    return frame


def _build_inference_graph(pipeline: dai.Pipeline, platform: str):
    app_dir = Path(__file__).parent

    det_yaml = app_dir / "depthai_models" / f"yunet.{platform}.yaml"
    det_desc = dai.NNModelDescription.fromYamlFile(str(det_yaml))
    det_archive = dai.NNArchive(dai.getModelFromZoo(det_desc))
    det_w, det_h = det_archive.getInputSize()

    landmark_blob = blobconverter.from_zoo(
        name="facemesh_192x192",
        zoo_type="depthai",
        shaves=2,
    )
    print(f"Using FaceMesh blob: {landmark_blob}")

    frame_type = dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p

    cam = pipeline.create(dai.node.Camera).build()
    cam_out = cam.requestOutput(size=(REQ_WIDTH, REQ_HEIGHT), type=frame_type, fps=FPS)

    resize = pipeline.create(dai.node.ImageManip)
    resize.initialConfig.setOutputSize(det_w, det_h)
    resize.initialConfig.setReusePreviousImage(False)
    resize.inputImage.setBlocking(True)
    cam_out.link(resize.inputImage)

    det_nn = pipeline.create(ParsingNeuralNetwork).build(resize.out, det_archive)
    det_nn.getParser(0).conf_threshold = 0.9
    det_bridge = pipeline.create(ImgDetectionsBridge).build(det_nn.out)

    script_node = pipeline.create(dai.node.Script)
    det_nn.out.link(script_node.inputs["det_in"])
    cam_out.link(script_node.inputs["preview"])
    script_node.setScript(
        generate_script_content(
            resize_width=FACEMESH_INPUT_SIZE,
            resize_height=FACEMESH_INPUT_SIZE,
        )
    )

    crop_node = pipeline.create(dai.node.ImageManip)
    crop_node.initialConfig.setOutputSize(FACEMESH_INPUT_SIZE, FACEMESH_INPUT_SIZE)
    crop_node.inputConfig.setWaitForMessage(True)
    script_node.outputs["manip_cfg"].link(crop_node.inputConfig)
    script_node.outputs["manip_img"].link(crop_node.inputImage)

    landmark_nn = pipeline.create(dai.node.NeuralNetwork)
    landmark_nn.setBlobPath(landmark_blob)
    crop_node.out.link(landmark_nn.input)
    landmark_nn.input.setBlocking(True)

    return cam_out, det_nn, det_bridge, landmark_nn


def run_browser(http_port: int):
    visualizer = dai.RemoteConnection(httpPort=http_port)
    device = dai.Device()

    with dai.Pipeline(device) as pipeline:
        platform = device.getPlatform().name
        print(f"Platform: {platform}")

        cam_out, det_nn, _det_bridge, landmark_nn = _build_inference_graph(pipeline, platform)

        visualizer.addTopic("Video", cam_out, "images")
        visualizer.addTopic("Detections", det_nn.out, "images")
        visualizer.addTopic("FaceMeshRaw", landmark_nn.out, "images")

        pipeline.start()
        visualizer.registerPipeline(pipeline)

        while pipeline.isRunning():
            key = visualizer.waitKey(1)
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                TOG.mode = 2 if TOG.mode == 1 else 1
            if key == ord("l"):
                TOG.show_landmarks = not TOG.show_landmarks
            if key == ord("h"):
                TOG.show_hline = not TOG.show_hline


def run_opencv():
    device = dai.Device()

    with dai.Pipeline(device) as pipeline:
        platform = device.getPlatform().name
        print(f"Platform: {platform}")

        cam_out, _det_nn, det_bridge, landmark_nn = _build_inference_graph(pipeline, platform)

        video_out = pipeline.create(dai.node.XLinkOut)
        video_out.setStreamName("video")
        cam_out.link(video_out.input)

        det_out = pipeline.create(dai.node.XLinkOut)
        det_out.setStreamName("detections")
        det_bridge.out.link(det_out.input)

        lm_out = pipeline.create(dai.node.XLinkOut)
        lm_out.setStreamName("landmarks")
        landmark_nn.out.link(lm_out.input)

        pipeline.start()

        q_video = device.getOutputQueue("video", maxSize=8, blocking=False)
        q_det = device.getOutputQueue("detections", maxSize=8, blocking=False)
        q_lm = device.getOutputQueue("landmarks", maxSize=32, blocking=False)

        state = defaultdict(lambda: {"frame": None, "det": None, "lm": []})

        window = "faceme_cv (ESC=exit, M=mode, L=landmarks, H=hline, SPACE=capture)"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        last_frame = None

        while True:
            msg = q_video.tryGet()
            while msg is not None:
                seq = int(msg.getSequenceNum())
                state[seq]["frame"] = msg.getCvFrame()
                msg = q_video.tryGet()

            dmsg = q_det.tryGet()
            while dmsg is not None:
                seq = int(dmsg.getSequenceNum())
                state[seq]["det"] = dmsg
                dmsg = q_det.tryGet()

            lmsg = q_lm.tryGet()
            while lmsg is not None:
                seq = int(lmsg.getSequenceNum())
                state[seq]["lm"].append(lmsg)
                lmsg = q_lm.tryGet()

            ready_seqs = sorted(state.keys())
            for seq in ready_seqs:
                entry = state[seq]
                frame = entry["frame"]
                det = entry["det"]
                lms = entry["lm"]
                if frame is None or det is None:
                    continue

                expected = len(getattr(det, "detections", []))
                if len(lms) < expected:
                    continue

                frame_bgr = frame.copy()
                render = _render_frame(frame_bgr, det, lms[:expected])
                cv2.imshow(window, render)
                last_frame = render
                del state[seq]

            if len(state) > 200:
                min_keep = sorted(state.keys())[-80:]
                state = defaultdict(lambda: {"frame": None, "det": None, "lm": []}, {k: state[k] for k in min_keep})

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key == ord("m"):
                TOG.mode = 2 if TOG.mode == 1 else 1
            if key == ord("l"):
                TOG.show_landmarks = not TOG.show_landmarks
            if key == ord("h"):
                TOG.show_hline = not TOG.show_hline
            if key == 32 and last_frame is not None:  # SPACE
                ts = time.strftime("%Y%m%d_%H%M%S")
                out = RESULTS_DIR / f"faceme_cv_{ts}.png"
                cv2.imwrite(str(out), last_frame)
                print(f"Captured: {out}")

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="OAK FaceMesh + Axioms (OpenCV or browser)")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in browser mode (no local OpenCV window).",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=8082,
        help="HTTP port for browser mode (used with --headless).",
    )
    args = parser.parse_args()

    if args.headless:
        print("Headless/browser mode enabled.")
        print(f"Open http://localhost:{args.http_port} in your browser.")
        run_browser(args.http_port)
    else:
        run_opencv()


if __name__ == "__main__":
    main()
