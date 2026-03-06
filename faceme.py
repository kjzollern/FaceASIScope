import math
import time
from pathlib import Path
from typing import Iterable, List, Sequence

import blobconverter
import depthai as dai
from depthai_nodes import ImgDetectionsExtended
from depthai_nodes.node import GatherData, ImgDetectionsBridge, ParsingNeuralNetwork
from depthai_nodes.node.utils import generate_script_content
from depthai_nodes.utils import AnnotationHelper

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

COLORS = [
    dai.Color(1.0, 0.31, 0.31),
    dai.Color(0.16, 0.63, 1.0),
    dai.Color(0.24, 0.24, 1.0),
    dai.Color(0.31, 1.0, 0.31),
    dai.Color(0.78, 0.39, 1.0),
]


class Toggles:
    def __init__(self) -> None:
        self.mode = 1  # 1=reference-line, 2=pair-line
        self.show_hline = True
        self.show_landmarks = True


TOG = Toggles()


def _parse_facemesh_points(msg: dai.NNData, xmin: float, ymin: float, xmax: float, ymax: float) -> List[tuple[float, float]]:
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
    scale = FACEMESH_INPUT_SIZE if max(max(xs, default=0.0), max(ys, default=0.0)) > 2.0 else 1.0
    width = max(xmax - xmin, 1e-6)
    height = max(ymax - ymin, 1e-6)

    points: List[tuple[float, float]] = []
    for raw_x, raw_y in zip(xs, ys):
        norm_x = float(raw_x) / scale
        norm_y = float(raw_y) / scale
        points.append((xmin + norm_x * width, ymin + norm_y * height))
    return points


def _compute_axioms_metrics(points: Sequence[tuple[float, float]]) -> dict | None:
    if not points or max(LM.values()) >= len(points):
        return None

    def p(name: str) -> tuple[float, float]:
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
    dists: List[tuple[float, float, tuple[float, float], tuple[float, float]]] = []
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


class AxiomsAnnotationNode(dai.node.HostNode):
    def build(self, gather_data_msg: dai.Node.Output):
        self.link_args(gather_data_msg)
        return self

    def _draw_reference_lines(
        self,
        annotations: AnnotationHelper,
        metrics: dict,
        points: Sequence[tuple[float, float]],
        color: dai.Color,
        frame_width: float,
        frame_height: float,
    ) -> None:
        xmid = metrics["xmid"]
        annotations.draw_line((xmid, 0), (xmid, frame_height), color=(180, 180, 180, 255), thickness=1.0)

        if TOG.show_hline:
            eye_l = points[LM["EYE_L"]]
            eye_r = points[LM["EYE_R"]]
            eye_y = max(0.0, min(frame_height - 1.0, (eye_l[1] + eye_r[1]) / 2.0))
            annotations.draw_line((0, eye_y), (frame_width, eye_y), color=(150, 150, 150, 255), thickness=1.0)

        for d_left, d_right, p_left, p_right in metrics["Pairs"]:
            annotations.draw_line((p_left[0], p_left[1]), (xmid, p_left[1]), color=color, thickness=1.0)
            annotations.draw_line((xmid, p_right[1]), (p_right[0], p_right[1]), color=color, thickness=1.0)
            annotations.draw_text(f"dL={d_left:.1f}", (p_left[0] + 5, max(0.0, p_left[1] - 8)), color=color, size=18)
            annotations.draw_text(f"dR={d_right:.1f}", (max(0.0, p_right[0] - 58), max(0.0, p_right[1] - 8)), color=color, size=18)

    def _draw_pair_lines(
        self,
        annotations: AnnotationHelper,
        metrics: dict,
        color: dai.Color,
    ) -> None:
        for _d_left, _d_right, p_left, p_right in metrics["Pairs"]:
            mid = ((p_left[0] + p_right[0]) / 2.0, (p_left[1] + p_right[1]) / 2.0)
            annotations.draw_line(p_left, p_right, color=color, thickness=1.0)
            annotations.draw_circle(mid, radius=2.5, outline_color=(255, 255, 255, 255), fill_color=(255, 255, 255, 255), thickness=1.0)
            annotations.draw_text("m", (mid[0] + 3, max(0.0, mid[1] - 6)), color=(255, 255, 255, 255), size=18)
        annotations.draw_text(
            f"Angle={math.degrees(metrics['Angle']):.2f} deg",
            (10, 20),
            color=(255, 255, 255, 255),
            size=22,
        )

    def _draw_summary_panel(self, annotations: AnnotationHelper, metrics: dict, face_id: int, color: dai.Color, frame_width: float) -> None:
        panel_width = 250.0
        x0 = max(5.0, frame_width - panel_width - 5.0)
        y0 = 5.0
        annotations.draw_rectangle(
            (x0, y0),
            (x0 + panel_width, y0 + 118),
            outline_color=color,
            fill_color=(25, 25, 25, 150),
            thickness=1.0,
        )
        lines = [
            f"Face #{face_id + 1}",
            f"AI={metrics['AI_overall']:.3f}",
            f"FA={metrics['FA_hor']:.2f}",
            f"Angle={math.degrees(metrics['Angle']):.2f} deg",
            f"Verdict: {metrics['Verdict']}",
            "OAK FaceMesh + Axioms",
        ]
        y = y0 + 16.0
        for text in lines:
            if "balanced" in text:
                txt_color = (0, 255, 0, 255)
            elif "asymmetry" in text:
                txt_color = (255, 80, 80, 255)
            else:
                txt_color = (255, 255, 255, 255)
            annotations.draw_text(text, (x0 + 8.0, y), color=txt_color, size=18)
            y += 16.0

    def _add_watermark(self, annotations: AnnotationHelper, frame_height: float) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        annotations.draw_text(f"faceme OAK Axioms  {ts}", (10, frame_height - 12), color=(255, 255, 255, 255), size=16)

    def process(self, gather_data_msg: dai.Buffer) -> None:
        detections_msg: ImgDetectionsExtended = gather_data_msg.reference_data
        assert isinstance(detections_msg, ImgDetectionsExtended)

        facemesh_msg_list: List[dai.NNData] = gather_data_msg.gathered
        assert isinstance(facemesh_msg_list, list)
        assert all(isinstance(msg, dai.NNData) for msg in facemesh_msg_list)
        assert len(detections_msg.detections) == len(facemesh_msg_list)

        annotations = AnnotationHelper()
        frame_width = float(getattr(detections_msg, "source_width", REQ_WIDTH))
        frame_height = float(getattr(detections_msg, "source_height", REQ_HEIGHT))

        for face_id, (detection, facemesh_msg) in enumerate(zip(detections_msg.detections, facemesh_msg_list)):
            color = COLORS[face_id % len(COLORS)]
            xmin, ymin, xmax, ymax = detection.rotated_rect.getOuterRect()
            points = _parse_facemesh_points(facemesh_msg, xmin, ymin, xmax, ymax)
            if not points:
                continue

            metrics = _compute_axioms_metrics(points)
            if metrics is None:
                continue

            annotations.draw_rectangle((xmin, ymin), (xmax, ymax), outline_color=color, thickness=1.0)
            if TOG.show_landmarks:
                annotations.draw_points(points=points, color=(120, 220, 255, 255), thickness=1.0)
                subset = [points[idx] for idx in LM.values() if idx < len(points)]
                annotations.draw_points(points=subset, color=color, thickness=3.0)

            if TOG.mode == 1:
                self._draw_reference_lines(annotations, metrics, points, color, frame_width, frame_height)
            else:
                self._draw_pair_lines(annotations, metrics, color)

            self._draw_summary_panel(annotations, metrics, face_id, color, frame_width)
            mode_text = "Mode: Reference Lines" if TOG.mode == 1 else "Mode: Pair Lines"
            annotations.draw_text(mode_text, (10, 34), color=(255, 255, 255, 255), size=22)

        self._add_watermark(annotations, frame_height)
        annotations_msg = annotations.build(
            timestamp=detections_msg.getTimestamp(),
            sequence_num=detections_msg.getSequenceNum(),
        )
        self.out.send(annotations_msg)


visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device()

with dai.Pipeline(device) as pipeline:
    platform = device.getPlatform().name
    app_dir = Path(__file__).parent
    print(f"Platform: {platform}")

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

    gather_data_node = pipeline.create(GatherData).build(
        FPS,
        landmark_nn.out,
        det_bridge.out,
    )
    axioms_node = pipeline.create(AxiomsAnnotationNode).build(gather_data_node.out)

    visualizer.addTopic("Video", cam_out, "images")
    visualizer.addTopic("Detections", det_nn.out, "images")
    visualizer.addTopic("Axioms", axioms_node.out, "images")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            break
        if key == ord("m"):
            TOG.mode = 2 if TOG.mode == 1 else 1
        if key == ord("l"):
            TOG.show_landmarks = not TOG.show_landmarks
        if key == ord("h"):
            TOG.show_hline = not TOG.show_hline
