from typing import List

import depthai as dai
from depthai_nodes import ImgDetectionsExtended, SECONDARY_COLOR
from depthai_nodes.utils import AnnotationHelper


class AnnotationNode(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(self, gather_data_msg: dai.Node.Output) -> "AnnotationNode":
        self.link_args(gather_data_msg)
        return self

    def process(self, gather_data_msg: dai.Buffer) -> None:
        detections_msg: ImgDetectionsExtended = gather_data_msg.reference_data
        assert isinstance(detections_msg, ImgDetectionsExtended)

        facemesh_msg_list: List[dai.NNData] = gather_data_msg.gathered
        assert isinstance(facemesh_msg_list, list)
        assert all(isinstance(msg, dai.NNData) for msg in facemesh_msg_list)
        assert len(detections_msg.detections) == len(facemesh_msg_list)

        annotations = AnnotationHelper()

        for detection, facemesh_msg in zip(detections_msg.detections, facemesh_msg_list):
            xmin, ymin, xmax, ymax = detection.rotated_rect.getOuterRect()
            annotations.draw_rectangle((xmin, ymin), (xmax, ymax))

            tensor = facemesh_msg.getFirstTensor(dequantize=True)
            if tensor is None:
                continue

            flat = tensor.flatten().tolist()
            if len(flat) < 2:
                continue

            if len(flat) >= 468 * 3:
                flat = flat[: 468 * 3]
                stride = 3
            else:
                usable = len(flat) - (len(flat) % 2)
                flat = flat[:usable]
                stride = 2

            xs = flat[0::stride]
            ys = flat[1::stride]
            scale = 192.0 if max(max(xs, default=0.0), max(ys, default=0.0)) > 2.0 else 1.0
            width = max(xmax - xmin, 1e-6)
            height = max(ymax - ymin, 1e-6)

            points = []
            for raw_x, raw_y in zip(xs, ys):
                norm_x = float(raw_x) / scale
                norm_y = float(raw_y) / scale
                points.append((xmin + norm_x * width, ymin + norm_y * height))

            annotations.draw_points(points=points, color=SECONDARY_COLOR, thickness=1.0)

        annotations_msg = annotations.build(
            timestamp=detections_msg.getTimestamp(),
            sequence_num=detections_msg.getSequenceNum(),
        )

        self.out.send(annotations_msg)
