#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2
import numpy as np

from yolox.yolox_onnx import YoloxONNX
from vehicle_reid_0001.vehicle_reid_0001 import VehicleReID0001


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--movie_01", type=str, default='assets/01.mp4')
    parser.add_argument("--movie_02", type=str, default='assets/02.mp4')

    parser.add_argument(
        "--yolox_model",
        type=str,
        default='yolox/model/yolox_tiny.onnx',
    )
    parser.add_argument(
        "--yolox_score_th",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--reid_model",
        type=str,
        default=
        'vehicle_reid_0001/model/osnet_ain_x1_0_vehicle_reid_optimized.onnx',
    )
    parser.add_argument(
        "--reid_score_th",
        type=float,
        default=0.5,
    )

    args = parser.parse_args()

    return args


def main():
    # 引数解析
    args = get_args()
    movie_01_path = args.movie_01
    movie_02_path = args.movie_02

    yolox_model_path = args.yolox_model
    yolox_score_th = args.yolox_score_th

    reid_model_path = args.reid_model
    reid_score_th = args.reid_score_th

    # 動画準備
    cap01 = cv2.VideoCapture(movie_01_path)
    cap02 = cv2.VideoCapture(movie_02_path)

    # モデルロード
    yolox = YoloxONNX(model_path=yolox_model_path, score_th=yolox_score_th)
    vehicle_reid = VehicleReID0001(reid_model_path, score_th=reid_score_th)

    # COCOクラスリスト読み込み
    with open('yolox/coco_classes.txt', 'rt') as f:
        coco_classes = f.read().rstrip('\n').split('\n')

    # 2:car 5:bus 7:truck
    target_id = [2, 5, 7]

    while True:
        # カメラキャプチャ
        ret, frame01 = cap01.read()
        if not ret:
            break
        ret, frame02 = cap02.read()
        if not ret:
            break
        debug_image01 = copy.deepcopy(frame01)
        debug_image02 = copy.deepcopy(frame02)

        # 物体検出 推論実施
        start_time01 = time.time()
        bboxes01, scores01, class_ids01 = yolox(frame01)
        elapsed_time01 = time.time() - start_time01
        start_time02 = time.time()
        bboxes02, scores02, class_ids02 = yolox(frame02)
        elapsed_time02 = time.time() - start_time02

        # 車系のIDのみでフィルタリング
        if len(class_ids01) > 0:
            target_index01 = np.in1d(class_ids01, np.array(target_id))
            bboxes01 = bboxes01[target_index01]
            scores01 = scores01[target_index01]
            class_ids01 = class_ids01[target_index01]

        if len(class_ids02) > 0:
            target_index02 = np.in1d(class_ids02, np.array(target_id))
            bboxes02 = bboxes02[target_index02]
            scores02 = scores02[target_index02]
            class_ids02 = class_ids02[target_index02]

        # Vehicle ReID
        start_time03 = time.time()
        tracker_result01 = vehicle_reid(
            frame01,
            bboxes01,
            scores01,
            class_ids01,
        )
        tracker_ids01 = tracker_result01[0]
        tracker_bboxes01 = tracker_result01[1]
        tracker_scores01 = tracker_result01[2]
        tracker_class_ids01 = tracker_result01[3]
        elapsed_time03 = time.time() - start_time03

        start_time04 = time.time()
        tracker_result02 = vehicle_reid(
            frame02,
            bboxes02,
            scores02,
            class_ids02,
        )
        tracker_ids02 = tracker_result02[0]
        tracker_bboxes02 = tracker_result02[1]
        tracker_scores02 = tracker_result02[2]
        tracker_class_ids02 = tracker_result02[3]
        elapsed_time04 = time.time() - start_time04

        # デバッグ描画
        debug_image01 = draw_debug(
            debug_image01,
            elapsed_time01,
            elapsed_time02,
            elapsed_time03,
            elapsed_time04,
            tracker_ids01,
            tracker_bboxes01,
            tracker_scores01,
            tracker_class_ids01,
            coco_classes,
        )
        debug_image02 = draw_debug(
            debug_image02,
            None,
            None,
            None,
            None,
            tracker_ids02,
            tracker_bboxes02,
            tracker_scores02,
            tracker_class_ids02,
            coco_classes,
        )

        # キー処理(ESC：終了)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映
        cv2.imshow('01', debug_image01)
        cv2.imshow('02', debug_image02)

    cap01.release()
    cap02.release()
    cv2.destroyAllWindows()


def get_id_color(index):
    temp_index = abs(int(index + 1)) * 3
    color = (
        (37 * temp_index) % 255,
        (17 * temp_index) % 255,
        (29 * temp_index) % 255,
    )
    return color


def draw_debug(
    image,
    elapsed_time01,
    elapsed_time02,
    elapsed_time03,
    elapsed_time04,
    tracker_ids,
    bboxes,
    scores,
    class_ids,
    coco_classes,
):
    debug_image = copy.deepcopy(image)

    for tracker_id, bbox, score, class_id in zip(
            tracker_ids,
            bboxes,
            scores,
            class_ids,
    ):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # バウンディングボックス
        debug_image = cv2.rectangle(
            debug_image,
            (x1, y1),
            (x2, y2),
            get_id_color(tracker_id),
            thickness=2,
        )

        # 追跡ID
        score = '%.2f' % score
        text = 'ID:%s(%s)' % (str(tracker_id), str(score))
        debug_image = cv2.putText(
            debug_image,
            text,
            (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            get_id_color(tracker_id),
            thickness=2,
        )
        # クラスID、スコア
        text = '%s' % (str(coco_classes[int(class_id)]))
        debug_image = cv2.putText(
            debug_image,
            text,
            (x1 + 2, y1 + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            get_id_color(tracker_id),
            thickness=2,
        )

    # 推論時間
    if elapsed_time01 is not None:
        text = 'YOLOX 01 : ' + '%.0f' % (elapsed_time01 * 1000)
        text = text + 'ms'
        debug_image = cv2.putText(
            debug_image,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            thickness=2,
        )
    if elapsed_time02 is not None:
        text = 'YOLOX 02 : ' + '%.0f' % (elapsed_time02 * 1000)
        text = text + 'ms'
        debug_image = cv2.putText(
            debug_image,
            text,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            thickness=2,
        )
    if elapsed_time03 is not None:
        text = 'VehicleReID 01 : ' + '%.0f' % (elapsed_time03 * 1000)
        text = text + 'ms'
        debug_image = cv2.putText(
            debug_image,
            text,
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            thickness=2,
        )
    if elapsed_time04 is not None:
        text = 'VehicleReID 02 : ' + '%.0f' % (elapsed_time04 * 1000)
        text = text + 'ms'
        debug_image = cv2.putText(
            debug_image,
            text,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            thickness=2,
        )

    return debug_image


if __name__ == '__main__':
    main()
