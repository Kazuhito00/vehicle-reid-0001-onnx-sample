# -*- coding: utf-8 -*-
import copy

import cv2
import numpy as np
import onnxruntime


class VehicleReID0001(object):

    def __init__(
        self,
        model_path='osnet_ain_x1_0_vehicle_reid.onnx',
        input_shape=(208, 208),
        score_th=0.5,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    ):
        # 閾値
        self.score_th = score_th

        # モデル読み込み
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            providers=providers,
        )

        self.input_detail = self.onnx_session.get_inputs()[0]
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_name = self.onnx_session.get_outputs()[0].name

        # 入力サイズ
        self.input_shape = input_shape

        # 特徴ベクトルリスト
        self.feature_vectors = None

    def __call__(self, image, bboxes, scores, class_ids, trim_offset=0):
        image_height, image_width = image.shape[0], image.shape[1]

        tracker_ids = []
        tracker_bboxes = []
        tracker_class_ids = []
        tracker_scores = []

        for bbox, class_id in zip(bboxes, class_ids):
            # 切り抜き
            xmin = int(np.clip(bbox[0] - trim_offset, 0, image_width - 1))
            ymin = int(np.clip(bbox[1] - trim_offset, 0, image_height - 1))
            xmax = int(np.clip(bbox[2] + trim_offset, 0, image_width - 1))
            ymax = int(np.clip(bbox[3] + trim_offset, 0, image_height - 1))
            vehicle_image = copy.deepcopy(image[ymin:ymax, xmin:xmax])

            # 前処理
            input_image = cv2.resize(
                vehicle_image,
                dsize=(self.input_shape[1], self.input_shape[0]),
            )
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            # mean = [0.485, 0.456, 0.406]
            # std = [0.229, 0.224, 0.225]
            # input_image = (input_image / 255 - mean) / std
            input_image = input_image.transpose(2, 0, 1)
            input_image = input_image.astype('float32')
            input_image = np.expand_dims(input_image, axis=0)

            # 推論実施
            result = self.onnx_session.run(
                None,
                {self.input_name: input_image},
            )
            result = np.array(result[0][0])

            # 初回推論時のデータ登録
            if self.feature_vectors is None:
                self.feature_vectors = copy.deepcopy(np.array([result]))

            # COS類似度計算
            cos_results = self._cos_similarity(result, self.feature_vectors)
            max_index = np.argmax(cos_results)
            max_value = cos_results[max_index]

            if max_value < self.score_th:
                # スコア閾値以下であれば特徴ベクトルリストに追加
                self.feature_vectors = np.vstack([
                    self.feature_vectors,
                    result,
                ])
            else:
                # スコア閾値以上であればトラッキング情報を追加
                tracker_ids.append(max_index)
                tracker_bboxes.append([xmin, ymin, xmax, ymax])
                tracker_class_ids.append(class_id)
                tracker_scores.append(max_value)

        return tracker_ids, tracker_bboxes, tracker_scores, tracker_class_ids

    def _cos_similarity(self, X, Y):
        Y = Y.T

        # (512,) x (n, 512) = (n,)
        result = np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y, axis=0))

        return result
