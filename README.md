# vehicle-reid-0001-onnx-sample
[vehicle-reid-0001](https://docs.openvino.ai/2023.2/omz_models_model_vehicle_reid_0001.html) を用いて、Vehicle ReIDを行うサンプルです。<Br>
モデルは同梱していますが、ダウンロードと最適化も試したい方は[vehicle-reid-0001-onnx-optimize.ipynb](vehicle-reid-0001-onnx-optimize.ipynb)をGoogle Colaboratoryなどで動かしてみてください。<br>

https://github.com/Kazuhito00/vehicle-reid-0001-onnx-sample/assets/37477845/2965b922-4cd5-46ad-9858-c26b1947186e

# Requirement 
* OpenCV 4.8.1.78 or later
* onnxruntime 1.16.3 or later

# Demo
```bash
python sample.py
```
* --movie_01<br>
視点1の動画<br>
デフォルト：assets/01.mp4
* --movie_02<br>
視点2の動画<br>
デフォルト：assets/02.mp4
* --yolox_model<br>
YOLOX（物体検出）のモデルパス<br>
デフォルト：yolox/model/yolox_tiny.onnx
* --yolox_score_th<br>
YOLOX（物体検出）のスコア閾値<br>
デフォルト：0.3
* --reid_model<br>
Vehicle ReIDのモデルパス<br>
デフォルト：vehicle_reid_0001/model/osnet_ain_x1_0_vehicle_reid_optimized.onnx
* --yolox_score_th<br>
Vehicle ReIDのスコア閾値<br>
デフォルト：0.5

# Reference
* [vehicle-reid-0001](https://docs.openvino.ai/2023.2/omz_models_model_vehicle_reid_0001.html)

# License 
vehicle-reid-0001-onnx-sampler is under [Apache-2.0 license](LICENSE).

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
