# SkatingApp
## Download checkpoints
### JumpDetection
* Current model: **[STGCNEncoderCRF](https://drive.google.com/drive/folders/1p0xlACx-v8-FU2lI7BEnOnuXnfRpD6F2?usp=share_link)**
    * Place `stgcn_encoder_crf` to `SkatingApp/engine/JumpDetection/checkpoint`

### VideoAlignment
* Current model: **[Checkpoint](https://drive.google.com/drive/folders/1-7-jx8yndf4aEDo7_X1aA7bc9MeYHI5l?usp=share_link)**
    * Place `flip` to `SkatingApp/engine/VideoAlignment/checkpoint`

* Current pivot video: **[Standard](https://drive.google.com/drive/folders/1-KvCidNhVKVLOk_ZP0pxohBk-kkWDNLN?usp=share_link)**
    * Place `alpha_pose_standard` to `SkatingApp/cache/align/data/flip/standard`

## Run
```
python wsgi.py
```
### APIs
* VideoAlignment
    * `/align?id1={}&id2={}`
* JumpDetection
    * `/jump?id={}`
### ngrok
```
./ngrok 5000
```