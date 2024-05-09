Potential training command:
~\.conda\envs\Pollen3\python.exe -u "e:\Coding\Pollen\yolov5\train.py" --img 640 --batch 16 --epochs 25 --data "E:/Coding/Pollen/datasets/SYNTH_POLEN23E_Manual_Filtered\_\_MIXED_BG_500_100/pollen_dataset.yaml" --weights "yolov5s.pt" --project "./runs/train" --name "2024_03_17_img-640_batch-25_epochs-50"

Infeerence Command:
~\.conda\envs\Pollen3\python.exe detect.py --weights "runs\train\exp7-AMP-False-Manual-Filtered3\weights\best.pt" --source "E:\Coding\Pollen\datasets\SYNTH_POLEN23E_Manual_Filtered_300_60\videos\val.avi"

Infeerence Command:
~\.conda\envs\Pollen3\python.exe detect.py --weights "runs\train\2024_03_17_img-640_batch-25_epochs-50\weights\best.pt" --source "E:\Coding\Pollen\datasets\SYNTH_POLEN23E_Manual_Filtered_300_100_filtered_FOR_INF\videos\val.avi"

Run inference demo
~\.conda\envs\CVMaster\python.exe -u "e:\Coding\Pollen\yolov5\counterV2.py"
