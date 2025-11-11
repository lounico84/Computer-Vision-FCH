from ultralytics import YOLO

model = YOLO('yolo_training/models/fifth_model/run1/weights/best.pt')

results = model.predict('input_videos_match/Test/wiesendangen_test_clip_short.mp4', save=True, stream=True, device='mps')
#print(results[0])
#print('========================================')
#for box in results[0].boxes:
#    print(box)

for i, r in enumerate(results):
    print(f"Frame {i}")
    print("========================================")
