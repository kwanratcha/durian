import ultralytics
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.font_manager as fm

def Check_Probability_Threshold(threshold):
    for probability in probs:
        if probability > (threshold/100):
            return True
    return False



if __name__ == '__main__':
    ultralytics.checks()
    Path = 'D:/final_project/use_to_train/train_aug'
    #train model
    model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
    print (model)
    #results = model.train(
    #     data=Path,
    #     epochs=30,
    #     imgsz=640,
    #     lr0=0.00001,
    #     dropout=0.1,
    #     cls=0.5)

    #plot loss
    results_path = 'D:/final_project/final report/code_1/runs/classify/train3/results.csv'
    results = pd.read_csv(results_path)
    plt.figure()
    plt.plot(results['                  epoch'], results['             train/loss'], label='train loss')
    plt.plot(results['                  epoch'], results['               val/loss'], label='val loss', c='red')
    plt.grid()
    plt.title('Loss vs epochs')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()
    #%matplotlib inline
    plt.figure()
    plt.plot(results['                  epoch'], results['  metrics/accuracy_top1'] * 100)
    plt.grid()
    plt.title('Validation accuracy vs epochs')
    plt.ylabel('accuracy (%)')
    plt.xlabel('epochs')
    plt.show()

    model = YOLO('D:/final_project/final report/code_1/runs/classify/train3/weights/last.pt')  # load a custom model
    # ให้โมเดลทำนาย
    Image_Predict = 'D:/final_project/final report/code_1/test/thongT(154).jpg'
    results = model(Image_Predict)  # predict on an image
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()
    print(names_dict)
    print(probs)
    print()
    print()
    print()
    print()
    print()
    print("Predict Result:", names_dict[np.argmax(probs)])
    lable = {
        "chani": "ชะนี",
        "kanyao": "ก้านยาว",
        "longL": "หลงลับแล",
        "med": "เม็ดในยายปราง",
        "moanthong": "หมอนทอง",
        "nokyib": "นกหยิบ",
        "salika": "สาลิกา",
        "thongY": "ทองย้อยฉัตร"
    }

    font = 'D:/final_project/final report/code_1/TumthaiThin.ttf'
    thai_font = fm.FontProperties(fname=font)
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()
    predicted_index = np.argmax(probs)  # หาดัชนีของความน่าจะเป็นสูงสุด
    predicted_class = names_dict[predicted_index]  # ชื่อคลาสที่สอดคล้อง
    predicted_prob = probs[predicted_index]  # ความน่าจะเป็นสูงสุด
    if predicted_prob >= 0.70:
        #print(f"Predicted Class: {predicted_class}, Probability: {predicted_prob}")
        x, y = 10, 20
        img = Image.open(Image_Predict)  # ใช้ภาพต้นฉบับในการแสดง
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.annotate(f"{lable[predicted_class]}: {predicted_prob:.2f}", (10, 20), color='white', weight='bold',
                    fontsize=18,
                    ha='left', va='top', fontproperties=thai_font)
        plt.show()
    else:
        x, y = 10, 20
        img = Image.open(Image_Predict)  # ใช้ภาพต้นฉบับในการแสดง
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.annotate(f"Unknown", (x, y), color='red', weight='bold', fontsize=18,
                    ha='left', va='top')
        plt.show()
