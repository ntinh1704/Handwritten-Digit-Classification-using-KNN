import numpy as np
import cv2
from matplotlib import pyplot as plt # type: ignore
import os

DATA_PATH_TRANNING = 'C:/Users/bravo/Downloads/KNN/dataset/training'
DATA_PATH_TESTING = 'C:/Users/bravo/Downloads/KNN/dataset/testing'
number_train = 0
number_test = 0
k = 0

def load_data(path, type): 
    labels = []
    images = []
    global number_train, number_test
    if type == "train":
        max_count = number_train
    else:
        max_count = number_test
    for foldername in os.listdir(path):
        label = int (foldername)
        folder_path = os.path.join(path, foldername)
        count = 0
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path, 0)
            images.append(image)
            labels.append(label)
            count+=1
            if count == max_count: 
                break
    return images, labels

def load_config():
    global number_train, number_test, k
    file =open("config.dat", 'r')
    number_train = int(file.readline())
    number_test = int(file.readline())
    k = int(file.readline())
    file.close()
    
def set_config(train, test, k):
    file = open("config.dat", "w")
    file.write(str(train)+"\n")
    file.write(str(test)+"\n")
    file.write(str(k)+"\n")
    file.close()
    print('\nThiết lập hoàn tất!!!\n')

def euclidean_distance(x1, x2):
    x1 = x1.reshape(-1, 784)
    x2 = x2.reshape(-1, 784)
    squared_dist = np.sum(np.square(x1 - x2))
    dist = np.sqrt(squared_dist)
    return dist

def get_k_nearest_neighbors(X_train, y_train, x_test, k):
    distances = []
    
    for i in range(len(X_train)):
        distance = euclidean_distance(x_test, X_train[i])
        distances.append((distance, y_train[i]))
    distances.sort()
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][1])
    return neighbors

def predict_label(X_train, y_train, x_test, k):
    neighbors = get_k_nearest_neighbors(X_train, y_train, x_test, k)
    label_counts = np.bincount(neighbors)
    return np.argmax(label_counts)

def predict():
    global k 
    file_name = input(f'Nhập tên file (lưu ý tên file phải có trong thư mục pred_input): ')
    file_path = 'pred_input/' + file_name
    image_file = cv2.imread(file_path, 0)
    image_file = handle_data(image_file)
    image_file = np.array(image_file)
    labels_train, images_train, labels_test, images_test = load_and_handle()
    pred = predict_label(images_train, labels_train, image_file, k)
    print('Kết quả dự đoán là: ' + str(pred))

def enhance_images(images):
    """
    Tiền xử lý ảnh bổ sung, bao gồm thay đổi kích thước, lọc nhiễu và cân bằng histogram.
    """
    processed_images = []
    for img in images:
        # Thay đổi kích thước ảnh về 28x28 (nếu cần)
        img = cv2.resize(img, (28, 28))
        
        # Loại bỏ nhiễu bằng Gaussian Blur
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Cân bằng histogram để cải thiện độ tương phản
        img = cv2.equalizeHist(img)
        
        processed_images.append(img)
    return processed_images

def handle_data(images):
    # Bước 1: Tiền xử lý ảnh bằng cách cải thiện chất lượng
    images = enhance_images(images)
    
    # Bước 2: Phân ngưỡng nhị phân và đảo ngược nếu cần
    new_imgs = []
    for img in images:
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(thresh) > 130: 
            thresh = cv2.bitwise_not(thresh)
        new_imgs.append(thresh)
    return new_imgs

def load_and_handle():
    imgs_train, lbs_train = load_data(DATA_PATH_TRANNING, 'train')
    imgs_train = handle_data(imgs_train)
    images_train = np.array(imgs_train)
    labels_train = np.array(lbs_train)
    imgs_test, lbs_test = load_data(DATA_PATH_TESTING, 'test')
    imgs_test = handle_data(imgs_test)
    images_test = np.array(imgs_test)
    labels_test = np.array(lbs_test)
    return labels_train, images_train, labels_test, images_test

def setNumberTrainAndTest():
    global number_train, number_test, k
    num_train = input(f'Nhập số lượng train (nhỏ hơn 5.500)')
    num_train = int(num_train)
    num_test = input(f'Nhập số lượng test (nhỏ hơn 1.000)')
    num_test = int(num_test)
    number_train = num_train
    number_test = num_test
    set_config(num_train, num_test, k)
    
def setK():
    global number_train, number_test, k
    new_k = input(f'Nhập K (nhỏ hơn K_max: 244)')
    new_k = int(new_k)
    k = new_k
    set_config(number_train, number_test, k)
    
def cal_accuracy():
    global number_train, number_test, k 
    n_correct = 0
    labels_train, images_train, labels_test, images_test = load_and_handle()
    for i in range(len(images_test)):
        pred = predict_label(images_train, labels_train, images_test[i], k)
        if pred == labels_test[i]:
            n_correct += 1
    accuracy = n_correct / len(images_test)
    print('Accuracy: %.2f%%' % (accuracy * 100))
    
def get_cal_accuracy():
    global number_train, number_test, k 
    n_correct = 0
    labels_train, images_train, labels_test, images_test = load_and_handle()
    for i in range(len(images_test)):
        pred = predict_label(images_train, labels_train, images_test[i], k)
        if pred == labels_test[i]:
            n_correct += 1
    accuracy = n_correct / len(images_test)
    return accuracy

def calculate_recall_and_f1():
    """
    Tính toán Recall và F1-Score cho từng lớp nhãn trong tập kiểm tra.
    """
    global k
    # Tải dữ liệu huấn luyện và kiểm tra
    labels_train, images_train, labels_test, images_test = load_and_handle()
    n_classes = 10  # Số lượng nhãn (0 đến 9)

    # Khởi tạo các biến đếm
    true_positives = [0] * n_classes
    false_negatives = [0] * n_classes
    false_positives = [0] * n_classes

    # Duyệt qua từng ảnh trong tập kiểm tra
    for i in range(len(images_test)):
        pred = predict_label(images_train, labels_train, images_test[i], k)
        actual = labels_test[i]

        if pred == actual:
            # Dự đoán đúng (True Positive)
            true_positives[actual] += 1
        else:
            # Dự đoán sai
            false_negatives[actual] += 1  # Sai thành lớp khác
            false_positives[pred] += 1  # Sai từ lớp khác thành lớp này

    # Tính Recall và F1-Score cho từng lớp
    recalls = []
    f1_scores = []
    for i in range(n_classes):
        TP = true_positives[i]
        FN = false_negatives[i]
        FP = false_positives[i]

        # Tính Recall
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        recalls.append(recall)

        # Tính Precision
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0

        # Tính F1-Score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    # Tính Recall và F1 trung bình (macro average)
    avg_recall = sum(recalls) / n_classes
    avg_f1 = sum(f1_scores) / n_classes

    # In kết quả
    print("\n--- Kết quả Recall và F1-Score ---")
    for i in range(n_classes):
        print(f"Lớp {i}: Recall = {recalls[i]:.2f}, F1-Score = {f1_scores[i]:.2f}")
    print(f"\nRecall trung bình: {avg_recall:.2f}")
    print(f"F1-Score trung bình: {avg_f1:.2f}")
 


def find_best_k(max_k):
    labels_train, images_train, labels_test, images_test = load_and_handle()

    # Tạo danh sách các giá trị K là số lẻ (1, 3, 5,...)
    k_values = range(1, max_k + 1, 2)
    accuracy_values = []  # Danh sách lưu độ chính xác

    # Tính độ chính xác cho từng giá trị K
    for k in k_values:
        n_correct = 0
        for i in range(len(images_test)):
            pred = predict_label(images_train, labels_train, images_test[i], k)
            if pred == labels_test[i]:
                n_correct += 1
        accuracy = n_correct / len(images_test)
        accuracy_values.append(accuracy * 100)  # Lưu độ chính xác theo %

    # Tìm K có độ chính xác cao nhất
    max_accuracy = max(accuracy_values)
    best_k = k_values[accuracy_values.index(max_accuracy)]

    # Vẽ đồ thị
    plt.plot(list(k_values), accuracy_values, marker='o')
    plt.xlabel('K')
    plt.ylabel('Accuracy (%)')
    plt.title('Elbow Method for KNN')
    plt.grid()
    plt.show()

    # In kết quả
    print(f'\nK tối ưu là {best_k} với độ chính xác {max_accuracy:.2f}%.\n')

if __name__ == "__main__":
    load_config()
    while True:
        key = input(
            f'\nNhập vào thao tác muốn thực hiện: \n1. Thiết lập số lượng train, test. \n2. Thiết lập K. \n3. Dự đoán ảnh. \n4. Tính độ chính xác. \n5. Tính Recall và F1-Score \n6. K tối ưu. \n7. Xem config.\n8. Thoát.\nLựa chọn của bạn: '
        ).strip()

        # Kiểm tra giá trị nhập vào
        if not key.isdigit():  # Nếu không phải số
            print("Lựa chọn không hợp lệ! Vui lòng nhập số từ 1 đến 8.")
            continue

        key = int(key)  # Chuyển đổi sang số nguyên

        # Xử lý các lựa chọn
        if key == 1:
            setNumberTrainAndTest()
        elif key == 2:
            setK()
        elif key == 3:
            predict()
        elif key == 4:
            cal_accuracy()
        elif key == 5:
            calculate_recall_and_f1()
        elif key == 6:
            max_k = int(input("Nhập giá trị tối đa của K: "))
            find_best_k(max_k)
        elif key == 7:
            print("NUM TRAIN: " + str(number_train))
            print("NUM TEST: " + str(number_test))
            print("K: " + str(k))
        elif key == 8:
            print("Thoát ứng dụng...")
            break
        else:
            print("Lựa chọn không hợp lệ! Vui lòng nhập số từ 1 đến 8.")

        