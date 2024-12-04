# Handwritten-Digit-Classification-using-KNN
Kho lưu trữ này giới thiệu dự án "Handwritten-Digit-Classification" của tôi sử dụng Bộ dữ liệu MNIST. Dự án này được triển khai và thực hiện bằng cách áp dụng thuật toán KNN, có thể dự đoán nhãn của một hình ảnh với độ chính xác nhận dạng khoảng 90-92%. Đây là một ứng dụng đơn giản dành cho việc thực hành và nghiên cứu các kỹ thuật phân loại bằng KNN trong lĩnh vực xử lý ảnh.
  
## Tổng quan hệ thống
Hệ thống bao gồm các bước cơ bản:
- Xử lý dữ liệu: Tải và tiền xử lý các ảnh đầu vào.
- Phân loại bằng KNN: Tìm kiếm K lân cận gần nhất để dự đoán nhãn.
- Đánh giá: Tính độ chính xác, Recall, F1-Score, và tìm K tối ưu bằng Elbow Method.
- Cấu hình: Thiết lập số lượng ảnh huấn luyện, kiểm tra và giá trị K.
- Dự đoán nhãn của ảnh dựa trên dữ liệu huấn luyện.

Công nghệ sử dụng
- Python
- Các thư viện:
  + numpy: Xử lý dữ liệu số.
  + opencv-python: Xử lý ảnh.
  + matplotlib: Vẽ đồ thị.

## Cấu trúc dữ liệu
- config.dat: Lưu trữ cấu hình số lượng ảnh huấn luyện, kiểm tra và tham số K.
- Thư mục dataset:
  + training: Chứa các thư mục con đại diện cho từng nhãn (0-9), mỗi thư mục chứa ảnh dùng để huấn luyện.
  + testing: Chứa các thư mục tương tự nhưng dành cho kiểm tra.
- Thư mục pred_input: Chứa ảnh cần dự đoán nhãn.
- Mã nguồn chính: Chứa toàn bộ logic xử lý từ tải dữ liệu, huấn luyện đến dự đoán.

Ví dụ cấu trúc thư mục
```
Handwritten-Digit-Classification-using-KNN/
├── main.py                # Mã nguồn chính
├── config.dat             # File cấu hình (train, test, K)
├── dataset/
│   ├── training/          # Thư mục ảnh huấn luyện
│   └── testing/           # Thư mục ảnh kiểm tra
├── pred_input/            # Thư mục chứa ảnh cần dự đoán
├── README.md              # File README mô tả dự án
```

## Các tính năng
1. Thiết lập số lượng ảnh huấn luyện và kiểm tra.
2. Thiết lập giá trị K (số lượng lân cận).
3. Dự đoán nhãn của một hình ảnh.
4. Tính toán độ chính xác (Accuracy).
5. Tính Recall và F1-Score cho từng lớp nhãn.
6. Tìm giá trị K tối ưu bằng phương pháp Elbow Method.
7. Hiển thị cấu hình hiện tại.

## Hướng dẫn cài đặt
### Yêu cầu hệ thống
- Python 3.x
- Các thư viện:
  + numpy
  + opencv-python
  + matplotlib
### Cài đặt
1. Clone hoặc tải về mã nguồn của dự án.
```
git clone https://github.com/ntinh1704/Handwritten-Digit-Classification-using-KNN
cd Handwritten-Digit-Classification-using-KNN
```
3. Cài đặt các thư viện cần thiết bằng lệnh:
```
pip install numpy opencv-python matplotlib
```
3. Đảm bảo có các thư mục sau:
- dataset/training
- dataset/testing
- pred_input
4. Tạo file config.dat với nội dung:
```
1000
200
3
```
- 1000: Số lượng ảnh huấn luyện.
- 200: Số lượng ảnh kiểm tra.
- 3: Giá trị K mặc định.

## Hướng dẫn sử dụng
Chạy chương trình:
```
python main.py
```
Chọn các thao tác từ menu hiển thị:
1. Thiết lập số lượng ảnh huấn luyện và kiểm tra.
2. Thiết lập giá trị K.
3. Dự đoán nhãn cho một ảnh trong thư mục pred_input.
4. Tính toán độ chính xác trên tập kiểm tra.
5. Tính toán Recall và F1-Score cho từng nhãn.
6. Tìm giá trị K tối ưu.
7. Hiển thị cấu hình hiện tại.
8. Thoát ứng dụng.

## Kết quả
- Độ chính xác: Hiển thị tỷ lệ phần trăm ảnh được phân loại đúng.
- Recall và F1-Score: Hiển thị chi tiết cho từng lớp nhãn.
- K tối ưu: Hiển thị giá trị K cho độ chính xác cao nhất kèm đồ thị minh họa.
## Kết luận
Ứng dụng này minh họa cách sử dụng thuật toán KNN trong phân loại ảnh và các bước cần thiết để đánh giá mô hình. Người dùng có thể điều chỉnh tham số K, số lượng ảnh huấn luyện và kiểm tra để tối ưu hiệu suất. Đây là một công cụ hữu ích cho việc học tập và nghiên cứu về xử lý ảnh và phân loại bằng KNN.






