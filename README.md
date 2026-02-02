# Graph-based-Statute-Law-Retrieval-for-Legal-Question-Answering
Bài toán:
cho query và cần tìm các articles liên quan
Bth:
sẽ ko có mối quan hệ giữa các articles 
một article có thể sẽ refer hoặc được refer bởi một article khác
nên nếu ko có mối quan hệ giữa các articles thì có thể sẽ bị thiếu mất các articles liên quan
-> dùng graph (node articles, egde article refer article)
//kiểm chứng: trong gold label thì liệu các article chỉ có article được refer đến thôi, hay là có cả các article trước đó mà không refer
//nhưng: nếu biểu diễn lại article mà có nhiều article cạnh nó hơn thì liệu có được thêm nhiều thông tin hơn hay không
//có nên phân biệt các loại cạnh giữa các article không, có cạnh do đứng cạnh nhau, có cạnh do refer đến nhau
// xem các biến thể của graphsage: recommendation graphsage
// sử dụng bash
// chạy xem embedding nào tốt nhất (lấy 2 cái) -> chạy graph -> build lại structure