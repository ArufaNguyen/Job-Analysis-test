# Phân tích dữ liệu tuyển dụng (Job Data Analysis)
## Mô tả dự án

Dự án này nhằm phân tích dữ liệu tuyển dụng từ các file JSON, xử lý văn bản mô tả công việc, trích xuất các từ khóa quan trọng, tạo n-grams, và trực quan hóa mối quan hệ giữa các từ khóa bằng đồ thị NetworkX.

### Các bước chính bao gồm:

**Tiền xử lý dữ liệu**: loại bỏ **stopwords**, chuẩn hóa văn bản, tách token.

Lấy các **từ khóa quan trọng** dựa trên **tần suất (Frequency)** và **TF-IDF**.

Tạo **n-grams** từ các từ khóa và lọc các **n-grams** không cần thiết.

Xây dựng đồ thị **NetworkX** từ n-grams và trực quan hóa bằng **Matplotlib**.

#### Cấu trúc thư mục
```
project/
│
├─ main.py                  # File chính để chạy toàn bộ quy trình
├─ Module/
│   ├─ __init__.py
│   ├─ analysis.py.py       # Hàm scoring
│   ├─ preprocess.py          # Hàm tiền xử lí
│   └─ cluster.py.py       # Hàm vẽ đồ thị cluster
└─ Data/
    └─ output.json                # Dữ liệu tuyển dụng
```
## Hướng dẫn sử dụng

### Cài đặt các thư viện cần thiết:
```
pip install -r requirements.txt
```

### Chạy chương trình:
```
python main.py
```

### Kết quả:

Danh sách các từ khóa quan trọng dựa trên tần suất và TF-IDF.

Đồ thị trực quan hóa mối quan hệ giữa các từ khóa.

## Flowchart
### main.py
```mermaid
flowchart TD
    main["main.py"] --> preprocess_process["preprocess.process()"]
    main --> cluster_stacking["cluster.stacking()"]
    main --> cluster_draw_skidder["cluster.draw_skidder()"]

    cluster_stacking --> analysis_get_top_words["analysis.get_top_words()"]
    cluster_stacking --> analysis_scoring_hook["analysis.scoring_hook()"]
    cluster_stacking --> preprocess_stop_word["preprocess.stop_word()"]

    analysis_get_top_words --> preprocess_stop_word
```
### main.py

```mermaid
flowchart TD
    A((Start)) --> B["Get Top Words (Freq and TF-IDF)"]
    B --> C["Score Words"]
    C --> D["Generate N-grams from Top Words"]
    D --> E{"Filter N-grams?"}
    E -->|Yes| F["Remove Stopwords & Repeated Words"]
    E -->|No| G["Count N-gram Frequencies"]
    F --> G
    G --> H["Create NetworkX Graph"]
    H --> I["Add Edges from N-grams"]
    I --> J["Add Special Nodes"]
    J --> K["Visualize Graph (Matplotlib)"]
    K --> L((End))
```
### cluster.py

```mermaid
flowchart TD
    A((Start)) --> B["Get Top Words (Freq and TF-IDF)"]
    B --> C["Score Words"]
    C --> D["Generate N-grams from Top Words"]
    D --> F1["Filter N-grams (Stopwords, Unique Words)"]
    F1 --> G["Count N-gram Frequencies"]
    G --> F2["Filter N-grams (No Repeated Words)"]
    F2 --> H["Create NetworkX Graph"]
    H --> I["Add Edges from N-grams"]
    I --> J["Add Special Nodes"]
    J --> K["Visualize Graph (Matplotlib)"]
    K --> L((End))
```
### preprocess.py

```mermaid
flowchart TD
    Start((Start))

    subgraph "Process Function"
        direction LR
        P_Start["Load JSON Data"] --> P_Normalize["Normalize Job Data"]
        P_Normalize --> P_Loop_Desc{"For Each Job Description"}
        P_Loop_Desc --> P_Clean1["Clean Text (Initial)"]
        P_Clean1 --> P_Check_Len{"Text Length < 2?"}
        P_Check_Len -- Yes --> P_Loop_Desc
        P_Check_Len -- No --> P_Gen_Ngrams["Generate N-grams (CountVectorizer)"]
        P_Gen_Ngrams --> P_Extract_Feat["Extract N-gram Features"]
        P_Extract_Feat --> P_Remove_Stop_Ngrams["Remove Stopwords from N-grams"]
        P_Remove_Stop_Ngrams --> P_Join_Ngrams["Join Filtered N-grams"]
        P_Join_Ngrams --> P_Clean2["Clean Text (Second Pass)"]
        P_Clean2 --> P_Remove_Stop2["Remove Stopwords (Second Pass)"]
        P_Remove_Stop2 --> P_Correct_Repeat["Correct Repeated Tokens"]
        P_Correct_Repeat --> P_Collect_Tokens["Collect Final Tokens"]
        P_Collect_Tokens --> P_Loop_Desc
        P_Loop_Desc -- End Loop --> P_Return["Return Processed Tokens"]
    end

    Start --> P_Start
    P_Return --> Get_Top_Words["Get Top Words (Freq and TF-IDF)"]
    Get_Top_Words --> Score_Words["Score Words"]
    Score_Words --> Gen_Ngrams_Top["Generate N-grams from Top Words"]
    Gen_Ngrams_Top --> Filter_Ngrams_1["Filter N-grams (Stopwords, Unique Words)"]
    Filter_Ngrams_1 --> Count_Ngrams["Count N-gram Frequencies"]
    Count_Ngrams --> Filter_Ngrams_2["Filter N-grams (No Repeated Words)"]
    Filter_Ngrams_2 --> Create_Graph["Create NetworkX Graph"]
    Create_Graph --> Add_Edges["Add Edges from N-grams"]
    Add_Edges --> Add_Special_Nodes["Add Special Nodes"]
    Add_Special_Nodes --> Visualize_Graph["Visualize Graph (Matplotlib)"]
    Visualize_Graph --> End((End))

```

# https://github.com/ArufaNguyen/Job-Analysis-test
### Assigment E-learning Finaly Áp dụng các kỹ thuật xử lý NLP, trực quan hoá dữ liệu. Tìm ra xu hướng nghành nghề CNTT, data. Làm việc theo nhóm.
## UTH 012012410014 - Ngôn ngữ python - Nhóm 9