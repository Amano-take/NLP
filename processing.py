import re

# ダウンロード済みのtext8ファイルのパス
text8_file_path = 'enwik8'

# ファイルを読み込む
with open(text8_file_path, 'r', encoding='utf-8') as file:
    text8_data = file.read()

# ピリオドで区切り、文を抽出
sentences = re.split(r'\.', text8_data)

# 最初の1000文を抽出
extracted_sentences = sentences[:1000]

# 出力ファイルのパス
output_file_path = 'extracted_sentences.txt'

# 抽出した文をファイルに保存
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for sentence in extracted_sentences:
        output_file.write(sentence.strip())

print(f"{len(extracted_sentences)} sentences have been extracted and saved to {output_file_path}")
