import json

# def convert_txt_to_json(txt_file_path, json_file_path):
#     with open(txt_file_path, 'r', encoding='utf-8') as txt_file:
#         data = json.load(txt_file)  # Reads JSON from the .txt file
#
#     with open(json_file_path, 'w', encoding='utf-8') as json_file:
#         json.dump(data, json_file, ensure_ascii=False, indent=4)
#
# # Example usage
# convert_txt_to_json('bns_and_ipc.txt', 'law_data.json')
# convert_txt_to_json('crpc_and_bnss.txt', 'law_data.json')
# convert_txt_to_json('iea_and_bsa.txt', 'law_data.json')

# def clean(json_file_path: str, output_file_path: str):
#     with open(json_file_path) as json_file:
#         data = json.load(json_file)
#
#     cleaned_data = []
#     for item in data:
#         cleaned_data.append({
#             "id": item.get("id"),
#             "bsa_section": item.get("bsa_section", "").strip(),
#             "iea_section": item.get("iea_section", "").strip(),
#             "subject": item.get("subject", "").strip(),
#             "summary": item.get("summary", "").strip(),
#             "extra_data": item.get("extra_data", {})
#         })
#
#     with open(output_file_path, 'w') as json_file:
#         json.dump(cleaned_data, json_file, indent=4)
#
# clean('law_data.json', 'cleaned_law_data.json')