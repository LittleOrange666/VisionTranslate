import os
import tempfile
import traceback

import gradio as gr
import ollama

MODEL_NAME = os.getenv("MODEL_NAME","translategemma:12b")

host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

port = os.getenv("SERVER_PORT", "7860")

client = ollama.Client(host=host)

translate_prompt_template = """You are a professional {SOURCE_LANG} ({SOURCE_CODE}) to {TARGET_LANG} ({TARGET_CODE}) translator. Your goal is to accurately convey the meaning and nuances of the original {SOURCE_LANG} text while adhering to {TARGET_LANG} grammar, vocabulary, and cultural sensitivities.
Produce only the {TARGET_LANG} translation, without any additional explanations or commentary. Please translate the following {SOURCE_LANG} text into {TARGET_LANG}:


{TEXT}"""

SOURCE_LANG = os.getenv("SOURCE_LANG","English")
SOURCE_CODE = os.getenv("SOURCE_CODE","en")
TARGET_LANG = os.getenv("TARGET_LANG","Traditional Chinese")
TARGET_CODE = os.getenv("TARGET_CODE","zh-Hant")

SUPPORTED_LANGUAGES = [
    ("English", "en"),
    ("Traditional Chinese", "zh-Hant"),
    ("Japanese", "ja")
]


lang_choices = [f"{lang[0]} ({lang[1]})" for lang in SUPPORTED_LANGUAGES]

lang_map = {f"{lang[0]} ({lang[1]})":lang for lang in SUPPORTED_LANGUAGES}

def process_logic(image, manual_text, src_lang_name, tgt_lang_name):
    original_text = ""
    image_file = None
    try:
        src_name,src_code = lang_map.get(src_lang_name)
        tgt_name,tgt_code = lang_map.get(tgt_lang_name)
        if image is not None:
            image_file = tempfile.NamedTemporaryFile(suffix=".png",delete=False)
            image.save(image_file.name)
            image_file.close()
            ocr_prompt = f"Carefully transcribe all the {src_lang_name} text found in this image. Do not translate. Output the original text as is."
            ocr_response = client.chat(
                model=MODEL_NAME,
                messages=[{
                    'role': 'user',
                    'content': ocr_prompt,
                    'images': [image_file.name]
                }]
            )
            original_text = ocr_response['message']['content'].strip()

        elif manual_text and manual_text.strip():
            original_text = manual_text.strip()

        else:
            return "請提供圖片或輸入文字。", "請提供圖片或輸入文字。"

        # 進行翻譯
        translation_prompt = translate_prompt_template.format(SOURCE_CODE=src_code, SOURCE_LANG=src_name, TARGET_LANG=tgt_name, TARGET_CODE=tgt_code, TEXT=original_text)
        translation_response = client.chat(
            model=MODEL_NAME,
            messages=[{
                'role': 'user',
                'content': translation_prompt
            }]
        )
        translated_text = translation_response['message']['content'].strip()

        return original_text, translated_text

    except Exception as e:
        traceback.print_exception(e)
        return f"錯誤: {str(e)}", f"錯誤: {str(e)}"
    finally:
        if image_file is not None:
            os.remove(image_file.name)

with gr.Blocks(title="Gemma 3 進階翻譯器") as demo:
    gr.Markdown("# 🍊 TranslateGemma 多功能翻譯器 (進階版)")

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                src_lang = gr.Dropdown(choices=lang_choices, value=f"{SOURCE_LANG} ({SOURCE_CODE})", label="來源語言 (From)")
                tgt_lang = gr.Dropdown(choices=lang_choices, value=f"{TARGET_LANG} ({TARGET_CODE})", label="目標語言 (To)")

            with gr.Tabs():
                with gr.TabItem("圖片上傳"):
                    img_input = gr.Image(type="pil", label="上傳圖片")

                with gr.TabItem("手動輸入"):
                    txt_input = gr.Textbox(label="原文內容", lines=10, placeholder="在此輸入文字...")

            submit_btn = gr.Button("開始翻譯", variant="primary")

        with gr.Column(scale=3):
            out_original = gr.Textbox(label="偵測/提取的原文", lines=10, buttons=["copy"])
            out_translated = gr.Textbox(label="翻譯結果", lines=10, buttons=["copy"])

    submit_btn.click(
        fn=process_logic,
        inputs=[img_input, txt_input, src_lang, tgt_lang],
        outputs=[out_original, out_translated]
    )

if __name__ == "__main__":
    demo.launch(share=False,debug=True,server_name="0.0.0.0", server_port=int(port))