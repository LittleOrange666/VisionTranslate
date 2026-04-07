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
TARGET_CODE = os.getenv("TARGET_CODE","zh-hant")

def process_logic(image, manual_text):
    original_text = ""
    image_file = None
    try:
        if image is not None:
            image_file = tempfile.NamedTemporaryFile(suffix=".png",delete=False)
            image.save(image_file.name)
            image_file.close()
            ocr_prompt = "Carefully transcribe all the text found in this image. Do not translate. Output the original text as is."
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
        translation_prompt = translate_prompt_template.format(SOURCE_CODE=SOURCE_CODE, SOURCE_LANG=SOURCE_LANG, TARGET_LANG=TARGET_LANG, TARGET_CODE=TARGET_CODE, TEXT=original_text)
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


with gr.Blocks(title="Gemma 3 翻譯工具") as demo:
    gr.Markdown("# 🍊 TranslateGemma 多功能翻譯器")

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("圖片翻譯"):
                    img_input = gr.Image(type="pil", label="上傳包含文字的圖片")

                with gr.TabItem("手動輸入"):
                    txt_input = gr.Textbox(label="請輸入欲翻譯的原文", lines=10, placeholder="在此輸入文字...")

            submit_btn = gr.Button("開始處理", variant="primary")

        with gr.Column(scale=3):
            out_original = gr.Textbox(label="提取/輸入的原文", lines=8, buttons=["copy"])
            out_translated = gr.Textbox(label="繁體中文譯文", lines=8, buttons=["copy"])

    submit_btn.click(
        fn=process_logic,
        inputs=[img_input, txt_input],
        outputs=[out_original, out_translated]
    )

if __name__ == "__main__":
    demo.launch(share=False,debug=True,server_name="0.0.0.0", server_port=int(port))