import os
import tempfile

import gradio as gr
import ollama
import PIL.Image
import io

# 設定你想使用的模型 (需確保 Ollama 已下載)
MODEL_NAME = "translategemma:12b" # 建議使用 27b 或 12b 效果較好

def process_image(image):
    """
    處理使用者上傳的圖片：
    1. 使用 translategemma 提取圖片中的所有文字 (原文)。
    2. 使用 translategemma 將提取出的文字翻譯為繁體中文。
    """
    if image is None:
        return "請先上傳圖片。", "請先上傳圖片。"
    image_file = tempfile.NamedTemporaryFile(suffix=".png",delete=False)
    image.save(image_file.name)
    image_file.close()

    try:
        # Step 1: 提取圖片文字 (OCR 任務)
        # 提示詞需要明確告訴模型只需提取，不需翻譯
        ocr_prompt = "Carefully transcribe all the text found in this image. Do not translate. Output the original text as is."
        
        # 呼叫 Ollama API 進行 OCR
        ocr_response = ollama.chat(
            model=MODEL_NAME,
            messages=[{
                'role': 'user',
                'content': ocr_prompt,
                'images': [image_file.name]
            }]
        )
        
        original_text = ocr_response['message']['content'].strip()
        
        if not original_text:
            return "無法從圖片中提取文字。", "無法進行翻譯。"

        # Step 2: 翻譯文字為繁體中文 (翻譯任務)
        # 這次我們只給模型文字，不給圖片，讓它專注於翻譯
        translation_prompt = f"Translate the following text to Traditional Chinese: {original_text}"
        
        translation_response = ollama.chat(
            model=MODEL_NAME,
            messages=[{
                'role': 'user',
                'content': translation_prompt
            }]
        )
        
        translated_text = translation_response['message']['content'].strip()
        
        return original_text, translated_text

    except Exception as e:
        return f"發生錯誤: {str(e)}", f"發生錯誤: {str(e)}"
    finally:
        os.remove(image_file.name)

# --- 構建 Gradio 介面 ---
with gr.Blocks(title="Gemma 3 視覺翻譯器") as demo:
    gr.Markdown("# Gemma 3 圖片原文與翻譯提取")
    gr.Markdown("上傳一張包含文字的圖片，本程式將利用 `translategemma` 模型為你提取圖中的原文，並翻譯為繁體中文。")
    
    with gr.Row():
        with gr.Column(scale=2):
            # 圖片輸入區
            image_input = gr.Image(type="pil", label="上傳圖片")
            # 執行按鈕
            submit_btn = gr.Button("開始處理", variant="primary")
            
        with gr.Column(scale=3):
            # 文字輸出區
            original_output = gr.Textbox(label="提取出的原文 (Original Text)", lines=8)
            translated_output = gr.Textbox(label="翻譯後的繁體中文 (Traditional Chinese)", lines=8)

    # 設定按鈕點擊事件
    submit_btn.click(
        fn=process_image,
        inputs=image_input,
        outputs=[original_output, translated_output]
    )

# --- 啟動服務 ---
if __name__ == "__main__":
    # share=True 可以生成一個公開的連結，供他人測試
    demo.launch(share=False)