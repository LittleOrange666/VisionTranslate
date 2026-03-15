import os
import tempfile

import gradio as gr
import ollama

MODEL_NAME = os.getenv("MODEL_NAME","translategemma:12b")

host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

port = os.getenv("SERVER_PORT", "7860")

client = ollama.Client(host=host)

def process_image(image):
    if image is None:
        return "請先上傳圖片。", "請先上傳圖片。"
    image_file = tempfile.NamedTemporaryFile(suffix=".png",delete=False)
    image.save(image_file.name)
    image_file.close()

    try:
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
        
        if not original_text:
            return "無法從圖片中提取文字。", "無法進行翻譯。"

        translation_prompt = f"Translate the following text to Traditional Chinese: {original_text}"
        
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
        return f"發生錯誤: {str(e)}", f"發生錯誤: {str(e)}"
    finally:
        os.remove(image_file.name)

with gr.Blocks(title="Gemma 3 視覺翻譯器") as demo:
    gr.Markdown("# Gemma 3 圖片原文與翻譯提取")
    gr.Markdown("上傳一張包含文字的圖片，本程式將利用 `translategemma` 模型為你提取圖中的原文，並翻譯為繁體中文。")
    
    with gr.Row():
        with gr.Column(scale=2):
            image_input = gr.Image(type="pil", label="上傳圖片")
            submit_btn = gr.Button("開始處理", variant="primary")
            
        with gr.Column(scale=3):
            original_output = gr.Textbox(label="提取出的原文 (Original Text)", lines=8)
            translated_output = gr.Textbox(label="翻譯後的繁體中文 (Traditional Chinese)", lines=8)

    submit_btn.click(
        fn=process_image,
        inputs=image_input,
        outputs=[original_output, translated_output]
    )

if __name__ == "__main__":
    demo.launch(share=False,debug=True,server_name="0.0.0.0", server_port=int(port))