from fastapi import FastAPI, Body
from concurrent.futures import ThreadPoolExecutor
from modules import scripts, shared, codeformer_model
from modules.api import api
import gradio as gr
from scripts.mask_and_analysis_generator import MaskAndAnalysisGenerator, load_image
import scripts.prompt_generator as prompt_generator
from PIL import Image


def human_regional_prompter_api(_: gr.Blocks, app: FastAPI):
    app.state.executor = ThreadPoolExecutor(max_workers=8)

    @app.post("/regional-people-prompt-generator/generate")
    async def rppg_generate(
            image: str = Body("", title="Image"),
            prompt_template: str = Body(prompt_generator.get_default_prompt_template(), title="Prompt template"),
            enhance_photo: bool = Body(True, title="Enhance photo")
    ):
        if enhance_photo:
            print('Enhancing photo')
            image = load_image(image)
            image = codeformer_model.codeformer.restore(image, w=1)
        generator = MaskAndAnalysisGenerator()
        mask_image, analysis = generator.process(image)
        rendered_prompt = prompt_generator.generate_prompt(prompt_template, analysis)

        return {"mask_image": api.encode_pil_to_base64(Image.fromarray(mask_image)), "rendered_prompt": rendered_prompt}


try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(human_regional_prompter_api)
except:
    pass
