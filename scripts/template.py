import modules.scripts as base_scripts
import gradio as gr
from modules.processing import process_images, Processed
from scripts.mask_and_analysis_generator import MaskAndAnalysisGenerator, load_image
import scripts.prompt_generator as prompt_generator
from modules import codeformer_model


class ExtensionTemplateScript(base_scripts.Script):
    # Extension title in menu UI
    def title(self):
        return "Regional people prompt generator"

    # Decide to show menu in txt2img or img2img
    # - in "txt2img" -> is_img2img is `False`
    # - in "img2img" -> is_img2img is `True`
    #
    # below code always show extension menu
    def show(self, is_img2img):
        return base_scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion('Regional people prompt generator', open=False):
            with gr.Row():
                self.original_image = gr.Image(
                    label="Original image",
                    tool="editor",
                    source="upload",
                    height=400
                )
                self.prompt_template = gr.Textbox(label="Regional prompt template",
                                                  value=prompt_generator.get_default_prompt_template())
            with gr.Row():
                self.send_text_button = gr.Button(value="Generate mask and regional prompt", variant='primary')
            with gr.Row():
                self.enhance_photo = gr.Checkbox(
                    label="Enhance photo",
                    value=True
            )
            with gr.Row():
                self.generated_prompt = gr.Textbox(label="Generated regional prompt", interactive=False, lines=15,
                                                   show_copy_button=True)
            with gr.Row():
                self.generated_mask_image = gr.Image(
                    label="Generated mask image",
                    tool="editor",
                    source="upload",
                    height=400
                )
            self.send_text_button.click(fn=self.compute, inputs=[self.prompt_template, self.original_image, self.enhance_photo],
                                        outputs=[self.generated_prompt, self.generated_mask_image])

        return [self.original_image, self.prompt_template]

    def compute(self, prompt_template, image, enhance_photo):
        if enhance_photo:
            print('Enhancing photo')
            image = load_image(image)
            image = codeformer_model.codeformer.restore(image, w=1)

        generator = MaskAndAnalysisGenerator()
        mask, analysis = generator.process(image)
        rendered_prompt = prompt_generator.generate_prompt(prompt_template, analysis)

        return rendered_prompt, mask

    # Extension main process
    # Type: (StableDiffusionProcessing, List<UI>) -> (Processed)
    # args is [StableDiffusionProcessing, UI1, UI2, ...]
    def run(self, p):
        # TODO: get UI info through UI object
        proc = process_images(p)
        # TODO: add image edit process via Processed object proc
        return proc
