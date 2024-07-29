import launch

# TODO: add pip dependency if need extra module only on extension

if not launch.is_installed("jinja2"):
    launch.run_pip("install jinja2", desc="Installing jinja2", live=True)
if not launch.is_installed("ultralytics"):
    launch.run_pip("install ultralytics", desc="Installing ultralytics", live=True)
if not launch.is_installed("opencv-python"):
    launch.run_pip("install opencv-python", desc="Installing opencv-python", live=True)
if not launch.is_installed("numpy"):
    launch.run_pip("install numpy", desc="Installing numpy", live=True)
if not launch.is_installed("torch"):
    launch.run_pip("install torch", desc="Installing torch", live=True)
if not launch.is_installed("tf-keras"):
    launch.run_pip("install 'tf-keras<=2.15.1'", desc="Installing keras", live=True)
if not launch.is_installed("deepface"):
    launch.run_pip("install deepface", desc="Installing deepface", live=True)
if not launch.is_installed("pillow"):
    launch.run_pip("install pillow", desc="Installing pillow", live=True)
if not launch.is_installed("fastapi"):
    launch.run_pip("install fastapi", desc="Installing fastapi", live=True)