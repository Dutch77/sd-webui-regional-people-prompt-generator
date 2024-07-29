import launch

# TODO: add pip dependency if need extra module only on extension

if not launch.is_installed("jinja2"):
    launch.run_pip("install jinja2", live=True)
if not launch.is_installed("ultralytics"):
    launch.run_pip("install ultralytics", live=True)
if not launch.is_installed("opencv-python"):
    launch.run_pip("install opencv-python", live=True)
if not launch.is_installed("numpy"):
    launch.run_pip("install numpy", live=True)
if not launch.is_installed("torch"):
    launch.run_pip("install torch", live=True)
if not launch.is_installed("tf-keras"):
    launch.run_pip("install 'tf-keras<=2.15.1'", live=True)
if not launch.is_installed("deepface"):
    launch.run_pip("install deepface", live=True)
if not launch.is_installed("pillow"):
    launch.run_pip("install pillow", live=True)
if not launch.is_installed("fastapi"):
    launch.run_pip("install fastapi", live=True)