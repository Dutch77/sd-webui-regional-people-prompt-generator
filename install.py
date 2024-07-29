import launch

# TODO: add pip dependency if need extra module only on extension

if not launch.is_installed("jinja2"):
    launch.run_pip("install jinja2")
if not launch.is_installed("ultralytics"):
    launch.run_pip("install ultralytics")
if not launch.is_installed("opencv-python"):
    launch.run_pip("install opencv-python")
if not launch.is_installed("numpy"):
    launch.run_pip("install numpy")
if not launch.is_installed("torch"):
    launch.run_pip("install torch")
if not launch.is_installed("tf-keras"):
    launch.run_pip("install 'tf-keras<=2.15.1'")
if not launch.is_installed("deepface"):
    launch.run_pip("install deepface")
if not launch.is_installed("pillow"):
    launch.run_pip("install pillow")
if not launch.is_installed("fastapi"):
    launch.run_pip("install fastapi")