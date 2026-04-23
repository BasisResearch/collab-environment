# Import the InferencePipeline object
from inference import InferencePipeline
# Import the built in render_boxes sink for visualizing results
from inference.core.interfaces.stream.sinks import render_boxes

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("path_to_video", type=str)

    args = parser.parse_args()


    # initialize a pipeline object
    pipeline = InferencePipeline.init(
        model_id=args.model_id, # Roboflow model to use
        video_reference=args.path_to_video, # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
        on_prediction=render_boxes, # Function to run after each prediction
    )
    pipeline.start()
    pipeline.join()
